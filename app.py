import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

# ---------------------------------------------------------------------------
# Configuration & Limits (tuned for Render Free Tier: 512 MB RAM, 0.1 vCPU)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Rate limit: 5 processing requests per 10 minutes (600 seconds)
RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 600

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_DURATION_SECONDS = 180.0  # 180 seconds (3 minutes)
MAX_CLIPS = 5
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 60.0
FFMPEG_TIMEOUT_SECONDS = 120.0
GEMINI_ACTIVE_TIMEOUT_SECONDS = 180.0

ALLOWED_ORIGINS = [
    "https://mrnoob8752-lang.github.io",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
]

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
    "video/x-msvideo",
    "application/octet-stream",
}

app = FastAPI(
    title="CLIP AI Backend",
    description="MVP backend that analyzes uploaded videos with Gemini, cuts clips with FFmpeg, and returns a ZIP archive.",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https://mrnoob8752-lang\.github\.io$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Clip-Count", "X-Video-Duration"],
)

# In-memory sliding window per client IP (resets on container restart/spin-down)
_rate_limit_buckets: Dict[str, Deque[float]] = defaultdict(deque)


# ---------------------------------------------------------------------------
# Structured Output Schema for Gemini
# ---------------------------------------------------------------------------
class ClipMoment(BaseModel):
    title: str = Field(..., description="Short descriptive title for the highlight moment")
    start_seconds: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_seconds: float = Field(..., gt=0.0, description="End timestamp in seconds")
    reason: str = Field(..., description="Explanation of why this moment is engaging")


class VideoAnalysisResult(BaseModel):
    moments: List[ClipMoment] = Field(
        default_factory=list,
        description="Up to 5 engaging moments ordered chronologically",
    )


# ---------------------------------------------------------------------------
# Rate Limiting (Render Proxy-Aware Client IP Strategy)
# ---------------------------------------------------------------------------
def get_render_safe_client_ip(request: Request) -> str:
    # Clients can spoof the leftmost entry in X-Forwarded-For.
    # When behind a single reverse proxy like Render, the last hop before the
    # server is more reliable than blindly trusting header[0], falling back to
    # request.client.host if the header is absent.
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        hops = [hop.strip() for hop in forwarded_for.split(",") if hop.strip()]
        if hops:
            return hops[-1]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_rate_limit(request: Request) -> None:
    client_ip = get_render_safe_client_ip(request)
    now = time.monotonic()
    bucket = _rate_limit_buckets[client_ip]

    while bucket and (now - bucket[0]) > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: maximum {RATE_LIMIT_MAX_REQUESTS} requests per "
                f"{RATE_LIMIT_WINDOW_SECONDS // 60} minutes. Please wait {retry_after} seconds."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GEMINI_API_KEY is not configured.",
        )
    return GEMINI_API_KEY


def sanitize_filename(name: str, fallback: str = "clip") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return cleaned[:40] or fallback


def validate_upload_metadata(upload: UploadFile) -> str:
    ext = Path(upload.filename or "upload.mp4").suffix.lower()
    if ext and ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Supported formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}."
            ),
        )

    if upload.content_type and upload.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{upload.content_type}'. Please upload a valid video file.",
        )

    return ext if ext in ALLOWED_VIDEO_EXTENSIONS else ".mp4"


def write_upload_stream_to_disk(file_obj, dest_path: Path) -> None:
    total_bytes = 0
    with dest_path.open("wb") as out_file:
        while True:
            chunk = file_obj.read(1024 * 256)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Video file exceeds the 50 MB limit (50 MB max).",
                )
            out_file.write(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")


async def save_uploaded_video(upload: UploadFile, dest_path: Path) -> None:
    try:
        await asyncio.to_thread(write_upload_stream_to_disk, upload.file, dest_path)
    finally:
        await upload.close()


def probe_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="FFprobe is not installed on the server.")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timed out while inspecting video metadata.")

    if proc.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail="Could not inspect video metadata. Please upload a valid, uncorrupted video file.",
        )

    try:
        duration = float(proc.stdout.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="Unable to determine video duration.")

    if duration <= 0:
        raise HTTPException(status_code=400, detail="Video has invalid or zero duration.")

    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Video duration ({duration:.1f}s) exceeds the 180-second limit.",
        )

    return duration


def clamp_clip_window(raw_start: float, raw_end: float, duration_seconds: float) -> Optional[tuple[float, float]]:
    if duration_seconds < MIN_CLIP_SECONDS:
        return None

    max_allowed_length = min(MAX_CLIP_SECONDS, duration_seconds)
    if max_allowed_length < MIN_CLIP_SECONDS:
        return None

    start = max(0.0, float(raw_start))
    end = max(start + MIN_CLIP_SECONDS, float(raw_end))

    length = end - start
    if length > MAX_CLIP_SECONDS:
        end = start + MAX_CLIP_SECONDS
        length = MAX_CLIP_SECONDS

    if end > duration_seconds:
        end = duration_seconds
        start = max(0.0, end - length)
        length = end - start

    if start < 0.0 or end > duration_seconds or length < MIN_CLIP_SECONDS or length > MAX_CLIP_SECONDS:
        return None

    return (round(start, 2), round(end, 2))


def analyze_video_with_gemini(
    video_path: Path,
    duration_seconds: float,
    focus: Optional[str] = None,
) -> List[ClipMoment]:
    api_key = require_gemini_key()
    client = genai.Client(api_key=api_key)

    uploaded_file = None
    try:
        uploaded_file = client.files.upload(file=str(video_path))

        # Wait until Gemini confirms the video is ACTIVE
        poll_start = time.monotonic()

        while True:
            state = getattr(uploaded_file, "state", None)
            state_name = getattr(state, "name", str(state) if state else "")

            if state_name == "ACTIVE":
                break

            if state_name == "FAILED":
                raise HTTPException(
                    status_code=422,
                    detail="Gemini failed to process the uploaded video.",
                )

            if time.monotonic() - poll_start > GEMINI_ACTIVE_TIMEOUT_SECONDS:
                raise HTTPException(
                    status_code=504,
                    detail="Gemini video processing timed out.",
                )

            time.sleep(5)
            uploaded_file = client.files.get(name=uploaded_file.name)

        focus_note = f" Focus specifically on: {focus.strip()}." if focus and focus.strip() else ""
        prompt = (
            f"Analyze this video (total duration: {duration_seconds:.2f} seconds). "
            f"Find up to {MAX_CLIPS} of the most engaging moments for short clips.{focus_note} "
            f"Every single clip MUST have a duration strictly between {MIN_CLIP_SECONDS:.0f} and {MAX_CLIP_SECONDS:.0f} seconds "
            f"(2.0s <= end_seconds - start_seconds <= 60.0s), with start_seconds >= 0, "
            f"end_seconds <= {duration_seconds:.2f}, and start_seconds < end_seconds. "
            "Avoid overlapping timestamps. Provide a short title, start_seconds, end_seconds, and reason for each moment."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoAnalysisResult,
                temperature=0.2,
            ),
        )

        raw_text = (response.text or "").strip()
        if not raw_text:
            raise HTTPException(status_code=502, detail="Gemini returned an empty response.")

        try:
            parsed = VideoAnalysisResult.model_validate_json(raw_text)
        except Exception:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()
            parsed = VideoAnalysisResult.model_validate(json.loads(cleaned))

        moments: List[ClipMoment] = []
        for item in parsed.moments[:MAX_CLIPS]:
            clamped = clamp_clip_window(item.start_seconds, item.end_seconds, duration_seconds)
            if clamped is not None:
                start, end = clamped
                moments.append(
                    ClipMoment(
                        title=item.title.strip() or "highlight",
                        start_seconds=start,
                        end_seconds=end,
                        reason=item.reason.strip() or "Engaging moment",
                    )
                )

        if not moments:
            fallback_window = clamp_clip_window(0.0, min(duration_seconds, 15.0), duration_seconds)
            if fallback_window is not None:
                start, end = fallback_window
                moments.append(
                    ClipMoment(
                        title="opening_highlight",
                        start_seconds=start,
                        end_seconds=end,
                        reason="Fallback highlight clip when no distinct moments were returned",
                    )
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Video is shorter than the minimum 2-second clip duration.",
                )

        return moments
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini video analysis failed: {exc}")
    finally:
        if uploaded_file is not None and getattr(uploaded_file, "name", None):
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def cut_mp4_clip(source_video: Path, output_clip: Path, start_sec: float, end_sec: float) -> None:
    clip_duration = end_sec - start_sec
    if clip_duration < MIN_CLIP_SECONDS or clip_duration > MAX_CLIP_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Clip duration ({clip_duration:.2f}s) must be between {MIN_CLIP_SECONDS:.0f} and {MAX_CLIP_SECONDS:.0f} seconds.",
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.2f}",
        "-i",
        str(source_video),
        "-t",
        f"{clip_duration:.2f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_clip),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="FFmpeg is not installed on the server.")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="FFmpeg timed out while cutting clips.")

    if proc.returncode != 0 or not output_clip.exists() or output_clip.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail=f"FFmpeg failed to produce MP4 clip for {start_sec:.2f}s - {end_sec:.2f}s.",
        )


def build_zip_archive(clips_dir: Path, zip_path: Path, moments: List[ClipMoment]) -> None:
    highlights = [
        {
            "clip_number": idx + 1,
            "title": m.title,
            "start_seconds": m.start_seconds,
            "end_seconds": m.end_seconds,
            "duration_seconds": round(m.end_seconds - m.start_seconds, 2),
            "explanation": m.reason,
        }
        for idx, m in enumerate(moments)
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for mp4_path in sorted(clips_dir.glob("*.mp4")):
            zf.write(mp4_path, arcname=mp4_path.name)
        zf.writestr("highlights.json", json.dumps(highlights, indent=2))


def cleanup_temp_dir(temp_dir: str) -> None:
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "clip-ai-backend",
        "gemini_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
        "limits": {
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_duration_seconds": MAX_DURATION_SECONDS,
            "min_clip_seconds": MIN_CLIP_SECONDS,
            "max_clip_seconds": MAX_CLIP_SECONDS,
            "max_clips": MAX_CLIPS,
            "rate_limit": f"{RATE_LIMIT_MAX_REQUESTS} per {RATE_LIMIT_WINDOW_SECONDS // 60} minutes",
        },
    }


@app.post("/api/clip", dependencies=[Depends(enforce_rate_limit)])
async def clip_video(
    file: UploadFile = File(...),
    focus: Optional[str] = Form(default=None),
):
    require_gemini_key()
    ext = validate_upload_metadata(file)

    temp_dir = tempfile.mkdtemp(prefix="clip_ai_")
    temp_path = Path(temp_dir)
    clips_dir = temp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = temp_path / "clip_ai_clips.zip"

    try:
        input_video = temp_path / f"source{ext}"
        await save_uploaded_video(file, input_video)

        duration_seconds = await asyncio.to_thread(probe_video_duration, input_video)
        moments = await asyncio.to_thread(
            analyze_video_with_gemini,
            input_video,
            duration_seconds,
            focus,
        )

        for idx, moment in enumerate(moments, start=1):
            slug = sanitize_filename(moment.title, fallback=f"moment_{idx}")
            clip_file = clips_dir / f"clip_{idx:02d}_{slug}.mp4"
            await asyncio.to_thread(
                cut_mp4_clip,
                input_video,
                clip_file,
                moment.start_seconds,
                moment.end_seconds,
            )

        await asyncio.to_thread(build_zip_archive, clips_dir, zip_path, moments)

        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename="clip_ai_clips.zip",
            headers={
                "X-Clip-Count": str(len(moments)),
                "X-Video-Duration": f"{duration_seconds:.2f}",
            },
            background=BackgroundTask(cleanup_temp_dir, temp_dir),
        )
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception as exc:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Unexpected error processing video: {exc}")
