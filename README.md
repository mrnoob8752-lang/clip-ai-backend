# CLIP AI Backend (MVP)

FastAPI backend for **CLIP AI** on a ₹0 budget using the Google Gemini API (`google-genai`) and FFmpeg.

## Endpoints

- `GET /health` — Health check and active MVP limits.
- `POST /api/clip` (`multipart/form-data`) — Accepts a required video upload (`file`, up to 50 MB and 180 seconds) and an optional text hint (`focus`). Returns a ZIP file (`clip_ai_clips.zip`) containing generated `.mp4` clips (2 to 60 seconds each, up to 5 clips) plus `highlights.json`.

## Environment Variables

- `GEMINI_API_KEY` (**Required**): Your Google Gemini API key. Store it only in Render's environment variables—never in GitHub or frontend code.
- `GEMINI_MODEL` (Optional): Defaults to `gemini-2.5-flash`.

## Render Free Tier Notes

- **Deployment & health check URL:** After creating your Web Service on Render, your live URL will follow `https://<your-service-name>.onrender.com` (for example, `https://clip-ai-backend.onrender.com`). Check `https://<your-service-name>.onrender.com/health` in your browser to confirm `{"status":"ok",...}`.
- **Cold starts:** Render free web services spin down after inactivity; the first request after sleep can take ~30–60 seconds to wake up.
- **Resource limits:** Free instances have limited RAM and CPU. Videos are capped at **50 MB** and **180 seconds**, clips are capped at **5 clips** of **2–60 seconds** each, and FFmpeg uses `-preset veryfast` to keep CPU/memory usage manageable.
- **Temporary storage & rate limits:** Temporary local files and Gemini Files API uploads are cleaned up after processing. Rate limiting (5 requests per 10 minutes) is stored in memory and resets on restart/spin-down.
