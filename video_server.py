import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ViralIQ Video Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNWAY_API_KEY = os.environ.get("RUNWAY_API_KEY", "")
RUNWAY_BASE    = "https://api.dev.runwayml.com/v1"
RUNWAY_VERSION = "2024-11-06"

def runway_headers():
    return {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "Content-Type": "application/json",
        "X-Runway-Version": RUNWAY_VERSION,
    }

# ── Models ────────────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    prompt: str
    duration: int = 5          # 5 or 10 seconds
    ratio: str = "768:1280"    # 768:1280 = vertical (Instagram Reels)
    model: str = "gen3a_turbo" # gen3a_turbo = faster, gen3a = higher quality

class TaskStatusResponse(BaseModel):
    taskId: str
    status: str                # PENDING | RUNNING | SUCCEEDED | FAILED
    videoUrl: str | None = None
    progress: int = 0          # 0-100

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    has_key = bool(RUNWAY_API_KEY)
    return {"status": "ok", "runway_key_set": has_key}


@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    """
    Kicks off a Runway Gen-3 text-to-video task.
    Returns taskId immediately — poll /video-status/{taskId} for results.
    """
    if not RUNWAY_API_KEY:
        raise HTTPException(500, "RUNWAY_API_KEY environment variable not set")

    payload = {
        "model": req.model,
        "promptText": req.prompt,
        "duration": req.duration,
        "ratio": req.ratio,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            f"{RUNWAY_BASE}/text_to_video",
            headers=runway_headers(),
            json=payload,
        )

    if res.status_code not in (200, 201):
        raise HTTPException(res.status_code, f"Runway API error: {res.text}")

    data = res.json()
    task_id = data.get("id")
    if not task_id:
        raise HTTPException(500, f"No task ID returned: {data}")

    return {"taskId": task_id, "status": "PENDING"}


@app.get("/video-status/{task_id}")
async def video_status(task_id: str):
    """
    Polls Runway for task status.
    Frontend should call this every 5 seconds until status == SUCCEEDED or FAILED.
    """
    if not RUNWAY_API_KEY:
        raise HTTPException(500, "RUNWAY_API_KEY not set")

    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.get(
            f"{RUNWAY_BASE}/tasks/{task_id}",
            headers=runway_headers(),
        )

    if res.status_code != 200:
        raise HTTPException(res.status_code, f"Runway status error: {res.text}")

    data = res.json()
    status = data.get("status", "UNKNOWN")
    output = data.get("output", [])
    progress = data.get("progress", 0)

    video_url = output[0] if output and status == "SUCCEEDED" else None

    return TaskStatusResponse(
        taskId=task_id,
        status=status,
        videoUrl=video_url,
        progress=int((progress or 0) * 100),
    )


@app.delete("/cancel-video/{task_id}")
async def cancel_video(task_id: str):
    """Cancel a running task to save credits."""
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.delete(
            f"{RUNWAY_BASE}/tasks/{task_id}",
            headers=runway_headers(),
        )
    return {"cancelled": res.status_code in (200, 204), "taskId": task_id}
