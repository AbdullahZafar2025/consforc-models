from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import multiprocessing, os, psutil
from t2v import generate_video  # your real video function
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="VideoCraftAI", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],      # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],      # Allow all headers
)

# -----------------------------
# Request model
# -----------------------------
class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    num_frames: int = Field(8, ge=4, le=24)
    num_inference_steps: int = Field(10, ge=5, le=50)
    height: int = Field(256, ge=128, le=576)
    width: int = Field(256, ge=128, le=1024)
    guidance_scale: float = Field(7.0, ge=1.0, le=20.0)
    fps: int = Field(4, ge=1, le=30)

# -----------------------------
# Worker function (separate process)
# -----------------------------
def run_high_priority_job(req: VideoGenerationRequest):
    try:
        # Boost process priority
        p = psutil.Process(os.getpid())
        try:
            p.nice(-10)  # higher CPU priority (needs root for <0)
        except psutil.AccessDenied:
            print("⚠️ Running without root, priority not raised")

        # Run heavy video generation
        result = generate_video(
            prompt=req.prompt,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            height=req.height,
            width=req.width,
            guidance_scale=req.guidance_scale,
            fps=req.fps,
        )
        return {"status": "Completed", "result": result}

    except Exception as e:
        return {"status": "Failed", "error": str(e)}

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/VideoCraftAI")
def start_video(req: VideoGenerationRequest):
    try:
        # Run in a separate process so CPU resources are isolated
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            result = pool.apply(run_high_priority_job, (req,))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

