from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from t2v import generate_video, get_video_model_info

app = FastAPI(title="Video Generation API", version="1.0.0")

class VideoGenerationRequest(BaseModel):
    prompt: str
    num_frames: int = 8
    num_inference_steps: int = 10
    height: int = 256
    width: int = 256
    guidance_scale: float = 7.0
    fps: int = 4

@app.post("/generate-video")
def generate_video_endpoint(req: VideoGenerationRequest):
    """
    Generate a video from text prompt.
    
    Request body:
    - prompt: Text description of the video to generate
    - num_frames: Number of frames (4-16, default: 8)
    - num_inference_steps: Quality vs speed (5-20, default: 10)  
    - height: Video height in pixels (128-512, default: 256)
    - width: Video width in pixels (128-512, default: 256)
    - guidance_scale: How closely to follow prompt (1-15, default: 7.0)
    - fps: Frames per second (2-10, default: 4)
    
    Returns:
    - video_base64: Base64 encoded MP4 video
    - prompt: The prompt used
    - duration_seconds: Video duration
    - resolution: Video resolution
    - model: Model used
    """
    try:
        return generate_video(
            prompt=req.prompt,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            height=req.height,
            width=req.width,
            guidance_scale=req.guidance_scale,
            fps=req.fps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-info")
def video_model_info():
    """
    Get information about the video generation model.
    
    Returns information about model capabilities, supported formats, etc.
    """
    try:
        return get_video_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))