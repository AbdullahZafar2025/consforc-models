# model5.py
import torch
import tempfile
import os
import base64
from typing import Dict, Any
import imageio
import numpy as np
from diffusers import DiffusionPipeline
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Load video generation model ONCE ----------
print("Loading video generation model...")

try:
    # Load the video diffusion model
    video_pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
        torch_dtype=torch.float32,
        safety_checker=None,  # Disable safety checker for speed
        requires_safety_checker=False
    )
    
    # Move to CPU (GPU would be faster if available)
    video_pipe.to("cpu")
    
    # Optimize for CPU usage
    torch.set_num_threads(4)  # Adjust based on your CPU
    
    print("Video generation model loaded successfully!")
    MODEL_LOADED = True
    
except Exception as e:
    print(f"Failed to load video model: {e}")
    print("Video generation will not be available.")
    MODEL_LOADED = False
    video_pipe = None

# ---------- Helper Functions ----------

def _validate_prompt(prompt: str) -> str:
    """Validate and clean the prompt."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    # Clean the prompt
    cleaned_prompt = prompt.strip()
    
    # Limit prompt length to prevent issues
    if len(cleaned_prompt) > 200:
        cleaned_prompt = cleaned_prompt[:200] + "..."
    
    return cleaned_prompt

def _save_video_as_base64(frames, fps: int = 4) -> str:
    """Save video frames as base64 encoded MP4."""
    try:
        # Create temporary file for video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        # Convert frames to numpy array if needed
        if isinstance(frames, list):
            frames = np.array(frames)
        
        # Ensure frames are in the correct format (uint8)
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)
        
        # Save as MP4
        imageio.mimsave(temp_filename, frames, fps=fps, codec='libx264')
        
        # Read the video file and encode as base64
        with open(temp_filename, 'rb') as video_file:
            video_data = video_file.read()
            video_base64 = base64.b64encode(video_data).decode('utf-8')
        
        # Clean up temporary file
        os.remove(temp_filename)
        
        return video_base64
        
    except Exception as e:
        raise ValueError(f"Failed to encode video: {str(e)}")

# ---------- Core Video Generation Function ----------

def generate_video(
    prompt: str,
    num_frames: int = 8,
    num_inference_steps: int = 10,
    height: int = 256,
    width: int = 256,
    guidance_scale: float = 7.0,
    fps: int = 4
) -> Dict[str, Any]:
    """
    Generate a video from text prompt.
    
    Args:
        prompt: Text description of the video
        num_frames: Number of frames to generate (affects video length)
        num_inference_steps: Quality vs speed tradeoff (lower = faster)
        height: Video height in pixels
        width: Video width in pixels  
        guidance_scale: How closely to follow the prompt (higher = more faithful)
        fps: Frames per second for output video
    
    Returns:
        Dictionary containing video data and metadata
    """
    
    if not MODEL_LOADED:
        return {
            "error": "Video generation model failed to load",
            "video_base64": None,
            "model_loaded": False
        }
    
    try:
        # Validate inputs
        cleaned_prompt = _validate_prompt(prompt)
        
        # Clamp parameters to reasonable ranges
        num_frames = max(4, min(16, num_frames))  # 4-16 frames
        num_inference_steps = max(5, min(20, num_inference_steps))  # 5-20 steps
        height = max(128, min(512, height))  # 128-512 pixels
        width = max(128, min(512, width))   # 128-512 pixels
        guidance_scale = max(1.0, min(15.0, guidance_scale))  # 1-15
        fps = max(2, min(10, fps))  # 2-10 fps
        
        print(f"Generating video: '{cleaned_prompt}' ({num_frames} frames, {num_inference_steps} steps)")
        
        # Generate the video
        result = video_pipe(
            prompt=cleaned_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale
        )
        
        # Extract frames
        frames = result.frames[0]  # Get the first (and only) video
        
        # Convert to base64
        video_base64 = _save_video_as_base64(frames, fps)
        
        # Calculate video duration
        duration_seconds = num_frames / fps
        
        return {
            "video_base64": video_base64,
            "prompt": cleaned_prompt,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": duration_seconds,
            "resolution": f"{width}x{height}",
            "model": "zeroscope_v2_576w",
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width
            },
            "video_format": "mp4",
            "model_loaded": True
        }
        
    except Exception as e:
        return {
            "error": f"Video generation failed: {str(e)}",
            "prompt": prompt,
            "video_base64": None,
            "model_loaded": MODEL_LOADED
        }

def get_video_model_info() -> Dict[str, Any]:
    """Get information about the video generation model."""
    return {
        "model_name": "cerspense/zeroscope_v2_576w" if MODEL_LOADED else "Not Loaded",
        "model_loaded": MODEL_LOADED,
        "supported_resolutions": ["128x128", "256x256", "512x512"],
        "max_frames": 16,
        "recommended_fps": [2, 4, 8],
        "output_format": "mp4 (base64 encoded)",
        "device": "cpu",
        "features": [
            "Text-to-video generation",
            "Adjustable quality/speed",
            "Multiple resolutions",
            "Base64 output for API"
        ]
    }