from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sent_and_image import generate_image

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],   # allow all headers
)

class ImageRequest(BaseModel):
    prompt: str

@app.post("/VisionaryAI")
def text_to_image(req: ImageRequest):
    return generate_image(req.prompt)

