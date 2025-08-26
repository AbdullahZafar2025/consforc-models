from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sent_and_image import generate_image

app = FastAPI()

class ImageRequest(BaseModel):
    prompt: str

@app.post("/text-to-image")
def text_to_image(req: ImageRequest):
    return generate_image(req.prompt)