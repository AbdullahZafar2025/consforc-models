from transformers import pipeline
from PIL import Image
import base64
import io
from diffusers import StableDiffusionPipeline
import torch, base64, io
from PIL import Image
# Sentiment Analysis (same as before)
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analyze_sentiment(text: str):
    return sentiment_pipeline(text)
######################################################
# load tiny SD for speed
device = "cpu"
image_pipeline = StableDiffusionPipeline.from_pretrained(
    "segmind/tiny-sd",
    torch_dtype=torch.float32
).to(device)
def generate_image(prompt: str):
    image = image_pipeline(prompt, num_inference_steps=15).images[0]  # 15 steps = fast
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"prompt": prompt, "image_base64": img_str}

