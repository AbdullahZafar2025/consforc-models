from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sent_and_image import analyze_sentiment
from ChatBot import chat
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS setup
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],   # allow all headers
)


class SentimentRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str

@app.post("/LiteText-AI")
def sentiment(req: SentimentRequest):
    return analyze_sentiment(req.text)

@app.post("/InstructFlowAI")
def chat_with_bot(req: ChatRequest):
    try:
        response = chat(req.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
