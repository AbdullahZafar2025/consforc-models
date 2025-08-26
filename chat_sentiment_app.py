from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sent_and_image import analyze_sentiment
from ChatBot import chat

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str

@app.post("/sentiment-analysis")
def sentiment(req: SentimentRequest):
    return analyze_sentiment(req.text)

@app.post("/chat")
def chat_with_bot(req: ChatRequest):
    try:
        response = chat(req.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))