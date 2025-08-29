from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Response
from pydantic import BaseModel
from Translation_transcription_audioGen import transcribe_and_translate_to_target, text_to_audio, text_to_audio_base64
import os
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



class TextToAudioRequest(BaseModel):
    text: str
    speaker_id: int = 0

@app.post("/VocalSyncAI")
async def transcribe_translate(
    file: UploadFile = File(...),
    target: str = Query("en", description="Target language code, e.g. 'es', 'fr', 'en'")
):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        result = transcribe_and_translate_to_target(file_location, target)

        if os.path.exists(file_location):
            os.remove(file_location)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-audio")
def text_to_speech_json(req: TextToAudioRequest):
    try:
        return text_to_audio_base64(req.text, req.speaker_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-audio/stream", response_class=Response)
def text_to_speech_stream(req: TextToAudioRequest):
    try:
        audio_bytes = text_to_audio(req.text, req.speaker_id)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3",
                "Content-Length": str(len(audio_bytes))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
