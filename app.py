from fastapi import FastAPI, Response
from pydantic import BaseModel
from sent_and_image import analyze_sentiment, generate_image

from fastapi import FastAPI, UploadFile, File
from sent_and_image import analyze_sentiment
import os
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Response
from Translation_transcription_audioGen import transcribe_and_translate_to_target, text_to_audio, text_to_audio_base64
from objectD_PostureD import detect_objects, analyze_posture, combined_detection
from ChatBot import chat

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    prompt: str

class TextToAudioRequest(BaseModel):
    text: str
    speaker_id: int = 0  # Optional speaker ID (0-109 for SpeechT5)

class ObjectDetectionRequest(BaseModel):
    image_base64: str
    confidence_threshold: float = 0.5  # Minimum confidence for detections

class PostureAnalysisRequest(BaseModel):
    image_base64: str

class CombinedDetectionRequest(BaseModel):
    image_base64: str
    confidence_threshold: float = 0.5

class ChatRequest(BaseModel):
    message: str

@app.post("/sentiment-analysis")
def sentiment(req: SentimentRequest):
    return analyze_sentiment(req.text)

@app.post("/text-to-image")
def text_to_image(req: ImageRequest):
    return generate_image(req.prompt)

@app.post("/transcribe-translate")
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
    """
    Convert text to speech and return as base64 JSON.
    
    Request body:
    - text: The text to convert to speech
    - speaker_id: Optional speaker voice ID (not used in gTTS)
    
    Returns:
    - text: Original text
    - audio_base64: Base64-encoded MP3 audio
    - sample_rate: Audio sample rate
    - format: Audio format (mp3)
    - model: TTS model used
    """
    try:
        return text_to_audio_base64(req.text, req.speaker_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-audio/stream", response_class=Response)
def text_to_speech_stream(req: TextToAudioRequest):
    """
    Convert text to speech and return audio file directly (streamable/previewable).
    Perfect for Postman preview or direct audio streaming.
    
    Request body:
    - text: The text to convert to speech
    - speaker_id: Optional speaker voice ID (not used in gTTS)
    
    Returns: Direct MP3 audio file
    """
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

@app.post("/object-detection")
def object_detection(req: ObjectDetectionRequest):
    """
    Detect objects in an image using DETR (JSON with base64).
    
    Request body:
    - image_base64: Base64 encoded image
    - confidence_threshold: Minimum confidence for detections (0.0-1.0, default: 0.5)
    
    Returns:
    - detections: List of detected objects with bounding boxes
    - total_objects: Number of objects detected
    - annotated_image_base64: Image with bounding boxes drawn
    - model: Model used (DETR-ResNet50)
    """
    try:
        return detect_objects(req.image_base64, req.confidence_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/object-detection/upload")
async def object_detection_upload(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, description="Minimum confidence for detections")
):
    """
    Detect objects in an uploaded image file.
    
    Form Data:
    - file: Image file (JPEG, PNG, etc.)
    - confidence_threshold: Minimum confidence for detections (query parameter)
    
    Returns: Same as /object-detection but accepts file upload
    """
    try:
        # Read the uploaded file
        file_content = await file.read()
        
        # Convert file to base64
        import base64
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Run detection
        result = detect_objects(image_base64, confidence_threshold)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_bot(req: ChatRequest):
    """
    Simple chat with AI bot.
    Just send a message, get a response.
    """
    try:
        response = chat(req.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/posture-analysis")
def posture_analysis(req: PostureAnalysisRequest):
    """
    Analyze posture in an image using basic computer vision (JSON with base64).
    
    Request body:
    - image_base64: Base64 encoded image
    
    Returns:
    - person_detected: Whether a person was found
    - posture_analysis: Detailed posture scores and recommendations
    - landmarks: Body pose landmarks
    - annotated_image_base64: Image with pose landmarks drawn
    - model: Model used (OpenCV_Basic_Analysis)
    """
    try:
        return analyze_posture(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/posture-analysis/upload")
async def posture_analysis_upload(file: UploadFile = File(...)):
    """
    Analyze posture in an uploaded image file.
    
    Form Data:
    - file: Image file (JPEG, PNG, etc.)
    
    Returns: Same as /posture-analysis but accepts file upload
    """
    try:
        # Read the uploaded file
        file_content = await file.read()
        
        # Convert file to base64
        import base64
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Run posture analysis
        result = analyze_posture(image_base64)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

