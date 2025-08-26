from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from objectD_PostureD import detect_objects, analyze_posture, combined_detection
import base64

app = FastAPI()

class ObjectDetectionRequest(BaseModel):
    image_base64: str
    confidence_threshold: float = 0.5

class PostureAnalysisRequest(BaseModel):
    image_base64: str

class CombinedDetectionRequest(BaseModel):
    image_base64: str
    confidence_threshold: float = 0.5

@app.post("/object-detection")
def object_detection(req: ObjectDetectionRequest):
    try:
        return detect_objects(req.image_base64, req.confidence_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/object-detection/upload")
async def object_detection_upload(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, description="Minimum confidence for detections")
):
    try:
        file_content = await file.read()
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        result = detect_objects(image_base64, confidence_threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/posture-analysis")
def posture_analysis(req: PostureAnalysisRequest):
    try:
        return analyze_posture(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/posture-analysis/upload")
async def posture_analysis_upload(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        result = analyze_posture(image_base64)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))