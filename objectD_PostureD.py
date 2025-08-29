# model3.py
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple
import requests
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection
import torch
import tempfile
import os

# ---------- Load models ONCE (module-level singletons) ----------

print("Loading DETR model for object detection...")
# Use DETR (Facebook's DEtection TRansformer) - works well with Python 3.13
object_detection_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
object_detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

print("Loading pose estimation model...")
# Use a simpler approach with pre-trained models that work with Python 3.13
# We'll use a human pose estimation pipeline
try:
    # Try to load a pose estimation model
    pose_pipeline = pipeline(
        "image-classification",
        model="microsoft/beit-base-patch16-224",  # We'll adapt this for pose analysis
        device=-1  # CPU
    )
    print("Alternative pose model loaded successfully!")
except Exception as e:
    print(f"Pose model loading failed, will use basic analysis: {e}")
    pose_pipeline = None

print("All compatible models loaded successfully!")

# ---------- Helper Functions ----------
def _decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image."""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL to OpenCV format (RGB to BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
        
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def _decode_base64_to_pil(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        return pil_image
        
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def _encode_image_to_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string."""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        
        # Encode to base64
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_string
        
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def _encode_pil_to_base64(pil_image: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    try:
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        
        # Encode to base64
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_string
        
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def _simple_posture_analysis(image: np.ndarray) -> Dict[str, Any]:
    """Simple posture analysis using basic computer vision techniques."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Haar cascades for face detection (built into OpenCV)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {
                "error": "No face detected in the image",
                "person_detected": False,
                "posture_score": None
            }
        
        # Basic analysis based on face position
        face_x, face_y, face_w, face_h = faces[0]  # Use first face
        image_center_x = image.shape[1] // 2
        image_center_y = image.shape[0] // 2
        
        # Calculate face center
        face_center_x = face_x + face_w // 2
        face_center_y = face_y + face_h // 2
        
        # Simple posture metrics
        head_tilt = abs(face_center_x - image_center_x) / image.shape[1] * 100
        head_position = (image_center_y - face_center_y) / image.shape[0] * 100
        
        # Basic scoring
        tilt_score = max(0, 100 - head_tilt * 2)
        position_score = max(0, 100 - abs(head_position) * 2)
        overall_score = (tilt_score + position_score) / 2
        
        # Determine quality
        if overall_score >= 80:
            quality = "Good"
        elif overall_score >= 60:
            quality = "Fair"
        else:
            quality = "Needs Improvement"
        
        return {
            "overall_score": round(overall_score, 1),
            "head_tilt_score": round(tilt_score, 1),
            "head_position_score": round(position_score, 1),
            "posture_quality": quality,
            "face_detected": True,
            "recommendations": [
                "Keep your head centered" if head_tilt > 20 else "Good head alignment",
                "Adjust your sitting position" if abs(head_position) > 30 else "Good posture"
            ]
        }
        
    except Exception as e:
        return {
            "error": f"Posture analysis failed: {str(e)}",
            "person_detected": False,
            "overall_score": 0
        }

# ---------- Core Detection Functions ----------

def detect_objects(image_base64: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Detect objects in image using DETR model.
    
    Args:
        image_base64: Base64 encoded image
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        Dictionary containing detection results and annotated image
    """
    try:
        # Decode image to PIL
        pil_image = _decode_base64_to_pil(image_base64)
        original_width, original_height = pil_image.size
        
        # Prepare image for DETR
        inputs = object_detection_processor(images=pil_image, return_tensors="pt")
        
        # Run detection
        with torch.no_grad():
            outputs = object_detection_model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([pil_image.size[::-1]])  # (height, width)
        results = object_detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        
        # Parse detections
        detections = []
        annotated_image = pil_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidence = score.item()
            class_id = label.item()
            class_name = object_detection_model.config.id2label[class_id]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            
            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1, "height": y2 - y1
                }
            })
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=1)
            
            # Add label
            label_text = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1 - 20), label_text, fill="red")
        
        # Encode annotated image
        annotated_base64 = _encode_pil_to_base64(annotated_image)
        
        return {
            "detections": detections,
            "total_objects": len(detections),
            "image_dimensions": {"width": original_width, "height": original_height},
            "annotated_image_base64": annotated_base64,
            "model": "DETR-ResNet50",
            "confidence_threshold": confidence_threshold
        }
        
    except Exception as e:
        raise ValueError(f"Object detection failed: {str(e)}")

def analyze_posture(image_base64: str) -> Dict[str, Any]:
    """
    Analyze posture using basic computer vision techniques.
    
    Args:B
        image_base64: Base64 encoded image
    
    Returns:
        Dictionary containing posture analysis results
    """
    try:
        # Decode image
        image = _decode_base64_image(image_base64)
        original_height, original_width = image.shape[:2]
        
        # Perform simple posture analysis
        posture_analysis = _simple_posture_analysis(image)
        
        if not posture_analysis.get("face_detected", False):
            return {
                "error": "No person/face detected in the image",
                "person_detected": False,
                "posture_score": None
            }
        
        # Create annotated image (draw face detection box)
        annotated_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, "Face Detected", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode annotated image
        annotated_base64 = _encode_image_to_base64(annotated_image)
        
        return {
            "person_detected": True,
            "posture_analysis": posture_analysis,
            "image_dimensions": {"width": original_width, "height": original_height},
            "annotated_image_base64": annotated_base64,
            "model": "OpenCV_Basic_Analysis"
        }
        
    except Exception as e:
        raise ValueError(f"Posture analysis failed: {str(e)}")

def combined_detection(image_base64: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Perform both object detection and posture analysis on the same image.
    
    Args:
        image_base64: Base64 encoded image
        confidence_threshold: Minimum confidence for object detections
    
    Returns:
        Combined results from both models
    """
    try:
        # Run both detections
        object_results = detect_objects(image_base64, confidence_threshold)
        posture_results = analyze_posture(image_base64)
        
        return {
            "object_detection": {
                "detections": object_results["detections"],
                "total_objects": object_results["total_objects"],
                "annotated_image_base64": object_results["annotated_image_base64"]
            },
            "posture_analysis": {
                "person_detected": posture_results["person_detected"],
                "posture_analysis": posture_results.get("posture_analysis"),
                "annotated_image_base64": posture_results.get("annotated_image_base64")
            },
            "image_dimensions": object_results["image_dimensions"],
            "models_used": ["DETR-ResNet50", "OpenCV_Basic_Analysis"],
            "confidence_threshold": confidence_threshold
        }
        
    except Exception as e:
        raise ValueError(f"Combined detection failed: {str(e)}")
