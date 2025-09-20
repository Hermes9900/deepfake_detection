from fastapi import FastAPI, UploadFile, File, HTTPException
from detectors.image_detector import model
import numpy as np
import cv2
import uuid

app = FastAPI(title="Image Detector Service")

@app.post("/api/image/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detects manipulated image.
    Steps:
    1. Read image file
    2. Preprocess
    3. Predict with Xception
    4. Return image_score (0-1), mask placeholder, reason_codes
    """
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (299, 299))
        
        score = model.predict_image(img_resized)
        
        # For demo, mask is all zeros (replace with U-Net for real localization)
        mask = np.zeros((299,299)).tolist()
        
        reason_codes = []
        if score > 0.7:
            reason_codes.append("manipulated_confident")
        
        return {"image_score": round(float(score),2),
                "mask": mask,
                "reason_codes": reason_codes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
