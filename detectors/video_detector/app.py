from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import tempfile
import subprocess
from detectors.video_detector import model_utils

app = FastAPI(title="Video Detector Service")

@app.post("/api/video/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detect manipulated video.
    Steps:
    1. Save uploaded video temporarily
    2. Extract frames
    3. Run frame-level image detector
    4. Compute AV-sync score (placeholder)
    5. Return video_score, av_sync_score, frame_flags
    """
    try:
        # Save uploaded video
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_file.write(await file.read())
        tmp_file.close()
        
        # Extract frames at 1fps for overview
        cap = cv2.VideoCapture(tmp_file.name)
        frame_flags = []
        frame_idx = 0
        success, frame = cap.read()
        while success:
            # Dummy frame score
            score = model_utils.predict_frame(frame)
            frame_flags.append({"frame_idx": frame_idx, "score": round(float(score),2)})
            frame_idx += 1
            # Skip 29 frames (~1s at 30fps)
            for _ in range(29):
                success, frame = cap.read()
        
        # Placeholder AV-sync score
        av_sync_score = model_utils.predict_av_sync(tmp_file.name)
        
        # Video-level score: average of frame scores
        video_score = np.mean([f["score"] for f in frame_flags]) if frame_flags else 0
        
        return {
            "video_score": round(float(video_score),2),
            "av_sync_score": round(float(av_sync_score),2),
            "frame_flags": frame_flags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
