from fastapi import FastAPI, UploadFile, File, HTTPException
import librosa
import numpy as np
from detectors.audio_detector import model_utils

app = FastAPI(title="Audio Detector Service")

@app.post("/api/audio/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detect fake/synthesized audio.
    Returns audio_score (0-1) and suspicious_segments (timestamps in seconds)
    """
    try:
        audio_bytes = await file.read()
        # Save temporarily
        temp_file = f"/tmp/{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
        
        # Load audio
        y, sr = librosa.load(temp_file, sr=16000, mono=True)
        audio_score, suspicious_segments = model_utils.predict_audio(y, sr)
        
        return {"audio_score": round(float(audio_score),2),
                "suspicious_segments": suspicious_segments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
