from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fusion_service import fusion_model
import numpy as np

app = FastAPI(title="Fusion Service")

class DetectorResults(BaseModel):
    text_score: float = 0.0
    llm_flag: float = 0.0
    percent_claims_refuted: float = 0.0
    source_score: float = 0.0
    image_score: float = 0.0
    image_localization_confidence: float = 0.0
    audio_score: float = 0.0
    video_score: float = 0.0
    av_sync_score: float = 0.0

@app.post("/api/fuse")
def fuse(results: DetectorResults):
    """
    Combine detector outputs into a single fake_probability and reason_codes
    """
    try:
        features = np.array([[
            results.text_score,
            results.llm_flag,
            results.percent_claims_refuted,
            results.source_score,
            results.image_score,
            results.image_localization_confidence,
            results.audio_score,
            results.video_score,
            results.av_sync_score
        ]])
        fake_prob, reason_codes = fusion_model.predict_fusion(features)
        return {"fake_probability": round(float(fake_prob),2),
                "reason_codes": reason_codes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
