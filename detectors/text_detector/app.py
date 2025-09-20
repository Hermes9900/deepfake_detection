from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from detectors.text_detector import bm25_index, model_utils

app = FastAPI(title="Text Detector Service")

class DetectRequest(BaseModel):
    artifact_s3: str
    max_evidence: int = 5

@app.post("/api/text/detect")
def detect(req: DetectRequest):
    """
    Detect fake content in the text artifact.
    Steps:
    1. Download text from S3 (placeholder: assume local file path)
    2. Split sentences and extract claims
    3. Retrieve evidence using BM25
    4. Verify claims with NLI model
    5. Return score, claims[], reason_codes
    """
    try:
        # For demo, assume artifact_s3 is local path
        text = open(req.artifact_s3, "r", encoding="utf-8").read()
        sentences = text.split(".")  # simple sentence split

        results = []
        num_refuted = 0
        for i, sent in enumerate(sentences):
            evidence_list = bm25_index.retrieve(sent, req.max_evidence)
            verdict, confidence = model_utils.verify_claim(sent, evidence_list)
            results.append({
                "id": f"c{i+1}",
                "verdict": verdict,
                "confidence": confidence,
                "evidence": [{"url": e, "snippet": e[:100]} for e in evidence_list]
            })
            if verdict == "REFUTED":
                num_refuted += 1

        text_score = num_refuted / max(len(sentences), 1)
        llm_flag = model_utils.detect_generated_text(text)

        reason_codes = []
        if llm_flag > 0.7:
            reason_codes.append("llm_stylometry_high")
        if text_score > 0.5:
            reason_codes.append("unsupported_claims_majority")

        return {
            "text_score": round(text_score, 2),
            "llm_flag": round(llm_flag, 2),
            "claims": results,
            "reason_codes": reason_codes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
