from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import os

app = FastAPI(title="Text Deepfake Detector")

# --- Model Loading ---
MODEL_PATH = "models/text_detector_model"
classifier = None

@app.on_event("startup")
def load_model():
    """Load the classification pipeline when the service starts."""
    global classifier
    if not os.path.isdir(MODEL_PATH):
        print(f"WARNING: Model directory not found at {MODEL_PATH}. The '/predict' endpoint will not work.")
        return

    try:
        # Use GPU if available (device=0), otherwise CPU (device=-1)
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=device
        )
        print(f"Text detector model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Pydantic model for validating the request body
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    """Prediction endpoint for the text detector."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        # The pipeline returns scores for all labels. We get both 'real' and 'fake'.
        results = classifier(request.text, top_k=2)

        # Find the score specifically for the 'fake' label
        fake_score = 0.0
        for result in results:
            if result['label'].lower() == 'fake':
                fake_score = result['score']
                break
        
        return {
            "prediction": "fake" if fake_score > 0.5 else "real",
            "fake_probability": round(fake_score, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok", "model_loaded": classifier is not None}