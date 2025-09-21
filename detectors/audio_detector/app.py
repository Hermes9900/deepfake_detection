from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import torch
import os

app = FastAPI(title="Audio Deepfake Detector")

# --- Model Loading ---
MODEL_PATH = "models/audio_detector_model"
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
            "audio-classification", 
            model=MODEL_PATH, 
            feature_extractor=MODEL_PATH,
            device=device
        )
        print(f"Audio detector model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading audio model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")
    
    # Read the audio file bytes directly from the upload
    audio_bytes = await file.read()
    
    try:
        # The pipeline can process raw bytes
        results = classifier(audio_bytes, top_k=2) # Get scores for both labels
        
        # Find the score for the 'fake' label
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