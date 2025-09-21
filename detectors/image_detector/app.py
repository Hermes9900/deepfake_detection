import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
import io
# CORRECTED IMPORT: Removed the leading dot from '.model'
from model import get_model, get_prediction_transform

app = FastAPI(title="Image Deepfake Detector")

# --- Model Loading ---
MODEL_PATH = "models/image_detector.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

@app.on_event("startup")
def load_model():
    """Load the trained model when the service starts."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}. The '/predict' endpoint will not work.")
        return
    
    try:
        model = get_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Image detector model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading image model: {e}")

transform = get_prediction_transform()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not available.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0).to(device)
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    label = "fake" if probability > 0.5 else "real"
    return {
        "filename": file.filename,
        "prediction": label,
        "fake_probability": round(probability, 4)
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}