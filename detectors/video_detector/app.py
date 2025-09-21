from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import requests
import os
import tempfile

app = FastAPI(title="Video Deepfake Detector")

# Get the URL of the image detector service from an environment variable.
# The default value "http://image_detector:8000" works inside Docker Compose.
IMAGE_DETECTOR_URL = os.getenv("IMAGE_DETECTOR_URL", "http://image_detector:8000/predict")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Use a temporary file to save the uploaded video so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")

        frame_scores = []
        frame_count = 0
        
        # Get the video's frames per second (fps) to sample correctly
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30 # Default to 30 if fps is not available

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample one frame per second to be efficient
            if frame_count % fps == 0:
                # Convert the OpenCV frame (numpy array) to bytes for the HTTP request
                _, img_encoded = cv2.imencode(".jpg", frame)
                frame_bytes = img_encoded.tobytes()

                try:
                    # Call the image detector service with the frame
                    response = requests.post(
                        IMAGE_DETECTOR_URL,
                        files={'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
                    )
                    if response.status_code == 200:
                        score = response.json().get('fake_probability', 0.0)
                        frame_scores.append(score)
                except requests.RequestException as e:
                    print(f"Warning: Could not connect to image detector: {e}")
                    # Allow the service to continue even if some frames fail
                    continue
            
            frame_count += 1
            
        cap.release()

        if not frame_scores:
            raise HTTPException(status_code=500, detail="No frames were successfully analyzed. Check if the image_detector service is running.")
            
        # Aggregate scores: The maximum fake score from any frame is a strong indicator.
        # If even one frame is clearly a deepfake, the video is likely fake.
        final_score = max(frame_scores)
        
        return {
            "prediction": "fake" if final_score > 0.5 else "real",
            "fake_probability": round(final_score, 4),
            "frames_analyzed": len(frame_scores),
            "detail": "The probability is based on the highest fake score found among the analyzed frames."
        }
    finally:
        # Clean up the temporary file from the system
        os.unlink(video_path)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}