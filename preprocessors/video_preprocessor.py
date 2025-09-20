import cv2
import numpy as np
from preprocessors.image_preprocessor import detector

def preprocess_video(video_path: str, sample_rate: int = 1):
    """
    Extract frames from video, detect faces per frame.
    sample_rate = number of frames per second to extract
    Returns list of dicts: {frame_number, face_crops}
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_data = []
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % int(fps/sample_rate) == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            crops = []
            for face in faces:
                x, y, w, h = face['box']
                crop = img_rgb[y:y+h, x:x+w]
                crops.append(crop)
            frames_data.append({"frame_number": frame_number, "face_crops": crops})
        frame_number += 1
    cap.release()
    return frames_data
