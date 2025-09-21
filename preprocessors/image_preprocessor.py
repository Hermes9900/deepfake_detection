import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN

detector = MTCNN()

def preprocess_image(image_path: str):
    """
    Load image, detect faces, resize and return face crops.
    Returns list of face images (as numpy arrays).
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    crops = []
    for face in faces:
        x, y, w, h = face['box']
        crop = img_rgb[y:y+h, x:x+w]
        crop = cv2.resize(crop, (224,224))
        crops.append(crop)
    return crops
