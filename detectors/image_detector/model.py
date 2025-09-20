import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

# Use pretrained Xception (via timm)
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Xception binary classifier
class ImageDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("xception", pretrained=True, num_classes=1)
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Instantiate model
detector_model = ImageDetector().to(device)
detector_model.eval()

# Transform pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def predict_image(img_np):
    """Predict image manipulation probability"""
    x = preprocess(img_np).unsqueeze(0).to(device)
    with torch.no_grad():
        score = detector_model(x).item()
    return score
