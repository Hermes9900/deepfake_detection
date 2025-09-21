import torch
import timm
from torchvision import transforms

def get_model(pretrained=False, num_classes=1):
    """Initializes the Xception model."""
    model = timm.create_model("xception", pretrained=pretrained, num_classes=num_classes)
    return model

def get_prediction_transform():
    """Returns the transform pipeline for inference."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])