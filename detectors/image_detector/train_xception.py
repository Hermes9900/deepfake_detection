import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from detectors.image_detector.model import ImageDetector
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.images = []
        self.labels = []
        for label in ["real","fake"]:
            folder = os.path.join(img_folder,label)
            for f in os.listdir(folder):
                self.images.append(os.path.join(folder,f))
                self.labels.append(0 if label=="real" else 1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Training example
transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

dataset = ImageDataset("data/images", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ImageDetector()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    for imgs, labels in loader:
        preds = model(imgs)
        loss = criterion(preds.squeeze(), labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")
torch.save(model.state_dict(), "xception_detector.pth")
