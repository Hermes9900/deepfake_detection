import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path

DATA_DIR = "../data_generators/image_dataset"
metadata_file = Path(DATA_DIR)/"metadata.jsonl"

class ImageDataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        with open(metadata_file) as f:
            self.items = [json.loads(line) for line in f]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(Path(DATA_DIR)/item['file']).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = 1 if item['label']=="fake" else 0
        return img, label

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = ImageDataset(metadata_file, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.xception(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "image_xception.pth")
print("Image Xception model trained and saved as image_xception.pth")
