import os
import torch
import pytorchvideo.models.resnet
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import make_dataset
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Normalize
from pytorchvideo.data.encoded_video import EncodedVideo

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed/video"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "video_detector_model.pth")
BATCH_SIZE = 4      # Video models are memory-heavy, keep this small
NUM_EPOCHS = 5      # Start with a few epochs, increase later for better accuracy
LEARNING_RATE = 1e-5  # Video models benefit from a smaller learning rate

# --- Fully Functional Custom Video Dataset ---
class VideoClipDataset(Dataset):
    def __init__(self, data_path, clip_duration=2, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Use torchvision's make_dataset to find all video files and their labels
        self.samples = make_dataset(
            self.data_path, class_to_idx={"real": 0, "fake": 1}, extensions=(".mp4", ".mov", ".avi")
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            # Load the video from the given path
            video = EncodedVideo.from_path(video_path)
            
            video_duration = video.duration
            # Select a random start time for a clip
            start_sec = 0.0
            if video_duration > 2:
                start_sec = torch.randint(0, int(video_duration) - 2, (1,)).item()
            
            end_sec = start_sec + 2 # Get a 2-second clip
            
            # Get the clip from the video
            clip_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            
            # Apply transformations if they exist
            if self.transform:
                clip_data = self.transform(clip_data)
                
            return clip_data["video"], label
        except Exception as e:
            print(f"Skipping corrupted or unreadable video file: {video_path} due to {e}")
            # Return a dummy clip and label if a video fails to load
            return torch.randn(3, 8, 224, 224), torch.tensor(0)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Video Transforms ---
    # These are the standard transformations for video models
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(8),       # Sample 8 frames from the clip
            Lambda(lambda x: x / 255.0),       # Normalize pixel values to [0, 1]
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)), # Standard normalization
            ShortSideScale(size=256),          # Scale the smaller side to 256
            ]
        ),
    )

    # --- Datasets and Dataloaders ---
    train_dataset = VideoClipDataset(os.path.join(PROCESSED_DATA_DIR, 'train'), transform=transform)
    val_dataset = VideoClipDataset(os.path.join(PROCESSED_DATA_DIR, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model, Loss, and Optimizer ---
    # Load a pre-trained SlowFast model, which is excellent for video action recognition
    model = pytorchvideo.models.resnet.create_slowfast(
        model_num_class=2 # Binary classification: real or fake
    )
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Full Training and Validation Loop ---
    print("--- Starting Video Model Training ---")
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if it has the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"New best model saved to {MODEL_PATH} with accuracy: {best_val_acc:.4f}")

    print("--- Training Complete ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()