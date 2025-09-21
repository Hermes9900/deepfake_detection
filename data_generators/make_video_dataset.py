import os
import shutil
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_DIR = 'data/raw/video'
PROCESSED_DATA_DIR = 'data/processed/video'
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

def copy_files(file_paths, destination_folder):
    """Copies a list of files to a destination folder."""
    os.makedirs(destination_folder, exist_ok=True)
    for file_path in file_paths:
        try:
            shutil.copy(file_path, destination_folder)
        except Exception as e:
            print(f"Error copying file {file_path}: {e}")

def main():
    print("Starting video dataset organization...")
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)

    real_path = os.path.join(RAW_DATA_DIR, 'real')
    fake_path = os.path.join(RAW_DATA_DIR, 'fake')

    # Find all common video files
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.lower().endswith(video_extensions)]
    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.lower().endswith(video_extensions)]
    print(f"Found {len(real_files)} real videos and {len(fake_files)} fake videos.")

    # Create train/validation splits
    real_train, real_val = train_test_split(real_files, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    fake_train, fake_val = train_test_split(fake_files, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)

    # Copy files to the processed directory structure
    copy_files(real_train, os.path.join(PROCESSED_DATA_DIR, 'train', 'real'))
    copy_files(fake_train, os.path.join(PROCESSED_DATA_DIR, 'train', 'fake'))
    copy_files(real_val, os.path.join(PROCESSED_DATA_DIR, 'val', 'real'))
    copy_files(fake_val, os.path.join(PROCESSED_DATA_DIR, 'val', 'fake'))

    print("Video dataset organization complete!")

if __name__ == '__main__':
    main()
