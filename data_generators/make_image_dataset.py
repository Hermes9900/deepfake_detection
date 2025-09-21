import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# --- Configuration ---
RAW_DATA_DIR = 'data/raw/images'
PROCESSED_DATA_DIR = 'data/processed/images'
IMG_SIZE = (299, 299)
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

def process_and_save_images(file_paths, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    for file_path in file_paths:
        try:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                file_name = os.path.basename(file_path)
                img.save(os.path.join(destination_folder, file_name))
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def main():
    print("Starting image dataset generation...")

    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Removing existing processed data at: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)

    real_images_path = os.path.join(RAW_DATA_DIR, 'real')
    fake_images_path = os.path.join(RAW_DATA_DIR, 'fake')

    real_files = [os.path.join(real_images_path, f) for f in os.listdir(real_images_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    fake_files = [os.path.join(fake_images_path, f) for f in os.listdir(fake_images_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images.")

    real_train, real_val = train_test_split(real_files, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    fake_train, fake_val = train_test_split(fake_files, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)

    print("Processing and saving training images...")
    process_and_save_images(real_train, os.path.join(PROCESSED_DATA_DIR, 'train', 'real'))
    process_and_save_images(fake_train, os.path.join(PROCESSED_DATA_DIR, 'train', 'fake'))

    print("Processing and saving validation images...")
    process_and_save_images(real_val, os.path.join(PROCESSED_DATA_DIR, 'val', 'real'))
    process_and_save_images(fake_val, os.path.join(PROCESSED_DATA_DIR, 'val', 'fake'))

    print("-" * 30)
    print("Dataset generation complete!")
    print(f"Processed data is located in: {PROCESSED_DATA_DIR}")

if __name__ == '__main__':
    main()
