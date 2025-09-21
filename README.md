# Deepfake Detection Platform

This repository contains a multimodal deepfake detection platform built with a microservice architecture. It can process images, text, audio, and video to provide a fused deepfake probability score.

---

## 🚀 New Workflow

The project has been refactored for a clear, repeatable, and robust machine learning workflow.

1.  **Add Raw Data**: Place your raw training files into the `/data/raw/` subdirectories.
    -   Images: `/data/raw/images/real/` and `/data/raw/images/fake/`
    -   Text: Create a source file for the text generator.

2.  **Generate Processed Datasets**: Run the scripts in `/data_generators/` to process your raw data. These scripts will create cleaned, split (train/validation), and formatted datasets in `/data/processed/`, ready for training.
    ```bash
    # From the project root
    python data_generators/make_image_dataset.py
    python data_generators/make_text_dataset.py
    ```

3.  **Train Models**: Run the training scripts located in `/training/`. These scripts will load the processed data, train the models, and save the final artifacts (like `.pth` or `.pkl` files) into the centralized `/models/` directory.
    ```bash
    # From the project root
    python training/train_image_xception.py
    python training/train_text_nli.py
    ```

4.  **Run the Entire System**: Use Docker Compose to build and run all the services, including the detectors, databases, and file storage.
    ```bash
    docker-compose up --build
    ```
    The services will now be running, and the detector APIs will automatically load the trained models from the `/models/` directory.

---

## 🛠️ Services

-   **`ingestion_service`**: The main entry point for uploads.
-   **`orchestrator_service`**: Manages the detection workflow by calling individual detectors and the fusion service.
-   **`detectors`**: Individual services for `image`, `text`, `audio`, and `video` detection.
-   **`fusion_service`**: Aggregates scores from all detectors to produce a final, unified prediction.
-   dataset download link : https://drive.google.com/drive/folders/1JxVAKiCXBHAOk4zeXNRJnD3bwmZEiW9R?usp=drive_link
-   **`MinIO` & `Postgres`**: Backend storage for files and job metadata.
