# Deepfake Detection MVP

This repository contains an MVP skeleton for a multimodal deepfake detection platform.

Services included (MVP):
- ingestion_service (FastAPI) — accepts uploads/URLs, stores raw blobs to MinIO, posts jobs
- preprocessors — text/image/audio/video preprocessing stubs
- detectors:
  - text_detector (FastAPI) — BM25 retrieval + NLI/stylometry stubs
  - image_detector (FastAPI) — face-crop + heuristic mask detector (placeholder)
- fusion_service (FastAPI) — deterministic weighted fusion aggregator
- data_generators — small scripts to seed synthetic/real-like data
- training — starter training scripts for image + fusion
- labelstudio — Label Studio config sample

Run locally:
1. Install Docker & Docker Compose.
2. `docker-compose up --build` (starts MinIO and service containers).
3. POST files/URLs to ingestion `/ingest` (port 8000) or call detectors directly.

This is an MVP with placeholders. Replace heuristics with real models:
- Use Hugging Face Transformers (CrossEncoder, MNLI) for text detection.
- Use EfficientNet/Xception + U-Net heads for image localization.
- Use ResNet/RawNet for audio; 3D CNN/Temporal Transformers for video.
