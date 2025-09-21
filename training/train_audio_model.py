# --- Force Hugging Face datasets to skip torchcodec ---
import datasets
datasets.config.USE_TORCHCODEC = False  # must come before load_dataset

import os
import torch
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments
import numpy as np

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed/audio"
MODEL_DIR = "models/audio_detector_model"
BASE_MODEL = "mit-han-lab/ast-finetuned-audioset-10-10-0.4593"

def main():
    # --- 1. Load the Dataset ---
    dataset = load_dataset("audiofolder", data_dir=PROCESSED_DATA_DIR)
    label2id = {"real": 0, "fake": 1}
    id2label = {0: "real", 1: "fake"}

    print(f"Dataset loaded. Training on {len(dataset['train'])} samples, validating on {len(dataset['validation'])} samples.")

    # --- 2. Preprocess ---
    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * 10.0),
            truncation=True
        )
        return inputs

    train_dataset = dataset['train'].map(preprocess_function, remove_columns="audio", batched=True)
    eval_dataset = dataset['validation'].map(preprocess_function, remove_columns="audio", batched=True)

    # --- 3. Model & Trainer ---
    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        label2id=label2id,
        id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
    )

    print("--- Starting Audio Model Training ---")
    trainer.train()

    print("--- Training Complete ---")
    trainer.save_model(MODEL_DIR)
    print(f"Best model saved to: {MODEL_DIR}")

if __name__ == '__main__':
    main()
