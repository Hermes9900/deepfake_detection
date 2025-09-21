import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# --- Paths ---
DATA_FILE = Path("data/processed/text/text_dataset.jsonl")
MODEL_DIR = Path("models/text_nli")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_FILE}. "
            "Run the preprocessing script first."
        )

    # Load JSONL dataset
    dataset = load_dataset("json", data_files=str(DATA_FILE))

    # Convert "real"/"fake" into numerical labels
    def encode_labels(example):
        example["label"] = 0 if example["label"] == "real" else 1
        return example

    dataset = dataset.map(encode_labels)

    # Train/test split
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    # --- Training arguments (legacy-compatible) ---
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "results"),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )

    # --- Start training ---
    print("Starting training on downstream task...")
    trainer.train()

    # --- Save model ---
    print(f"Saving model to {MODEL_DIR}...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
