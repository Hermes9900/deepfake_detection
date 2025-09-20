from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import json

# Load dataset from JSONL
dataset_file = "../data_generators/text_dataset.jsonl"
examples = []
with open(dataset_file) as f:
    for line in f:
        examples.append(json.loads(line))

# Convert to train/val split
train_examples = examples[:16]
val_examples = examples[16:]

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

# Prepare input & labels
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        tokens = self.tokenizer(item['text'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        label = 1 if any(cl['label']=="REFUTED" for cl in item['claims']) else 0
        return {key: val.squeeze(0) for key,val in tokens.items()}, torch.tensor(label)

train_dataset = TextDataset(train_examples)
val_dataset = TextDataset(val_examples)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./text_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
model.save_pretrained("./text_model")
tokenizer.save_pretrained("./text_model")

print("Text NLI model trained and saved in ./text_model")
