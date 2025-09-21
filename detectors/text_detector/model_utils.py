from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import random

# Load a NLI model for claim verification
MODEL_NAME = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nli_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

def verify_claim(claim: str, evidence_list):
    """
    Verifies a claim against evidence using NLI model.
    Returns verdict: SUPPORTED / REFUTED / NEI, and confidence (0-1)
    """
    if not evidence_list:
        return "NEI", 0.5
    premise = " ".join(evidence_list)
    result = nli_pipeline(f"{premise} </s></s> {claim}")[0]
    label = result["label"]
    score = float(result["score"])
    if label == "ENTAILMENT":
        return "SUPPORTED", score
    elif label == "CONTRADICTION":
        return "REFUTED", score
    else:
        return "NEI", score

def detect_generated_text(text: str):
    """
    Dummy stylometry / LLM detection placeholder.
    Returns a random score between 0 and 1.
    """
    return random.uniform(0, 1)
