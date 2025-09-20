import json
from pathlib import Path

# Example: generate 10 synthetic + 10 real articles
OUTPUT_FILE = "text_dataset.jsonl"
Path(OUTPUT_FILE).unlink(missing_ok=True)

real_articles = [
    {"id":f"real-{i}", "headline":f"Real headline {i}", "text":f"Real article text {i}",
     "claims":[{"id":"c1","span":[0,10],"text":"Claim X","label":"SUPPORTED"}]} 
    for i in range(10)
]

fake_articles = [
    {"id":f"fake-{i}", "headline":f"Fake headline {i}", "text":f"Fake article text {i}",
     "claims":[{"id":"c1","span":[0,10],"text":"Claim Y","label":"REFUTED"}]} 
    for i in range(10)
]

with open(OUTPUT_FILE,"w") as f:
    for article in real_articles + fake_articles:
        f.write(json.dumps(article)+"\n")

print(f"Text dataset created: {OUTPUT_FILE}")
