import pandas as pd
import json
from pathlib import Path

# --- Configuration ---
# This script is specifically designed for the WELFake_Dataset.csv
# 1. CREATE THIS FOLDER: Create a folder named 'text' inside 'data/raw'
# 2. PUT CSV HERE: Place your 'WELFake_Dataset.csv' file into it.
RAW_DATA_DIR = Path("data/raw/text")
PROCESSED_DATA_DIR = Path("data/processed/text")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DATA_DIR / "text_dataset.jsonl"
OUTPUT_FILE.unlink(missing_ok=True) # Delete old file to prevent errors

def main():
    print("Starting text dataset generation from 'WELFake_Dataset.csv'...")
    
    # Define the expected filename for this dataset
    input_csv_path = RAW_DATA_DIR / "WELFake_Dataset.csv"

    try:
        # Load the dataset from the CSV file
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Make sure 'WELFake_Dataset.csv' is in the '{RAW_DATA_DIR}' directory.")
        return

    # --- Standardize the data ---
    # The training script expects text labels 'real' or 'fake'.
    # This dataset uses 0 for 'real' and 1 for 'fake'. We need to translate them.
    df['label_text'] = df['label'].apply(lambda x: 'real' if x == 0 else 'fake')
    
    # Combine title and text for a richer input to the model
    # Handle cases where title or text might be missing (NaN) by filling with an empty string
    df['full_text'] = df['title'].fillna('') + ". " + df['text'].fillna('')

    # --- Write to JSONL format ---
    # The training script expects a 'text' and 'label' key.
    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for index, row in df.iterrows():
            # Skip any rows where the combined text is empty or just whitespace
            if not row['full_text'].strip():
                continue

            # Create a dictionary for each article
            article_record = {
                "id": f"welfake_{index}",
                "text": row['full_text'],
                "label": row['label_text'] # Use our translated text label
            }
            # Write each record as a new line in the JSONL file
            f.write(json.dumps(article_record) + "\n")
            count += 1

    print(f"Text dataset generation complete! Wrote {count} articles to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
