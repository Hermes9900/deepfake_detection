import json
from pathlib import Path
import numpy as np
import soundfile as sf

OUTPUT_DIR = "audio_dataset"
Path(OUTPUT_DIR).mkdir(exist_ok=True)
metadata_file = Path(OUTPUT_DIR)/"metadata.jsonl"
metadata_file.unlink(missing_ok=True)

sr = 16000
duration = 2  # seconds

for i in range(5):
    y = np.random.randn(sr*duration)
    fname = Path(OUTPUT_DIR)/f"real_{i}.wav"
    sf.write(fname, y, sr)
    meta = {"file":str(fname),"label":"bona_fide"}
    with open(metadata_file,"a") as f:
        f.write(json.dumps(meta)+"\n")

for i in range(5):
    y = np.random.randn(sr*duration)
    fname = Path(OUTPUT_DIR)/f"fake_{i}.wav"
    sf.write(fname, y, sr)
    meta = {"file":str(fname),"label":"spoof","attack_type":"TTS"}
    with open(metadata_file,"a") as f:
        f.write(json.dumps(meta)+"\n")

print(f"Audio dataset created in {OUTPUT_DIR}/")
