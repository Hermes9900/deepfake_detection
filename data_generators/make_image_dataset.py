import json
from pathlib import Path

OUTPUT_DIR = "image_dataset"
Path(OUTPUT_DIR).mkdir(exist_ok=True)
metadata_file = Path(OUTPUT_DIR)/"metadata.jsonl"

metadata_file.unlink(missing_ok=True)

# Create dummy images and metadata
for i in range(5):
    img_name = f"real_{i}.jpg"
    (Path(OUTPUT_DIR)/img_name).write_bytes(b"\x00")  # placeholder file
    meta = {"file":img_name,"label":"real","manip_type":None}
    with open(metadata_file,"a") as f:
        f.write(json.dumps(meta)+"\n")

for i in range(5):
    img_name = f"fake_{i}.jpg"
    (Path(OUTPUT_DIR)/img_name).write_bytes(b"\x00")  # placeholder file
    meta = {"file":img_name,"label":"fake","manip_type":"face_swap"}
    with open(metadata_file,"a") as f:
        f.write(json.dumps(meta)+"\n")

print(f"Image dataset created in {OUTPUT_DIR}/")
