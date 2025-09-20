import json
from pathlib import Path
import numpy as np
import cv2

OUTPUT_DIR = "video_dataset"
Path(OUTPUT_DIR).mkdir(exist_ok=True)
metadata_file = Path(OUTPUT_DIR)/"metadata.jsonl"
metadata_file.unlink(missing_ok=True)

# Create 2 real + 2 fake dummy videos
for i in range(2):
    fname = Path(OUTPUT_DIR)/f"real_{i}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(fname), fourcc, 5.0, (64,64))
    for _ in range(10):  # 10 frames
        frame = (np.random.rand(64,64,3)*255).astype(np.uint8)
        out.write(frame)
    out.release()
    meta = {"file":str(fname),"label":"real"}
    with open(metadata_file,"a") as f:
        f.write(json.dumps(meta)+"\n")

for i in range(2):
    fname = Path(OUTPUT_DIR)/f"fake_{i}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(fname), fourcc, 5.0, (64,64))
    for _ in range(10):
        frame = (np.random.rand(64,64,3)*255).astype(np.uint8)
        out.write(frame)
    out.release()
    meta = {"file":str(fname),"label":"fake","manip_type":"face_swap"}
    with open(metadata_file,"a") as f:
        f.write(json.dumps(meta)+"\n")

print(f"Video dataset created in {OUTPUT_DIR}/")
