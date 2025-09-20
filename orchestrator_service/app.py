# orchestrator_service/app.py
import os
import asyncio
import aiohttp
import boto3
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from uuid import uuid4
import psycopg2
import json

# ----------------- CONFIG -----------------
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET", "deepfake-blobs")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

DB_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/deepfake")

DETECTOR_URLS = {
    "text": os.getenv("TEXT_DETECTOR_URL","http://text_detector:8100/predict"),
    "image": os.getenv("IMAGE_DETECTOR_URL","http://image_detector:8150/predict"),
    "audio": os.getenv("AUDIO_DETECTOR_URL","http://audio_detector:8180/predict"),
    "video": os.getenv("VIDEO_DETECTOR_URL","http://video_detector:8200/predict")
}
FUSION_URL = os.getenv("FUSION_URL","http://fusion_service:8300/fuse")
# ------------------------------------------

app = FastAPI(title="Orchestrator Service")

s3_client = boto3.client("s3",
                         aws_access_key_id=AWS_ACCESS_KEY,
                         aws_secret_access_key=AWS_SECRET_KEY,
                         region_name=S3_REGION)

# ----------------- DB UTILS -----------------
def insert_job(job_id, s3_path, metadata, final_result=None):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO jobs (job_id, s3_path, metadata, result)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (job_id) DO UPDATE SET result=%s
    """, (job_id, s3_path, json.dumps(metadata), json.dumps(final_result),
          json.dumps(final_result)))
    conn.commit()
    cur.close()
    conn.close()

# ----------------- HELPERS -----------------
async def post_json(url, payload, timeout=30):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}

async def call_detectors(s3_path, metadata):
    payload = {"artifact_s3": s3_path, "metadata": metadata}
    tasks = []
    for modality, url in DETECTOR_URLS.items():
        tasks.append(post_json(url, payload))
    results = await asyncio.gather(*tasks)
    return dict(zip(DETECTOR_URLS.keys(), results))

# ----------------- API -----------------
@app.post("/ingest")
async def ingest(file: UploadFile = File(...), uploader_id: str = "unknown"):
    job_id = str(uuid4())
    s3_path = f"{job_id}_{file.filename}"
    try:
        contents = await file.read()
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_path, Body=contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    metadata = {"uploader_id": uploader_id}
    insert_job(job_id, s3_path, metadata, final_result=None)

    # Run async detector calls
    detector_results = await call_detectors(f"s3://{S3_BUCKET}/{s3_path}", metadata)

    # Call Fusion Service
    fusion_payload = {
        "job_id": job_id,
        "text_result": detector_results.get("text", {}),
        "image_result": detector_results.get("image", {}),
        "audio_result": detector_results.get("audio", {}),
        "video_result": detector_results.get("video", {}),
        "metadata": metadata
    }
    fusion_result = await post_json(FUSION_URL, fusion_payload)

    # Store final result
    insert_job(job_id, s3_path, metadata, final_result=fusion_result)

    return {"job_id": job_id, "fusion_result": fusion_result}

@app.get("/status/{job_id}")
def status(job_id: str):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT s3_path, metadata, result FROM jobs WHERE job_id=%s", (job_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    s3_path, metadata, result = row
    return {"job_id": job_id, "s3_path": s3_path, "metadata": metadata, "result": result}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8400)
