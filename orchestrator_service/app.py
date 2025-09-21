# orchestrator_service/app.py (Updated with JSON fix)

import os
import asyncio
import aiohttp
import mimetypes
import json # <--- ADDED THIS LINE
from fastapi import FastAPI, UploadFile, File, HTTPException
from uuid import uuid4
import asyncpg
import aiobotocore.session
from contextlib import asynccontextmanager

# --- Configuration ---
DB_URL = os.getenv("DB_URL")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = "minioadmin"
S3_SECRET = "minioadmin"

# URLs for detector services
TEXT_DETECTOR_URL = os.getenv("TEXT_DETECTOR_URL")
IMAGE_DETECTOR_URL = os.getenv("IMAGE_DETECTOR_URL")
AUDIO_DETECTOR_URL = os.getenv("AUDIO_DETECTOR_URL")
VIDEO_DETECTOR_URL = os.getenv("VIDEO_DETECTOR_URL")
FUSION_SERVICE_URL = os.getenv("FUSION_SERVICE_URL")

db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    for i in range(5):
        try:
            db_pool = await asyncpg.create_pool(dsn=DB_URL)
            print("✅ Database connection successful.")
            break
        except ConnectionRefusedError:
            print(f"Database connection refused. Retrying in 3 seconds... (Attempt {i+1}/5)")
            await asyncio.sleep(3)
    else:
        print("❌ Could not connect to the database after several retries. Exiting.")
        raise RuntimeError("Failed to connect to the database.")
    yield
    await db_pool.close()

app = FastAPI(title="Orchestrator Service", lifespan=lifespan)

async def post_json(session, url, data):
    try:
        async with session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        print(f"Error calling {url}: {e}")
        return {"fake_probability": 0.0, "error": str(e)}

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    job_id = str(uuid4())
    file_content = await file.read()
    s3_key = f"{job_id}-{file.filename}"

    # 1. Upload to S3 (asynchronously)
    session = aiobotocore.session.get_session()
    async with session.create_client("s3", endpoint_url=S3_ENDPOINT, aws_secret_access_key=S3_SECRET, aws_access_key_id=S3_KEY) as s3_client:
        await s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=file_content)

    # 2. Create job record in DB
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO jobs (job_id, s3_key, status) VALUES ($1, $2, $3)",
            job_id, s3_key, "processing"
        )

    # 3. Detect file type and call the correct detector
    mime_type, _ = mimetypes.guess_type(file.filename)
    detector_results = {}
    
    async with aiohttp.ClientSession() as http_session:
        detector_payload = {"s3_path": s3_key}
        
        if mime_type and mime_type.startswith("image/"):
            detector_results["image"] = await post_json(http_session, IMAGE_DETECTOR_URL, detector_payload)
        # Add other detector calls here based on mime_type if needed
        
        # 4. Call Fusion Service
        fusion_payload = {"detector_results": detector_results, "job_id": job_id}
        fusion_result = await post_json(http_session, FUSION_SERVICE_URL, fusion_payload)

    # 5. Update job with final result
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE jobs SET status = 'complete', result = $1 WHERE job_id = $2",
            json.dumps(fusion_result), job_id # <--- CORRECTED THIS LINE
        )
    
    return {"job_id": job_id, "fusion_result": fusion_result}