# app.py (new)
from fastapi import FastAPI, UploadFile, File, Form
import uuid
from ingestion_service import utils  # S3 & DB helpers

app = FastAPI(title="Deepfake Ingestion Service")

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), uploader_id: str = Form(...)):
    file_id = str(uuid.uuid4())
    s3_key = f"{file_id}_{file.filename}"
    utils.upload_file_to_s3(file.file, s3_key)
    utils.insert_job(file_id, uploader_id, s3_key)
    return {"job_id": file_id, "s3_path": f"s3://{utils.S3_BUCKET}/{s3_key}"}
