import boto3
import os
import psycopg2

# --- S3 Helpers ---
S3_BUCKET = os.getenv("S3_BUCKET", "deepfake-blobs")
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
S3_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION
)

def upload_file_to_s3(file_obj, s3_key: str) -> str:
    s3.upload_fileobj(file_obj, S3_BUCKET, s3_key)
    return f"s3://{S3_BUCKET}/{s3_key}"


# --- PostgreSQL Helpers ---
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_USER = os.getenv("DB_USER", "user")
DB_PASS = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "deepfake")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        dbname=DB_NAME
    )

def insert_job(job_id: str, uploader_id: str, s3_key: str, status: str = "pending"):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs (job_id, uploader, s3_key, status) VALUES (%s, %s, %s, %s)",
                (job_id, uploader_id, s3_key, status)
            )
            conn.commit()

def get_job_status(job_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
            return row[0] if row else None
