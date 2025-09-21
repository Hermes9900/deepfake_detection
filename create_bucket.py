import boto3
from botocore.exceptions import ClientError

# --- Configuration (matches your docker-compose file and utils.py) ---
S3_ENDPOINT = "http://localhost:9000"
S3_KEY = "minioadmin"
S3_SECRET = "minioadmin"
BUCKET_NAME = "deepfake" # The bucket name your ingestion_service needs

def main():
    """Connects to MinIO and creates the necessary bucket."""
    print("Attempting to connect to MinIO server...")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET
    )

    try:
        # Check if the bucket already exists so we don't create it twice
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' already exists. No action needed.")
    except ClientError as e:
        # If a 404 error is raised, the bucket does not exist, so we create it
        if e.response['Error']['Code'] == '404':
            print(f"Bucket '{BUCKET_NAME}' does not exist. Creating it now...")
            s3_client.create_bucket(Bucket=BUCKET_NAME)
            print(f"--- ✅ Bucket '{BUCKET_NAME}' created successfully! ---")
        else:
            # Handle other potential connection errors
            print("An unexpected error occurred:")
            print(e)

if __name__ == "__main__":
    print("--- MinIO Setup Script ---")
    print("This script will create the 'deepfake' bucket needed by the application.")
    
    # You need boto3 installed in your local venv for this script to run
    try:
        input("Press Enter to continue once your Docker services are running...")
        main()
    except ImportError:
        print("\nERROR: 'boto3' is not installed in your local venv.")
        print("Please run 'pip install boto3' in your active venv and try again.")
    except Exception as e:
        print(f"\nAn error occurred. Is Docker running? Details: {e}")