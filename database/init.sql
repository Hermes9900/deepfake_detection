-- This script will be run automatically by Postgres when the container first starts.

-- Create the main table to track analysis jobs
CREATE TABLE IF NOT EXISTS jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    uploader_id VARCHAR(255),
    s3_key VARCHAR(1024) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB,
    result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Optional: Add a message to the log to confirm the script ran
\echo 'Database initialized and "jobs" table created.'
