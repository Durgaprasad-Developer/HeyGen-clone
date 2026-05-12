import uuid
from fastapi import FastAPI, HTTPException, status
from .models.generation_job import VideoGenerationRequest, VideoGenerationResponse, JobStatus
from .database import JOBS_DB, save_job, get_job
from .config import settings

app = FastAPI(title=settings.APP_NAME, version="1.0")

@app.post(f"{settings.API_V1_STR}/generate", response_model=VideoGenerationResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_avatar_generation(payload: VideoGenerationRequest):
    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "user_id": payload.user_id,
        "avatar_id": payload.avatar_id,
        "script_text": payload.script_text,
        "status": "pending",
        "error_log": None
    }
    save_job(job_id, job_data)
    
    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Pipeline initialization successful."
    }

@app.get(f"{settings.API_V1_STR}/jobs/{{job_id}}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job entity not found")
    return job
