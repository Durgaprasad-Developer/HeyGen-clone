from typing import Dict, Any

# In-memory database tracking engine state for Phase 1 isolation
JOBS_DB: Dict[str, Dict[str, Any]] = {}

def save_job(job_id: str, job_data: Dict[str, Any]):
    JOBS_DB[job_id] = job_data

def get_job(job_id: str) -> Dict[str, Any]:
    return JOBS_DB.get(job_id)
