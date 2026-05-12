from pydantic import BaseModel, Field
from typing import Optional

class VideoGenerationRequest(BaseModel):
    avatar_id: str = Field(..., min_length=36, max_length=36, description="UUIDv4 of the target avatar")
    script_text: str = Field(..., min_length=1, max_length=5000, description="The textual script to render")
    user_id: str = Field(..., min_length=36, max_length=36, description="UUIDv4 of the triggering user")

class VideoGenerationResponse(BaseModel):
    status: str
    job_id: str
    message: str

class JobStatus(BaseModel):
    id: str
    user_id: str
    avatar_id: str
    script_text: str
    status: str
    error_log: Optional[str] = None
