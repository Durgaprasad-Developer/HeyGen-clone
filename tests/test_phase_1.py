import pytest
from fastapi.testclient import TestClient
import uuid
import sys
import os
import importlib.util

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the module with a hyphen in the name
module_name = "api-gateway.app.main"
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api-gateway", "app", "main.py"))
spec = importlib.util.spec_from_file_location(module_name, path)
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)
app = main_module.app

client = TestClient(app)

def test_create_job_success():
    user_id = str(uuid.uuid4())
    avatar_id = str(uuid.uuid4())
    payload = {
        "user_id": user_id,
        "avatar_id": avatar_id,
        "script_text": "Hello, this is a test script for the avatar generation pipeline."
    }
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert "job_id" in data
    
    # Verify job status via GET
    job_id = data["job_id"]
    get_response = client.get(f"/api/v1/jobs/{job_id}")
    assert get_response.status_code == 200
    job_data = get_response.json()
    assert job_data["id"] == job_id
    assert job_data["user_id"] == user_id
    assert job_data["avatar_id"] == avatar_id
    assert job_data["script_text"] == payload["script_text"]
    assert job_data["status"] == "pending"

def test_create_job_validation_error():
    # Invalid UUID (too short)
    payload = {
        "user_id": "short-uuid",
        "avatar_id": str(uuid.uuid4()),
        "script_text": "Test"
    }
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422
    
    # Missing field
    payload = {
        "user_id": str(uuid.uuid4()),
        "avatar_id": str(uuid.uuid4())
        # script_text missing
    }
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422

def test_get_job_status_not_found():
    random_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/jobs/{random_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job entity not found"
