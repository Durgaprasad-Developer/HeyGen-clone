import pytest
import uuid
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workers.audio_worker.worker import AudioWorker
from workers.audio_worker.tts_engine import TTSEngineStub

def test_tts_synthesis_success():
    """
    Verifies that valid text + voice parameters successfully execute through the engine stub.
    """
    worker = AudioWorker()
    job_payload = {
        "id": str(uuid.uuid4()),
        "avatar_id": str(uuid.uuid4()),
        "script_text": "This is a high-fidelity synthetic voice test."
    }
    
    result = worker.process_job(job_payload)
    
    assert result["status"] == "audio_complete"
    assert result["audio_path"].startswith("/mock_storage/audio_chunks/")
    assert result["audio_path"].endswith(".wav")
    assert result["error"] is None

def test_tts_empty_script_error():
    """
    Verifies that empty script strings throw explicit ValueError handling states.
    """
    engine = TTSEngineStub()
    with pytest.raises(ValueError) as excinfo:
        engine.clone_and_synthesize("", "/mock/path.wav")
    assert "Script content cannot be empty" in str(excinfo.value)

def test_worker_error_handling():
    """
    Verifies that the worker properly captures and returns error states from the engine.
    """
    worker = AudioWorker()
    job_payload = {
        "id": str(uuid.uuid4()),
        "avatar_id": str(uuid.uuid4()),
        "script_text": "" # Should trigger ValueError
    }
    
    result = worker.process_job(job_payload)
    
    assert result["status"] == "failed"
    assert "Script content cannot be empty" in result["error"]
    assert result["audio_path"] is None
