import pytest
import uuid
import sys
import os
import torch

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workers.video_worker.worker import VideoWorker
from workers.video_worker.inference import VideoInferenceStub

def test_viseme_generation_dimensions():
    """
    Verifies that the generated frame sequences match the expected tensor dimensions.
    """
    engine = VideoInferenceStub()
    num_frames = 45
    audio_path = "/mock_storage/audio/test.wav"
    
    frames = engine.map_audio_to_visemes(audio_path, num_frames=num_frames)
    
    # Check shape: (frames, channels, height, width)
    assert frames.shape == (num_frames, 3, 512, 512)
    assert isinstance(frames, torch.Tensor)

def test_video_orchestration_success():
    """
    Verifies that providing valid paths produces a success status and output path.
    """
    worker = VideoWorker()
    job_payload = {
        "job_id": str(uuid.uuid4()),
        "audio_path": "/mock_storage/audio/speech.wav",
        "reference_video_path": "/mock_storage/refs/avatar_1.mp4"
    }
    
    result = worker.process_job(job_payload)
    
    assert result["status"] == "video_complete"
    assert result["video_path"].endswith(".mp4")
    assert result["frame_count"] == 60
    assert result["error"] is None

def test_video_worker_missing_reference_error():
    """
    Verifies graceful failure when reference video path is missing.
    """
    worker = VideoWorker()
    job_payload = {
        "job_id": str(uuid.uuid4()),
        "audio_path": "/mock_storage/audio/speech.wav"
        # Missing reference_video_path
    }
    
    result = worker.process_job(job_payload)
    
    assert result["status"] == "failed"
    assert "Reference video path is missing" in result["error"]

def test_inference_missing_audio_error():
    """
    Verifies that the inference engine throws an error if no audio path is provided.
    """
    engine = VideoInferenceStub()
    with pytest.raises(FileNotFoundError) as excinfo:
        engine.map_audio_to_visemes(None)
    assert "Audio track path is required" in str(excinfo.value)
