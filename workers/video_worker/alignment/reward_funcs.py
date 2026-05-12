import torch

def lip_sync_reward(generated_frames: torch.Tensor, phonetic_targets: torch.Tensor) -> torch.Tensor:
    """
    Calculates a reward based on the alignment between mouth shapes and phonetic targets.
    Simulated using MSE between flattened viseme matrices.
    """
    # generated_frames: (num_frames, 3, 512, 512)
    # phonetic_targets: (num_frames, viseme_dim)
    
    # Simple simulation: Higher reward for lower distance to 'ideal' viseme targets
    # We simulate a target comparison
    diff = torch.abs(generated_frames.mean(dim=(1, 2, 3)) - phonetic_targets.mean(dim=1))
    reward = 1.0 / (diff + 1e-6)
    
    return reward

def temporal_smoothness_reward(generated_frames: torch.Tensor) -> float:
    """
    Calculates a reward based on the smoothness between adjacent frames.
    Higher variance between frames results in a lower reward to prevent jitter.
    """
    # Calculate pixel-wise difference between adjacent frames
    # frames: (num_frames, channels, height, width)
    diffs = generated_frames[1:] - generated_frames[:-1]
    
    # Sum of absolute differences (SAD) across the sequence
    jitter = torch.mean(torch.abs(diffs))
    
    # Smoothness reward is the inverse of jitter
    reward = 1.0 / (jitter + 1e-6)
    
    return reward.item()
