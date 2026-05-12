import pytest
import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workers.video_worker.alignment.grpo_trainer import GRPOTrainer
from workers.video_worker.alignment.reward_funcs import lip_sync_reward, temporal_smoothness_reward

class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

def test_advantage_normalization():
    """
    Verifies that the advantage calculation correctly scales rewards to zero mean.
    """
    policy = SimplePolicy()
    trainer = GRPOTrainer(policy)
    
    # Synthetic rewards for a group of 5 variations
    rewards = torch.tensor([10.0, 12.0, 8.0, 15.0, 5.0])
    
    advantages = trainer.compute_advantages(rewards)
    
    # Mean should be 0 (within floating point error)
    assert torch.abs(advantages.mean()) < 1e-6
    # Standard deviation should be 1
    assert torch.abs(advantages.std() - 1.0) < 1e-6

def test_grpo_training_step():
    """
    Verifies that a backward pass executes and returns a numerical loss.
    """
    policy = SimplePolicy()
    trainer = GRPOTrainer(policy)
    
    group_size = 4
    # Mock inputs for the policy
    inputs = torch.randn(group_size, 10)
    # Mock log probabilities derived from policy
    old_log_probs = torch.randn(group_size, requires_grad=False)
    current_log_probs = policy(inputs).squeeze()
    rewards = torch.tensor([1.5, 2.0, 0.5, 3.0])
    
    loss = trainer.execute_grpo_step(old_log_probs, current_log_probs, rewards)
    
    assert isinstance(loss, float)
    # Check that gradients were computed for the policy
    for param in policy.parameters():
        assert param.grad is not None

def test_reward_functions():
    """
    Verifies that the reward functions return valid numerical scores.
    """
    # Create mock frames (3 frames, RGB, 64x64 for speed)
    frames = torch.randn(3, 3, 64, 64)
    targets = torch.randn(3, 128)
    
    ls_reward = lip_sync_reward(frames, targets)
    ts_reward = temporal_smoothness_reward(frames)
    
    assert ls_reward.shape == (3,)
    assert isinstance(ts_reward, float)
    assert ls_reward.min() >= 0
    assert ts_reward >= 0
