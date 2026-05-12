import torch
import torch.nn as nn
import torch.optim as optim

class GRPOTrainer:
    """
    Implements Group Relative Policy Optimization (GRPO).
    Optimizes a policy by comparing group rewards without a critic network.
    """
    def __init__(self, policy_network, learning_rate=1e-4, epsilon=0.2):
        self.policy = policy_network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.epsilon = epsilon

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Calculates advantage scores relative to the group metrics.
        Formula: A_i = (R_i - mean(R)) / (std(R) + 1e-8)
        """
        if rewards.size(0) < 2:
            return torch.zeros_like(rewards)
            
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        # Advantage normalization across the group
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        return advantages

    def execute_grpo_step(self, old_log_probs: torch.Tensor, current_log_probs: torch.Tensor, rewards: torch.Tensor) -> float:
        """
        Executes a single optimization step using the GRPO objective function.
        Uses PPO-style clipping for stability.
        """
        self.optimizer.zero_grad()
        
        # 1. Compute group-relative advantages
        advantages = self.compute_advantages(rewards)
        
        # 2. Calculate the probability ratio: r_t(theta) = pi_theta / pi_old
        # current_log_probs and old_log_probs are log-probabilities
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # 3. Apply the clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        
        # GRPO maximizes the minimum of these two components
        # We use negative for minimization by the optimizer
        loss = -torch.min(surr1, surr2).mean()
        
        # 4. Backpropagation
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
