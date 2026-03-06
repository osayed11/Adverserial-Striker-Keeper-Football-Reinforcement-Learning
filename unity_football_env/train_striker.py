"""
Striker Agent Configuration & Network
Contains the striker-specific ActorCritic network and hyperparameters.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

# ========================
# STRIKER HYPERPARAMETERS
# ========================
BEHAVIOR_NAME = "Walker"  # Will match "Walker?team=0"
LR = 3e-4
HIDDEN_SIZE = 512
NUM_LAYERS = 3
BUFFER_SIZE = 4096
BATCH_SIZE = 2048


# ========================
# STRIKER ACTOR-CRITIC
# ========================
class StrikerActorCritic(nn.Module):
    """
    Actor-Critic for the humanoid striker.
    Large network: 243 observations -> 39 actions.
    """
    def __init__(self, obs_size, action_size):
        super().__init__()
        
        # Shared backbone: obs_size -> HIDDEN_SIZE -> HIDDEN_SIZE -> HIDDEN_SIZE
        self.shared = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        
        # TODO: Actor head - outputs action means
        self.actor_mean = nn.Linear(HIDDEN_SIZE, action_size)
        
        # Learnable log standard deviation - Increased to 0.5 to force exploration of new height rewards
        self.actor_log_std = nn.Parameter(torch.full((action_size,), 0.5))
        
        # TODO: Critic head - outputs state value V(s)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, obs):
        """Forward pass through shared backbone."""
        return self.shared(obs)
    
    def get_action_and_value(self, obs):
        """
        Given observations, return sampled action, log_prob, entropy, value.
        
        Steps:
            1. Forward pass to get shared features
            2. Compute action_mean from actor head
            3. action_std = self.actor_log_std.exp()
            4. Create Normal(action_mean, action_std) distribution
            5. Sample action, compute log_prob (sum over action dims), entropy
            6. Compute value from critic head
        """
        shared_out = self.forward(obs)
        
        action_mean = self.actor_mean(shared_out)
        action_std = self.actor_log_std.exp()
        dist = Normal(action_mean, action_std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(shared_out).squeeze()
        
        return action, log_prob, entropy, value
    
    def get_deterministic_action(self, obs):
        """Returns the exact mean action WITHOUT sampling. Used for smooth inference."""
        shared_out = self.forward(obs)
        action_mean = self.actor_mean(shared_out)
        return action_mean
    
    def get_value(self, obs):
        """Get just the value estimate (for GAE bootstrap)."""
        shared_out = self.forward(obs)
        return self.critic(shared_out).squeeze()
    
    def evaluate_actions(self, obs, actions):
        """
        Re-evaluate stored actions under current policy (for PPO ratio).
        Same as get_action_and_value but uses provided actions instead of sampling.
        """
        shared_out = self.forward(obs)
        
        action_mean = self.actor_mean(shared_out)
        action_std = self.actor_log_std.exp()
        dist = Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(shared_out).squeeze()
        
        return log_prob, entropy, value
