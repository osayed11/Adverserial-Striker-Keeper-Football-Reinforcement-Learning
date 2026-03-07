"""
Goalkeeper Agent Configuration & Network
Contains the goalkeeper-specific ActorCritic network and hyperparameters.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

# ========================
# GOALKEEPER HYPERPARAMETERS
# ========================
BEHAVIOR_NAME = "Goalkeeper"  # Will match "Goalkeeper?team=0"
LR = 3e-4
HIDDEN_SIZE = 128      # Reverted for simpler observation processing
NUM_LAYERS = 2
BUFFER_SIZE = 128     # One entry per episode (one-shot agent), so updates every ~128 episodes
BATCH_SIZE = 32
ENTROPY_COEFF = 0.15   # Higher entropy to prevent policy collapse (single decision per episode)
MIN_LOG_STD = -1.0     # Floor on log_std — prevents std from collapsing below ~0.37


# ========================
# GOALKEEPER ACTOR-CRITIC
# ========================
class GoalkeeperActorCritic(nn.Module):
    """
    Actor-Critic for the goalkeeper.
    Small network: 15 observations -> 1 action. (Updated for hips + foot)
    """
    def __init__(self, obs_size, action_size):
        super().__init__()
        
        # Shared backbone: obs_size -> HIDDEN_SIZE -> HIDDEN_SIZE
        self.shared = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor_mean = nn.Linear(HIDDEN_SIZE, action_size)
        
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        
        # Critic head
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, obs):
        return self.shared(obs)
    
    def get_action_and_value(self, obs):
        """Same interface as striker."""
        shared_out = self.forward(obs)
        
        action_mean = self.actor_mean(shared_out)
        clamped_log_std = torch.clamp(self.actor_log_std, min=MIN_LOG_STD)
        action_std = clamped_log_std.exp()
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
        shared_out = self.forward(obs)
        return self.critic(shared_out).squeeze()
    
    def evaluate_actions(self, obs, actions):
        shared_out = self.forward(obs)
        
        action_mean = self.actor_mean(shared_out)
        clamped_log_std = torch.clamp(self.actor_log_std, min=MIN_LOG_STD)
        action_std = clamped_log_std.exp()
        dist = Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(shared_out).squeeze()
        
        return log_prob, entropy, value
