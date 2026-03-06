"""
Adversarial Training — Striker vs Goalkeeper
Shared training loop that runs both agents simultaneously.

Usage:
    1. python main.py
    2. Press Play in Unity Editor
    3. tensorboard --logdir=runs/
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from train_striker import StrikerActorCritic, BEHAVIOR_NAME as STRIKER_BEHAVIOR
from train_striker import LR as STRIKER_LR, BUFFER_SIZE as STRIKER_BUFFER, BATCH_SIZE as STRIKER_BATCH
from train_goalkeeper import GoalkeeperActorCritic, BEHAVIOR_NAME as GK_BEHAVIOR
from train_goalkeeper import LR as GK_LR, BUFFER_SIZE as GK_BUFFER, BATCH_SIZE as GK_BATCH

# ========================
# SHARED HYPERPARAMETERS
# ========================
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.02
VALUE_COEFF = 0.5
NUM_EPOCHS = 3
MAX_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 3000  # Safety net: force reset if episode runs too long
SAVE_INTERVAL = 100
CHECKPOINT_DIR = "checkpoints"


# ========================
# ROLLOUT BUFFER
# ========================
class RolloutBuffer:
    """Stores transitions for on-policy PPO updates."""
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def store(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.obs)


# ========================
# COMPUTE GAE
# ========================
def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: list of rewards
        values:  list of V(s) estimates
        dones:   list of episode termination flags
        next_value: V(s') bootstrap for the last state
        gamma: discount factor
        lam: GAE lambda
    
    Returns:
        advantages, returns (both np.arrays)
    
    TODO: Implement GAE
    Hint: Work backwards from the last step:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
    """
    advantages = np.zeros(len(rewards))
    
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value  # Bootstrap from argument
        else:
            next_val = values[t + 1]
        
        if dones[t]:
            next_val = 0
            last_advantage = 0
        
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = delta + gamma * lam * last_advantage
        last_advantage = advantages[t]
    
    returns = advantages + np.array(values)
    return advantages, returns


# ========================
# PPO UPDATE
# ========================
def ppo_update(model, optimizer, buffer, next_value, batch_size):
    """
    Perform PPO clipped objective update.
    
    TODO: Implement the PPO update:
        1. Compute GAE advantages and returns
        2. Normalize advantages (mean=0, std=1)
        3. For NUM_EPOCHS:
            a. Shuffle data, iterate in mini-batches of batch_size
            b. evaluate_actions → new log_probs, entropy, values
            c. ratio = exp(new_log_prob - old_log_prob)
            d. surr1 = ratio * advantages
            e. surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
            f. policy_loss = -min(surr1, surr2).mean()
            g. value_loss  = MSE(values, returns)
            h. loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy
            i. Backprop and step
    """
    obs = np.array(buffer.obs)
    actions = np.array(buffer.actions)
    old_log_probs = np.array(buffer.log_probs)
    
    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones, next_value
    )
    
    # Convert to tensors
    obs_t = torch.FloatTensor(obs)
    actions_t = torch.FloatTensor(actions)
    old_log_probs_t = torch.FloatTensor(old_log_probs)
    advantages_t = torch.FloatTensor(advantages)
    returns_t = torch.FloatTensor(returns)
    
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    
    for epoch in range(NUM_EPOCHS):
        new_log_prob, entropy, new_value = model.evaluate_actions(obs_t, actions_t)
        ratio = torch.exp(new_log_prob - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_value, returns_t)
        loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ========================
# AGENT WRAPPER
# ========================
class AgentRunner:
    """Wraps a model + buffer + optimizer for one agent."""
    def __init__(self, name, model, lr, buffer_size, batch_size, writer_tag):
        self.name = name
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.writer_tag = writer_tag
        self.episode_reward = 0.0


# ========================
# MAIN TRAINING LOOP
# ========================
def main():
    parser = argparse.ArgumentParser(description="Run Adversarial Football RL Training")
    parser.add_argument("--restart", action="store_true", help="Start training from scratch, ignoring saved checkpoints")
    parser.add_argument("--inference", action="store_true", help="Run in inference mode (no training, no jitter)")
    args = parser.parse_args()

    os.makedirs(f"{CHECKPOINT_DIR}/striker", exist_ok=True)
    os.makedirs(f"{CHECKPOINT_DIR}/goalkeeper", exist_ok=True)
    
    # Connect to Unity
    env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
    print("Environment connected! Press Play in Unity.")
    env.reset()
    
    # Find behavior names
    striker_name, gk_name = None, None
    for name in env.behavior_specs:
        if STRIKER_BEHAVIOR in name:
            striker_name = name
        if GK_BEHAVIOR in name:
            gk_name = name
    
    print(f"Found behaviors: Striker={striker_name}, GK={gk_name}")
    
    if striker_name is None or gk_name is None:
        print("ERROR: Could not find both behaviors!")
        env.close()
        return
    
    # Get specs
    striker_spec = env.behavior_specs[striker_name]
    gk_spec = env.behavior_specs[gk_name]
    
    striker_obs = striker_spec.observation_specs[0].shape[0]
    striker_act = striker_spec.action_spec.continuous_size
    gk_obs = gk_spec.observation_specs[0].shape[0]
    gk_act = gk_spec.action_spec.continuous_size
    
    print(f"Striker: obs={striker_obs}, act={striker_act}")
    print(f"Goalkeeper: obs={gk_obs}, act={gk_act}")
    
    # Initialize agents
    writer = SummaryWriter(log_dir="runs/adversarial")
    
    striker = AgentRunner(
        name=striker_name,
        model=StrikerActorCritic(striker_obs, striker_act),
        lr=STRIKER_LR,
        buffer_size=STRIKER_BUFFER,
        batch_size=STRIKER_BATCH,
        writer_tag="striker"
    )
    
    goalkeeper = AgentRunner(
        name=gk_name,
        model=GoalkeeperActorCritic(gk_obs, gk_act),
        lr=GK_LR,
        buffer_size=GK_BUFFER,
        batch_size=GK_BATCH,
        writer_tag="goalkeeper"
    )
    
    agents = [striker, goalkeeper]
    total_steps = 0
    
    # Track moving average of rewards
    striker_rewards = []
    gk_rewards = []
    start_episode = 0
    
    # Load checkpoints if they exist and --restart is NOT passed
    if not args.restart:
        striker_pt = f"{CHECKPOINT_DIR}/striker/final.pt"
        gk_pt = f"{CHECKPOINT_DIR}/goalkeeper/final.pt"
        
        if os.path.exists(striker_pt):
            striker.model.load_state_dict(torch.load(striker_pt))
            print(f"Loaded Striker weights from {striker_pt}")
            
            # Find the highest episode checkpoint to resume counting
            import glob
            import re
            pt_files = glob.glob(f"{CHECKPOINT_DIR}/striker/ep*.pt")
            highest_ep = 0
            for pt in pt_files:
                match = re.search(r'ep(\d+)\.pt', pt)
                if match:
                    ep_num = int(match.group(1))
                    if ep_num > highest_ep:
                        highest_ep = ep_num
            if highest_ep > 0:
                start_episode = highest_ep
                print(f"Resuming episode counter from {start_episode}")
        
        if os.path.exists(gk_pt):
            goalkeeper.model.load_state_dict(torch.load(gk_pt))
            print(f"Loaded Goalkeeper weights from {gk_pt}")
    else:
        print("Restart flag found: Starting training from scratch (overwriting old checkpoints).")
        
    try:
        for episode in range(start_episode, MAX_EPISODES):
            env.reset()
            for agent in agents:
                agent.episode_reward = 0.0
            done = False
            ep_steps = 0
            
            while not done:
                # Collect obs and send actions for each agent
                agent_data = {}  # Store per-agent data for this step
                
                for agent in agents:
                    decision_steps, terminal_steps = env.get_steps(agent.name)
                    
                    if len(decision_steps) == 0:
                        agent_data[agent.name] = None
                        continue
                    
                    obs = decision_steps.obs[0][0]
                    reward = decision_steps.reward[0]
                    
                    # Get action from model
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        if args.inference:
                            action = agent.model.get_deterministic_action(obs_tensor)
                            log_prob_val = 0.0
                            val_val = 0.0
                        else:
                            action, log_prob, entropy, value = agent.model.get_action_and_value(obs_tensor)
                            log_prob_val = log_prob.item()
                            val_val = value.item()
                    
                    action_np = action.squeeze(0).numpy()
                    action_tuple = ActionTuple(continuous=np.expand_dims(action_np, axis=0))
                    env.set_actions(agent.name, action_tuple)
                    
                    agent.episode_reward += reward
                    agent_data[agent.name] = (obs, action_np, log_prob_val, reward, val_val)
                
                env.step()
                total_steps += 1
                ep_steps += 1
                
                # Safety net: force end episode if too long
                if ep_steps >= MAX_STEPS_PER_EPISODE:
                    done = True
                
                # Check for episode end and store transitions
                for agent in agents:
                    if agent_data.get(agent.name) is None:
                        continue
                        
                    # Get terminal steps
                    _, terminal_steps = env.get_steps(agent.name)
                    is_done = len(terminal_steps) > 0
                    
                    # Store experience
                    obs, action_np, lp, rew, val = agent_data[agent.name]
                    
                    if not args.inference:
                        agent.buffer.store(obs, action_np, lp, rew, is_done, val)
                    
                    # Handle terminal reward safely
                    if is_done:
                        term_reward = terminal_steps.reward[0]
                        agent.episode_reward += term_reward
                        agent_data[agent.name] = None # Prevent pulling this agent again this episode
                    
                    # Training Update
                    if not args.inference and len(agent.buffer) >= agent.buffer_size:
                        ds, _ = env.get_steps(agent.name)
                        if len(ds) > 0:
                            next_obs = torch.FloatTensor(ds.obs[0][0]).unsqueeze(0)
                            with torch.no_grad():
                                next_val = agent.model.get_value(next_obs)
                                next_val = next_val.item() if next_val is not None else 0.0
                        else:
                            next_val = 0.0
                        
                        ppo_update(agent.model, agent.optimizer, agent.buffer, next_val, agent.batch_size)
                        agent.buffer.clear()
                        print(f"  [{agent.writer_tag} PPO update at step {total_steps}]")
                
                # Check for termination AFTER processing both agents
                _, striker_term = env.get_steps(striker.name)
                if len(striker_term) > 0:
                    done = True
            
            # Logging
            for agent in agents:
                writer.add_scalar(f"{agent.writer_tag}/episode_reward", agent.episode_reward, episode)
                
            striker_rewards.append(striker.episode_reward)
            gk_rewards.append(goalkeeper.episode_reward)
            
            # Keep only last 100 for moving average
            if len(striker_rewards) > 100: striker_rewards.pop(0)
            if len(gk_rewards) > 100: gk_rewards.pop(0)
            
            if episode % 10 == 0:
                s_avg = sum(striker_rewards[-10:]) / min(10, len(striker_rewards))
                g_avg = sum(gk_rewards[-10:]) / min(10, len(gk_rewards))
                print(f"Ep {episode} | Striker (Avg 10): {s_avg:.2f} | GK (Avg 10): {g_avg:.2f} | Steps: {total_steps}")
            
            if not args.inference and episode % SAVE_INTERVAL == 0 and episode > 0:
                torch.save(striker.model.state_dict(), f"{CHECKPOINT_DIR}/striker/ep{episode}.pt")
                torch.save(goalkeeper.model.state_dict(), f"{CHECKPOINT_DIR}/goalkeeper/ep{episode}.pt")
                print(f"  Saved checkpoints at episode {episode}")
    
    except KeyboardInterrupt:
        print("\nTraining stopped.")
    finally:
        torch.save(striker.model.state_dict(), f"{CHECKPOINT_DIR}/striker/final.pt")
        torch.save(goalkeeper.model.state_dict(), f"{CHECKPOINT_DIR}/goalkeeper/final.pt")
        writer.close()
        env.close()
        print("Environment closed. Final models saved.")


if __name__ == "__main__":
    main()
