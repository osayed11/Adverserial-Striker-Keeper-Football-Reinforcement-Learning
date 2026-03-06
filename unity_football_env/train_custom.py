import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

def get_walker_observations(obs_vector):
    """
    Parses the long observation vector from WalkerAgent.cs
    
    Mapping based on WalkerAgent.cs + CollectObservationBodyPart:
    0-26: Global state (Vel, Goal Dist, Rot, Ball, GK)
    27-36: Hips (10 floats: Ground(1), Vel(3), AngVel(3), Pos(3))
    37-51: Chest (15 floats)
    ... (others are 15 each: Spine, Head, ThighL, ShinL, FootL)
    127-141: ThighR (15) <-- YOUR KICKING LEG START
    142-156: ShinR (15)
    157-171: FootR (15)
    """
    res = {
        "ball_pos": obs_vector[18:21],
        "ball_vel": obs_vector[21:24],
        "goalkeeper_pos": obs_vector[24:27],
        "thigh_r": obs_vector[127:142], 
        "shin_r": obs_vector[142:157],
        "foot_r": obs_vector[157:172],
    }
    return res

# 1. Connect to the Unity Editor 
# (You must press Play in Unity AFTER running this script!)
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])

print("Environment connected!")

env.reset()
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

print(f"Behavior Name: {behavior_name}")
print(f"Observation space: {spec.observation_specs}")
print(f"Action space: {spec.action_spec}")

try:
    for episode in range(100):
        env.reset()
        done = False
        
        while not done:
            for b_name in env.behavior_specs:
                current_spec = env.behavior_specs[b_name] # Fetch correct spec for this behavior
                decision_steps, terminal_steps = env.get_steps(b_name)
                
                if len(decision_steps) == 0:
                    continue
                
                # --- ACCESSING OBSERVATIONS ---
                if "Walker" in b_name:
                    # Get observations for the first agent
                    obs = decision_steps.obs[0][0] 
                    data = get_walker_observations(obs)
                    
                    if episode % 10 == 0: # Print occasionally
                        print(f"Ball: {data['ball_pos']} | ThighR: {data['thigh_r']}")

                # Take actions (Random for now)
                action_array = current_spec.action_spec.random_action(len(decision_steps))
                action_tuple = ActionTuple(continuous=action_array.continuous)
                env.set_actions(b_name, action_tuple)
                
                if len(terminal_steps) > 0 and "Walker" in b_name:
                    done = True
            
            env.step()
        print(f"Episode {episode} finished")

except KeyboardInterrupt:
    print("\nTraining stopped manually.")
finally:
    env.close()
    print("Environment closed.")
