# Adversarial Football RL

An advanced multi-agent reinforcement learning environment built with **Unity ML-Agents** and **PyTorch**. This project features a Striker and a Goalkeeper training in an adversarial loop using Proximal Policy Optimization (PPO).

![Project Banner](https://img.shields.io/badge/Unity-ML--Agents-blue)
![Project Banner](https://img.shields.io/badge/PyTorch-PPO-orange)

## 🚀 Key Features

### ⚽️ Striker (The Humanoid)
- **Physics Overdrive:** Custom joint torque amplification (3x) allowing for explosive, high-velocity strikes.
- **Reward Shaping for Power:** Velocity-scaled rewards that highly prioritize "screamers" and powerful goals.
- **Chipping Mechanics:** Enhanced height rewards (+3.0/sec airborne) to encourage lofted and top-corner shots.
- **Lateral Randomization:** Starting positions are randomized laterally to ensure the agent generalizes and doesn't overfit to a single shooting lane.

### 🧤 Goalkeeper (The Defender)
- **Physics-Based Agility:** Real velocity-based movement (12m/s) instead of scripted teleportation, allowing for dynamic saves.
- **Advanced Observation Space:** Sees both the dynamic kicking foot and the stable Striker Hips to "read" shots before they happen.
- **Sharpened Incentives:** High-stakes reward structure (-5.0 for goals, +15.0 for saves, +2.0 for reading direction) to promote decisive defensive behavior.

## 🛠 Setup

### 1. Unity Environment
- Open the `FootballRL` folder in **Unity 6 (or compatible version)**.
- Ensure the **ML-Agents Game SDK** is installed via the Package Manager.
- Open the main training scene.

### 2. Python Training Environment
Install the required dependencies:
```bash
pip install mlagents torch
```

## 🧠 Training

To start the adversarial training session:
```bash
python unity_football_env/main.py
```
*The script will wait for you to press **Play** in the Unity Editor.*

### Training Flags:
- `--restart`: Start from scratch (ignore existing checkpoints).
- `--inference`: Run in visualization mode (non-random, no learning).

## 📊 Monitoring
Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=runs/
```

## 📁 Repository Structure
- `FootballRL/`: The Unity project assets, scripts, and scenes.
- `unity_football_env/`: Python implementation of the PPO algorithm and multi-agent training loop.
- `checkpoints/`: Saved models for both Striker and Goalkeeper.

---
Created as part of the UCL Robot Learning Coursework.
