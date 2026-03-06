# Adversarial Football RL: High-Performance Multi-Agent Training

An advanced reinforcement learning project implementing an adversarial training loop between a **Humanoid Striker** and a **Dynamic Goalkeeper** using **Unity ML-Agents** and **PyTorch**.

This project explores the intersection of high-fidelity physics and multi-agent competitive dynamics, featuring custom torque-amplification systems and precise reward shaping to develop "pro-level" football behaviors.

---

## 🚀 Advanced AI Features

### ⚽️ The Striker (Humanoid)
*   **Physics Overdrive System:** Developed a custom torque-injection layer that bypasses default Unity `JointDrive` limits (3x amplification), enabling the Striker to generate explosive 25m/s+ strikes.
*   **Velocity-Scaled Rewards:** Implemented a non-linear reward function `(Base + BallSpeed + HeightBonus)` that incentivizes "screamers" and top-corner finishes over simple nudges.
*   **Chipping Mechanics:** A dedicated airborne reward (+3.0/sec) encourages the emergence of lofted shots and chipped techniques to beat dived keepers.
*   **Anti-Overfitting Lateral Randomization:** The agent's starting X-position is randomized within a ±1.5m range, preventing the Goalkeeper from memorizing shooting patterns and forcing generalized defensive reading.

### 🧤 The Goalkeeper (The Defender)
*   **Physics-Based Reaction:** Movement is strictly physics-driven (using Rigidbody Velocity at 12m/s) rather than Kinematic teleportation, resulting in realistic diving arcs and collision momentum.
*   **Enriched Observation Space (15-D):** The keeper "sees" the dynamic kicking foot velocity + the stable humanoid Hips (center of mass). This stable reference point allows the agent to predict the "lane" of the attack before the strike connects.
*   **Sharpened Decision Incentives:** A high-stakes incentive structure:
    *   **Goal Conceded:** -5.0 penalty.
    *   **Correct Dive (Positioning):** +2.0 bonus (even if a goal is scored).
    *   **Successful Save:** +15.0 massive reward.
    *   *Result:* This creates an 18-point delta that drives the agent to prioritize saves above all else.

---

## 🧠 Technical Architecture

### Adversarial PPO Loop
The system utilizes two independent **Actor-Critic (PPO)** networks training simultaneously in the same environment. This creates a natural curriculum: as the Striker learns more powerful shots, the Goalkeeper is forced to develop faster reaction times and better positioning.

### Observation Space Analysis
*   **Keeper Vision:** [Self-Pos, Ball-Pos, Ball-Vel, Striker-Hips, Striker-Foot, Foot-Vel].
*   **Coordinate Space Sync:** Implemented world-to-local matrix transformations to ensure Goalkeeper rewards are calculated correctly regardless of scene orientation or agent rotation.

---

## 🛠 Setup & Installation

### 1. Unity Environment
- **Platform:** Unity 6 (or compatible).
- **Core SDK:** ML-Agents Game SDK (installed via Package Manager).
- **Scene:** `Humanoid_Football.unity`.

### 2. Python Backend
Clone the repository and install the dependencies:
```bash
pip install mlagents torch numpy tensorboard
```

### 3. Training Commands
Launch the adversarial loop from the root directory:
```bash
python unity_football_env/main.py
```
*Wait for the "Press Play" prompt, then start the Unity Editor.*

### 🛠 Calibration Flags:
- `--restart`: Wipes existing checkpoints and starts training from scratch.
- `--inference`: Disables randomization and training for high-quality visualization.

---

## 📁 Repository Map
- `FootballRL/`: Full Unity Project (Scripts, Prefabs, Environments).
- `unity_football_env/`: Python Training Engine (PPO, Buffers, Multi-Agent Loop).
- `README.md`: Technical documentation.
- `.gitignore`: Configured for Unity + Python + ML-Agents.

---

## 📊 Results Summary
Training results show the emergence of distinct tactical behaviors:
1.  **Striker:** Moves from random leg-flailing to high-power "top corner" aimed shots.
2.  **Goalkeeper:** Transitions from "lazy" center-standing to proactive diving based on the Striker's limb velocity "tells".

*Developed for the UCL Robot Learning Coursework.*
