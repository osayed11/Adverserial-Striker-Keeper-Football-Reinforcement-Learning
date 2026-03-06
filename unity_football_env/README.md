# Unity ML-Agents Football Striker Environment

I have prepared the essential C# scripts and the Python training config file for you.

## Files Provided
- `Scripts/FootballStrikerAgent.cs`: The main agent script to be attached to your Humanoid. Handles resetting the ball/agent and shaping rewards (moving to ball, kicking it towards goal).
- `Scripts/GoalTrigger.cs`: To be attached to the Goal trigger volume. Simply rewards the agent when the ball enters the goal and resets the episode.
- `Config/trainer_config.yaml`: The ML-Agents config file optimized for continuous control problems like humanoid locomotion.

## Unity Setup Instructions
1. Open Unity Hub and create a new **3D Core** project.
2. Open the Package Manager, click the `+` icon, and select "Add package from git URL...". Enter `com.unity.ml-agents`.
3. In your scene, create:
   - A ground Plane.
   - A Goal (Cubes + an invisible trigger volume in the middle with the `GoalTrigger` script).
   - A Ball (Sphere + Rigidbody + Bouncy Physics Material, tagged as `Ball`).
   - A Humanoid (use a prefab from the ML-Agents examples if possible, attached with the `FootballStrikerAgent` script).
4. Add the `Behavior Parameters` component to the Humanoid. Set the Behavior Name to `FootballStriker` and Space Size to `12`.
5. Add a `Decision Requester` component to the Humanoid and set Decision Period to 5.

## Training Instructions
1. Install Python 3.10 and run `pip install mlagents protobuf==3.20.3`.
2. Open a terminal in `unity_football_env/Config`.
3. Run the training command:
   ```bash
   mlagents-learn trainer_config.yaml --run-id=FootballStriker_01
   ```
4. Press **Play** in the Unity Editor when prompted!
