using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using BodyPart = Unity.MLAgentsExamples.BodyPart;
using Random = UnityEngine.Random;

public class WalkerAgent : Agent
{
    [Header("Walk Speed")]
    [Range(0.1f, 10)]
    [SerializeField]
    //The walking speed to try and achieve
    private float m_TargetWalkingSpeed = 10;

    public float MTargetWalkingSpeed // property
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }

    const float m_maxWalkingSpeed = 10; //The max walking speed

    //Should the agent sample a new goal velocity each episode?
    //If true, walkSpeed will be randomly set between zero and m_maxWalkingSpeed in OnEpisodeBegin()
    //If false, the goal velocity will be walkingSpeed
    public bool randomizeWalkSpeedEachEpisode;

    //The direction an agent will walk during training.
    private Vector3 m_WorldDirToWalk = Vector3.right;

    [Header("Target To Walk Towards")] public Transform target; //Target the agent will walk towards during training.

    [Header("Body Parts")] public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;
    
    [Header("Football References")]
    public Transform ball;
    public Transform goal;
    public Transform goalkeeper;
    private Rigidbody ballRb;
    private Vector3 startingBallPosition;
    private Vector3 startingAgentPosition;

    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;
    EnvironmentParameters m_ResetParams;
    private int m_WarmupSteps = 0;
    private float m_RightLegXOffset = 0f;
    [SerializeField] [Range(0, 1)] private float m_KickStrengthFloor = 0.3f;
    [SerializeField] private float m_KickingJointOverdrive = 3.0f;

    public override void Initialize()
    {
        m_OrientationCube = GetComponentInChildren<OrientationCubeController>(true);
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>(true);
        
        if (m_OrientationCube == null)
        {
             Debug.LogWarning("OrientationCubeController not found in children. Searching entire prefab...");
             m_OrientationCube = GetComponent<OrientationCubeController>();
        }

        //Setup each body part
        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        m_JdController.SetupBodyPart(chest);
        m_JdController.SetupBodyPart(spine);
        m_JdController.SetupBodyPart(head);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(footL);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);
        m_JdController.SetupBodyPart(footR);
        m_JdController.SetupBodyPart(armL);
        m_JdController.SetupBodyPart(forearmL);
        m_JdController.SetupBodyPart(handL);
        m_JdController.SetupBodyPart(armR);
        m_JdController.SetupBodyPart(forearmR);
        m_JdController.SetupBodyPart(handR);

        // STABILIZATION: Make hips kinematic so it doesn't fall over
        if (hips != null)
        {
            var hipsRb = hips.GetComponent<Rigidbody>();
            if (hipsRb != null) hipsRb.isKinematic = true;
        }

        m_ResetParams = Academy.Instance.EnvironmentParameters;
        
        if (ball)
        {
            ballRb = ball.GetComponent<Rigidbody>();
            startingBallPosition = ball.localPosition;
        }
        startingAgentPosition = transform.localPosition;
        
        // Calculate the lateral offset of the right leg relative to the agent root
        if (thighR != null)
        {
            m_RightLegXOffset = transform.InverseTransformPoint(thighR.position).x;
        }

        // DISABLE FALL RESETS:
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            if (bodyPart.groundContact != null)
            {
                bodyPart.groundContact.agentDoneOnGroundContact = false;
                bodyPart.groundContact.penalizeGroundContact = false;
            }
        }
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        // Reset all of the body parts
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        // LATERAL RANDOMIZATION: Move the entire agent +/- 1.5m on X
        float randomX = Random.Range(-1.5f, 1.5f);
        transform.localPosition = new Vector3(startingAgentPosition.x + randomX, startingAgentPosition.y, startingAgentPosition.z);

        // Reset the root rotation to avoid hidden offsets in TransformPoint
        transform.localRotation = Quaternion.identity;
        hips.rotation = Quaternion.identity;

        UpdateOrientationObjects();

        m_TargetWalkingSpeed = 0f;
        randomizeWalkSpeedEachEpisode = false;

        //Set our goal walking speed
        MTargetWalkingSpeed =
            randomizeWalkSpeedEachEpisode ? Random.Range(0.1f, m_maxWalkingSpeed) : MTargetWalkingSpeed;

        if (ball)
        {
            // Reduce ball air resistance/damping to allow for screamers
            ballRb.linearDamping = 0.05f; 
            ballRb.angularDamping = 0.05f;
            ballRb.linearVelocity = Vector3.zero;
            ballRb.angularVelocity = Vector3.zero;
            
            // CENTERING ON RIGHT LEG (Robust World Space calculation):
            float lateralWiggle = Random.Range(-0.05f, 0.05f); 
            float forwardWiggle = Random.Range(0.35f, 0.55f); 
            
            // We use the hips as the stabilized center position
            float targetX = hips.position.x + m_RightLegXOffset + lateralWiggle;
            float targetZ = hips.position.z + forwardWiggle;
            
            // Set ball position (using original Y to match floor height)
            ball.position = new Vector3(targetX, startingBallPosition.y, targetZ);
            
            var ballHandler = ball.GetComponent<BallCollisionHandler>();
            if (ballHandler != null) ballHandler.ResetState();
        }

        m_WarmupSteps = 10; // Hold for 10 frames to ensure user sees it

        // COCKED LEG START: Thigh back, Knee bent BACKWARDS.
        // Flipped signs: Positive X is now backward for this prefab
        float backswingAngle = 50f;
        float kneeBendAngle = 60f; 
        thighR.localRotation = Quaternion.Euler(backswingAngle, 0, 0);
        shinR.localRotation = Quaternion.Euler(kneeBendAngle, 0, 0);

        // Force physics target immediately so it doesn't snap back
        m_JdController.bodyPartsDict[thighR].joint.targetRotation = Quaternion.Euler(-backswingAngle, 0, 0);
        m_JdController.bodyPartsDict[shinR].joint.targetRotation = Quaternion.Euler(-kneeBendAngle, 0, 0);
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround); // Is this bp touching the ground

        //Get velocities in the context of our orientation cube's space
        //Note: You can get these velocities in world space as well but it may not train as well.
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.linearVelocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));

        //Get position relative to hips in the context of our orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - hips.position));

        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR)
        {
            sensor.AddObservation(bp.rb.transform.localRotation);
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_OrientationCube.transform.forward;

        //velocity we want to match
        var velGoal = cubeForward * MTargetWalkingSpeed;
        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        //vel goal relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));

        //rotation deltas
        sensor.AddObservation(Quaternion.FromToRotation(hips.forward, cubeForward));
        sensor.AddObservation(Quaternion.FromToRotation(head.forward, cubeForward));

        //Position of target position relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(target.transform.position));

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var continuousActions = actionBuffers.ContinuousActions;
        var i = -1;

        // FIXED JOINTS: Set everything except the right leg to Identity and high strength
        foreach (var bp in bpDict.Values)
        {
            if (bp.rb.transform == thighR || bp.rb.transform == shinR || bp.rb.transform == footR) continue;
            bp.SetJointTargetRotation(0, 0, 0);
            bp.SetJointStrength(1.0f);
        }

        // KICKING LEG ACTIONS:
        if (m_WarmupSteps > 0)
        {
            // PHYSICALLY LOCK the backswing for visibility
            bpDict[thighR].SetJointTargetRotation(0.8f, 0, 0); // Thigh back (positive)
            bpDict[shinR].SetJointTargetRotation(1.0f, 0, 0);  // Shin back (positive)
            bpDict[footR].SetJointTargetRotation(0, 0, 0);
            
            bpDict[thighR].SetJointStrength(1.0f);
            bpDict[shinR].SetJointStrength(1.0f);
            bpDict[footR].SetJointStrength(1.0f);
            
            m_WarmupSteps--;
            return;
        }

        // Use the first few actions for the kicking leg (actions 0-7)
        bpDict[thighR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[shinR].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[footR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        
        // Strength control for kicking leg (Actions 5-7)
        // We apply a floor to ensure the leg isn't too floppy at start of training
        float tRStrength = (continuousActions[++i] + 1f) * 0.5f;
        float sRStrength = (continuousActions[++i] + 1f) * 0.5f;
        float fRStrength = (continuousActions[++i] + 1f) * 0.5f;

        // PHYSICS OVERDRIVE: Quadruple the spring and force for the kick
        SetOverdrivenJointStrength(bpDict[thighR], Mathf.Max(m_KickStrengthFloor, tRStrength) * 2f - 1f, m_KickingJointOverdrive);
        SetOverdrivenJointStrength(bpDict[shinR], Mathf.Max(m_KickStrengthFloor, sRStrength) * 2f - 1f, m_KickingJointOverdrive);
        SetOverdrivenJointStrength(bpDict[footR], Mathf.Max(m_KickStrengthFloor, fRStrength) * 2f - 1f, m_KickingJointOverdrive);
    }

    /// <summary>
    /// Bypasses the default JointDriveController limits to allow for explosive kicks
    /// </summary>
    void SetOverdrivenJointStrength(BodyPart bp, float strength, float multiplier)
    {
        var rawVal = (strength + 1f) * 0.5f * m_JdController.maxJointForceLimit * multiplier;
        var jd = new JointDrive
        {
            positionSpring = m_JdController.maxJointSpring * multiplier,
            positionDamper = m_JdController.jointDampen,
            maximumForce = rawVal
        };
        bp.joint.slerpDrive = jd;
        bp.currentStrength = jd.maximumForce;
    }


    //Update OrientationCube and DirectionIndicator
    void UpdateOrientationObjects()
    {
        if (target == null || hips == null || m_OrientationCube == null)
            return;

        m_WorldDirToWalk = target.position - hips.position;
        m_OrientationCube.UpdateOrientation(hips, target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    void FixedUpdate()
    {
        // Reward Shaping: Encourage ball being in the air (Incentivize Chipping)
        if (ball != null && ball.localPosition.y > 0.3f)
        {
            // Tripled from 0.02 to 0.06 per frame (~ +3.0 reward per second)
            AddReward(0.06f); 
        }

        if (m_OrientationCube == null)
            return;

        UpdateOrientationObjects();

        if (ball && goal)
        {
            // ===== PER-STEP SHAPING REWARDS =====
            float distanceToBall = Vector3.Distance(hips.position, ball.position);
            AddReward(-distanceToBall * 0.001f);
            
            float ballDistanceToGoal = Vector3.Distance(ball.position, goal.position);
            AddReward(-ballDistanceToGoal * 0.001f);
            
            // Reward ball moving towards goal
            float velocityTowardsGoal = Vector3.Dot(ballRb.linearVelocity, (goal.position - ball.position).normalized);
            if (velocityTowardsGoal > 0)
            {
                // Multiplier increased for more visible power progression
                AddReward(velocityTowardsGoal * 0.1f);
            }

            // ===== RESET CONDITIONS WITH DIRECTIONAL PENALTIES =====
            
            // Ball past the goal line
            if (ball.position.z > goal.position.z)
            {
                bool betweenPosts = Mathf.Abs(ball.position.x) < 2f;
                bool underBar = ball.position.y < 3f;
                
                if (betweenPosts && underBar)
                {
                    // GOAL! Striker scores, OnGoalScored handles rewarding both agents
                    OnGoalScored();
                }
                else
                {
                    // Missed wide over the goal line
                    EndKeeperEpisode("miss");
                    AddReward(-0.5f);
                    EndEpisode();
                }
                return; // Don't check other conditions
            }
            
            // Ball went BACKWARDS (behind the striker) — worst outcome
            if (ball.position.z < -1f)
            {
                EndKeeperEpisode("miss");
                AddReward(-2.0f);
                EndEpisode();
                return;
            }
            
            // Ball went wide to the side
            if (Mathf.Abs(ball.position.x) > 3f)
            {
                EndKeeperEpisode("miss");
                float howFarWide = Mathf.Abs(ball.position.x) - 3f;
                AddReward(-1.5f - howFarWide * 0.2f);
                EndEpisode();
                return;
            }
            
            // Ball fell off the pitch
            if (ball.position.y < -0.5f)
            {
                EndKeeperEpisode("miss");
                AddReward(-1.0f);
                EndEpisode();
                return;
            }
        }
    }

    /// <summary>
    /// End the keeper's episode with appropriate reward BEFORE the ball resets.
    /// </summary>
    private void EndKeeperEpisode(string reason)
    {
        GoalkeeperAgent gk = null;
        if (goalkeeper != null)
            gk = goalkeeper.GetComponent<GoalkeeperAgent>();
        if (gk == null)
            gk = FindFirstObjectByType<GoalkeeperAgent>();
        if (gk == null) return;
        
        // Map the ball's position into the keeper's local space for accurate direction checking
        Vector3 localBallPos = gk.transform.parent != null ? gk.transform.parent.InverseTransformPoint(ball.position) : ball.position;
        
        float reward = (reason == "goal") ? -5.0f : 0f;
        
        // DIRECTIONAL BONUS: Reward "reading" the shot correctly
        // This applies to BOTH goals and misses so the keeper learns to track the ball
        float ballLocalX = localBallPos.x;
        float centerLocalX = gk.GetStartingPositionX(); 
        float keeperLocalX = gk.transform.localPosition.x;
        
        float distFromKeeper = Mathf.Abs(keeperLocalX - ballLocalX);
        float distFromCenter = Mathf.Abs(centerLocalX - ballLocalX);
        bool divedCorrectly = distFromKeeper < distFromCenter;

        if (divedCorrectly) 
        {
            reward += 2.0f; 
            Debug.Log($"[GK {reason}] Correct Dive! Reward: {reward}");
        }
        else
        {
            reward += -2.0f;
            Debug.Log($"[GK {reason}] Wrong way/No dive. Reward: {reward}");
        }
        
        gk.AddReward(reward);
        gk.EndEpisode();
    }

    //Returns the average velocity of all of the body parts
    //Using the velocity of the hips only has shown to result in more erratic movement from the limbs, so...
    //...using the average helps prevent this erratic movement
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;

        //ALL RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.linearVelocity;
        }

        var avgVel = velSum / numOfRb;
        return avgVel;
    }

    //normalized value of the difference in avg speed vs goal walking speed.
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        //distance between our actual velocity and goal velocity
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, MTargetWalkingSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / MTargetWalkingSpeed, 2), 2);
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        AddReward(1f);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Ball"))
        {
            AddReward(1.0f); // Reward for making contact with the ball
        }
        
        // Detect ball hitting the goal post (close but no goal!)
        if (collision.gameObject.name == "PostLeft" || collision.gameObject.name == "PostRight" || collision.gameObject.name == "Crossbar")
        {
            // This fires on the ragdoll body parts, not useful here
        }
    }

    // Called by GoalTrigger when ball enters goal — on-target + scored!
    public void OnGoalScored(float ballHeight = 0f)
    {
        EndKeeperEpisode("goal"); // Guarantee keeper gets evaluated
        
        // BASE REWARD: 10.0f
        float reward = 10.0f;
        
        // 1. HEIGHT BONUS: (0 to +10.0)
        // Doubled to incentivize chipping into top corners
        float heightFactor = Mathf.Clamp(ballHeight, 0f, 2.5f) / 2.5f; 
        float heightBonus = heightFactor * 10.0f;
        
        // 2. POWER BONUS: (Scale based on speed)
        // If speed is 10 m/s, bonus is +10.0. If 20 m/s, bonus is +20.0
        float ballSpeed = ballRb.linearVelocity.magnitude;
        float powerBonus = ballSpeed; 
        
        SetReward(reward + heightBonus + powerBonus);
        Debug.Log($"[GOAL!] Base: {reward} | Height: {heightBonus:F1} | Power: {powerBonus:F1} (Speed: {ballSpeed:F1})");
        EndEpisode();
    }

    // Called by GoalkeeperAgent when keeper saves — on-target but saved
    public void OnShotSaved()
    {
        AddReward(2.0f); // Still positive: shot was on target!
        EndEpisode();
    }
}
