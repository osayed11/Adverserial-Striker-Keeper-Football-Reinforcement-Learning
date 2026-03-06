using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class GoalkeeperAgent : Agent
{
    public Transform ball;
    public WalkerAgent walkerAgent; // Reference to striker — for leg observations + episode sync
    public Transform goal;
    public float moveSpeed = 12f;
    public float boundaryX = 2f; // Half-width of the goal
    public float kickDetectionSpeed = 0.1f; // Lowered to detect soft shots

    private Rigidbody rb;
    private Rigidbody ballRb;
    private Vector3 startingPosition;
    private bool ballKicked = false;
    private bool hasDecided = false;
    private float targetPositionX = 0f; // Exact target on goal line

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        startingPosition = transform.localPosition;

        if (rb != null)
        {
            rb.isKinematic = false;
            rb.useGravity = false;
            // Lock movement to X axis only, and lock all rotation
            rb.constraints = RigidbodyConstraints.FreezePositionY | 
                             RigidbodyConstraints.FreezePositionZ | 
                             RigidbodyConstraints.FreezeRotation;
        }
        
        // Auto-find references if not assigned in Inspector
        if (ball == null)
        {
            var ballObj = GameObject.FindWithTag("Ball");
            if (ballObj != null) ball = ballObj.transform;
        }
        if (walkerAgent == null)
            walkerAgent = FindFirstObjectByType<WalkerAgent>();
        if (goal == null)
        {
            var goalObj = GameObject.Find("Goal");
            if (goalObj != null) goal = goalObj.transform;
        }
        // Fallback: use walker's goal reference (we know it works)
        if (goal == null && walkerAgent != null && walkerAgent.goal != null)
            goal = walkerAgent.goal;
        
        if (ball != null)
            ballRb = ball.GetComponent<Rigidbody>();
        
        Debug.Log($"[GK Init] ball={ball != null}, goal={goal != null}, walker={walkerAgent != null}, goalPos={goal?.position}");
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = startingPosition;
        ballKicked = false;
        hasDecided = false;
        targetPositionX = 0f;
        
        if (rb != null && !rb.isKinematic)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }

    void FixedUpdate()
    {
        // Detect when ball is moving fast enough = it was kicked
        if (!ballKicked && ballRb != null && ballRb.linearVelocity.magnitude > kickDetectionSpeed)
        {
            ballKicked = true;
        }

        // PHYSICS-BASED MOVEMENT:
        // Use the Rigidbody to move towards the target instead of teleporting
        if (hasDecided)
        {
            float xDiff = targetPositionX - transform.localPosition.x;
            
            // Set velocity proportional to distance (simple PD-like control)
            // This allows the keeper to use its full moveSpeed and physics
            Vector3 targetVel = new Vector3(xDiff * 10f, 0, 0); 
            targetVel.x = Mathf.Clamp(targetVel.x, -moveSpeed, moveSpeed);
            
            rb.linearVelocity = new Vector3(targetVel.x, rb.linearVelocity.y, 0);
        }
        else
        {
            // Keep the keeper still while waiting
            rb.linearVelocity = new Vector3(0, rb.linearVelocity.y, 0);
        }
    }

    // Public accessors for WalkerAgent to query keeper state
    public bool HasDecided() => hasDecided;
    public float GetTargetPositionX() => targetPositionX; // Return exact target instead of direction
    public float GetStartingPositionX() => startingPosition.x; // To calculate absolute distance moved

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. Observe own position (1 float)
        sensor.AddObservation(transform.localPosition.x);
        
        // 2. Observe ball position and velocity (2 + 3 = 5 floats)
        if (ball != null)
        {
            sensor.AddObservation(ball.localPosition.x);
            sensor.AddObservation(ball.localPosition.z);
            
            if (ballRb != null)
                sensor.AddObservation(ballRb.linearVelocity);
            else
                sensor.AddObservation(Vector3.zero);
        }
        else
        {
            sensor.AddObservation(new float[5]);
        }
        
        // 3. Observe striker's center (Hips) (3 floats)
        if (walkerAgent != null && walkerAgent.hips != null)
        {
            sensor.AddObservation(walkerAgent.hips.position - transform.position);
        }
        else
        {
            sensor.AddObservation(new float[3]);
        }

        // 4. Observe striker's kicking leg (3 + 3 = 6 floats)
        if (walkerAgent != null && walkerAgent.footR != null)
        {
            // Relative Position of the foot (3)
            sensor.AddObservation(walkerAgent.footR.position - transform.position);
            
            // Velocity of the foot (3)
            var footRb = walkerAgent.footR.GetComponent<Rigidbody>();
            sensor.AddObservation(footRb != null ? footRb.linearVelocity : Vector3.zero);
        }
        else
        {
            sensor.AddObservation(new float[6]);
        }
        
        // TOTAL: 1 + 5 + 3 + 6 = 15 floats
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Only process ONE action — the first one after the ball is kicked
        if (!ballKicked || hasDecided)
            return;

        // Action is between -1 and 1
        float moveX = Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
        
        // Convert to actual target position along the goal line
        targetPositionX = moveX * boundaryX;
        hasDecided = true;
        
        Debug.Log($"[GK] Decided to dive to target x={targetPositionX:F2} (action={moveX:F2})");
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxisRaw("Horizontal");
    }
}
