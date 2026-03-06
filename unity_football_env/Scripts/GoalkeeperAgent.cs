using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class GoalkeeperAgent : Agent
{
    public Transform ball;
    public float moveSpeed = 5f;
    public float boundaryX = 2f; // Half-width of the goal

    private Rigidbody rb;
    private Vector3 startingPosition;
    private bool ballKicked = false;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        startingPosition = transform.localPosition;
    }

    public override void OnEpisodeBegin()
    {
        // Reset to middle
        transform.localPosition = startingPosition;
        ballKicked = false;
        
        if (rb != null && !rb.isKinematic)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }

    /// <summary>
    /// Called by WalkerAgent when the striker makes contact with the ball.
    /// </summary>
    public void NotifyBallKicked()
    {
        ballKicked = true;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Observe own position
        sensor.AddObservation(transform.localPosition.x);
        
        // Observe ball position relative to goal line
        if (ball != null)
        {
            sensor.AddObservation(ball.localPosition.x);
            sensor.AddObservation(ball.localPosition.z);
            
            Rigidbody ballRb = ball.GetComponent<Rigidbody>();
            if (ballRb != null)
                sensor.AddObservation(ballRb.linearVelocity);
            else
                sensor.AddObservation(Vector3.zero);
        }
        else
        {
            sensor.AddObservation(new float[5]); // Fallback
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Don't move until the ball has been kicked
        if (!ballKicked)
            return;
        
        float moveX = actionBuffers.ContinuousActions[0];
        
        // Move position directly (best for kinematic objects)
        Vector3 pos = transform.localPosition;
        pos.x += moveX * moveSpeed * Time.fixedDeltaTime;
        
        // Keep within goal boundaries
        pos.x = Mathf.Clamp(pos.x, -boundaryX, boundaryX);
        transform.localPosition = pos;

        // Small reward for existing
        AddReward(0.01f);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxisRaw("Horizontal");
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Ball"))
        {
            // Big reward for saving the ball!
            AddReward(5.0f);
            EndEpisode();
        }
    }
}
