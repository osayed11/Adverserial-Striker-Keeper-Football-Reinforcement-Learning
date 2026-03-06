using UnityEngine;
using Unity.MLAgents;

public class GoalTrigger : MonoBehaviour
{
    public WalkerAgent striker;

    private void Start()
    {
        if (striker == null)
            striker = FindFirstObjectByType<WalkerAgent>();
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Ball") && striker != null)
        {
            // GOAL SCORED! Striker gets big reward.
            // Pass the ball's Y-position to the Striker so it can add a bonus for high shots.
            striker.OnGoalScored(other.transform.position.y);
        }
    }
}
