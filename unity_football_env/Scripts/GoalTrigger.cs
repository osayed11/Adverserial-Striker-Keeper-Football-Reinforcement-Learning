using UnityEngine;

public class GoalTrigger : MonoBehaviour
{
    public WalkerAgent agent;

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Ball"))
        {
            agent.SetReward(5f); 
            agent.EndEpisode(); 
        }
    }
}
