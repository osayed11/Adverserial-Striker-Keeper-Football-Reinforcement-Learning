using UnityEngine;

/// <summary>
/// Attach this to the Ball. Detects collisions with posts (waits for rebound)
/// and with the goalkeeper (triggers save).
/// </summary>
public class BallCollisionHandler : MonoBehaviour
{
    public WalkerAgent striker;
    public GoalkeeperAgent goalkeeper;
    public float postHitWaitTime = 2.0f;

    private bool hitPost = false;
    private float postHitTimer = 0f;

    private void Start()
    {
        if (striker == null)
            striker = FindFirstObjectByType<WalkerAgent>();
        if (goalkeeper == null)
            goalkeeper = FindFirstObjectByType<GoalkeeperAgent>();
    }

    void FixedUpdate()
    {
        // Post-hit timer: wait for potential rebound into goal
        if (hitPost)
        {
            postHitTimer += Time.fixedDeltaTime;
            if (postHitTimer >= postHitWaitTime)
            {
                hitPost = false;
                postHitTimer = 0f;

                // Post hit but no rebound goal — reward striker for being close
                if (striker != null)
                {
                    striker.AddReward(1.0f);
                    striker.EndEpisode();
                }
                // Keeper handles its own episode end via its FixedUpdate
            }
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        // Ball hit a goal post or crossbar
        if (collision.gameObject.name == "PostLeft" || 
            collision.gameObject.name == "PostRight" || 
            collision.gameObject.name == "Crossbar")
        {
            if (!hitPost)
            {
                hitPost = true;
                postHitTimer = 0f;
                Debug.Log("[Ball] Hit post/crossbar! Waiting for rebound...");
            }
        }

        // Ball hit the goalkeeper — SAVE!
        // Check by component OR by name containing "keeper"/"goalkeeper"
        bool isKeeper = false;
        if (collision.gameObject.GetComponent<GoalkeeperAgent>() != null || 
            collision.gameObject.GetComponentInParent<GoalkeeperAgent>() != null)
        {
            isKeeper = true;
        }
        else
        {
            // Fallback: check names of this object and all parents
            Transform curr = collision.gameObject.transform;
            while (curr != null)
            {
                string nameLower = curr.name.ToLower();
                if (nameLower.Contains("keeper") || nameLower.Contains("goalkeeper"))
                {
                    isKeeper = true;
                    break;
                }
                curr = curr.parent;
            }
        }
        
        if (isKeeper)
        {
            Debug.Log("[Ball] SAVED by goalkeeper!");
            
            // Reward keeper for saving
            if (goalkeeper != null)
            {
                goalkeeper.AddReward(15.0f);
                goalkeeper.EndEpisode();
            }

            // Striker gets partial reward (on-target but saved) + end episode
            if (striker != null)
            {
                striker.OnShotSaved();
            }
        }
    }

    public void ResetState()
    {
        hitPost = false;
        postHitTimer = 0f;
    }
}
