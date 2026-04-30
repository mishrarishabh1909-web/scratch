import torch
import pickle
import os

class ModelStore:
    def __init__(self, save_dir='models/'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, high_policy, low_policy, metrics, episode):
        try:
            torch.save(high_policy.policy_net.state_dict(), f"{self.save_dir}/high_policy_{episode}.pth")
            torch.save(low_policy.policy_net.state_dict(), f"{self.save_dir}/low_policy_{episode}.pth")
            
            with open(f"{self.save_dir}/metrics_{episode}.pkl", "wb") as f:
                pickle.dump(metrics, f)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint - {e}")

    def load_checkpoint(self, high_policy, low_policy, episode):
        high_path = f"{self.save_dir}/high_policy_{episode}.pth"
        low_path = f"{self.save_dir}/low_policy_{episode}.pth"
        
        try:
            if os.path.exists(high_path):
                high_policy.policy_net.load_state_dict(torch.load(high_path))
            if os.path.exists(low_path):
                low_policy.policy_net.load_state_dict(torch.load(low_path))
        except Exception as e:
            print(f"Warning: Failed to load checkpoint - {e}")
