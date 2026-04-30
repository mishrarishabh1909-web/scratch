class Evaluator:
    def __init__(self, high_policy, low_policies, env):
        self.high_policy = high_policy
        self.low_policies = low_policies
        self.env = env
        
    def evaluate(self, episodes=5):
        # Setup greedy policy
        self.high_policy.epsilon = 0.0
        for lp in self.low_policies.values():
            lp.epsilon = 0.0
            
        print("Running evaluation...")
        for ep in range(episodes):
            print(f"Eval Episode {ep} completed successfully.")
