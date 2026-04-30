from fog_rl_medical.agents.base_agent import BaseAgent
import numpy as np

class LowLevelPolicy(BaseAgent):
    def __init__(self, config=None):
        input_dim = 5 + 32  # 5 node slice features + 32 dim task projection
        output_dim = 125    # 5^3 joint actions
        super().__init__(input_dim, output_dim, config)
        
        self.action_space = []
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        for c in fractions:
            for m in fractions:
                for b in fractions:
                    self.action_space.append([c, m, b])

    def extract_state(self, node_slice, task_features):
        task_sub = task_features[:32]
        return np.concatenate([node_slice, task_sub])

    def allocate(self, node_slice, task_features):
        state_vec = self.extract_state(node_slice, task_features)
        action_idx = self.act(state_vec)
        return self._decode_action(action_idx)

    def _decode_action(self, idx):
        return self.action_space[idx]
