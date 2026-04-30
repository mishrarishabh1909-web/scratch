from fog_rl_medical.agents.base_agent import BaseAgent, DuelingDQNNetwork, RainbowDuelingDQNNetwork
import torch
import numpy as np

class HighLevelPolicy(BaseAgent):
    def __init__(self, num_nodes, config=None):
        # Increased input dim: cluster state (5*num_nodes + 6) + task features (3)
        input_dim = 5 * num_nodes + 6 + 3
        output_dim = num_nodes + 1  # n fog nodes + 1 cloud node
        super().__init__(input_dim, output_dim, config)
        self.num_nodes = num_nodes
        
        # Replace standard networks with Rainbow Dueling DQN architecture for better performance
        self._setup_rainbow_networks(input_dim, output_dim)
    
    def _setup_rainbow_networks(self, input_dim, output_dim):
        """Use Rainbow Dueling DQN for superior performance"""
        self.policy_net = RainbowDuelingDQNNetwork(input_dim, output_dim, hidden_dim=256)
        self.target_net = RainbowDuelingDQNNetwork(input_dim, output_dim, hidden_dim=256)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Improved learning rate for faster convergence
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

    def extract_state(self, cluster_state):
        """Extract cluster state features"""
        cpu = cluster_state.cpu_utilization
        mem = cluster_state.memory_utilization
        bw = cluster_state.bandwidth_utilization
        q = cluster_state.queue_depths
        sla = cluster_state.sla_violations
        p_dist = cluster_state.priority_distribution
        buf = np.array([cluster_state.buffer_level])
        t = np.array([cluster_state.timestep / 500.0])
        return np.concatenate([cpu, mem, bw, q, sla, p_dist, buf, t])
    
    def extract_task_features(self, task):
        """Extract task-specific features for improved decision making"""
        priority = getattr(task, 'priority', 1) / 4.0  # Normalize to [0, 1]
        modality_id = getattr(task, 'modality_id', 0) / 5.0  # Normalize modality
        deadline = getattr(task, 'deadline', 100) / 200.0  # Normalize deadline
        return np.array([priority, modality_id, deadline])

    def select_node(self, cluster_state, priority_tasks):
        """Task-aware node selection using task features"""
        assignments = {}
        cluster_vec = self.extract_state(cluster_state)
        
        for task in priority_tasks:
            # Include task-specific features in state vector
            task_features = self.extract_task_features(task)
            state_with_task = np.concatenate([cluster_vec, task_features])
            
            # Select best node considering both cluster state and task characteristics
            node_idx = self.act(state_with_task)
            assignments[task.task_id] = node_idx
        
        return assignments
