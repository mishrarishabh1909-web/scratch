"""
Baseline agents for comparison with Hierarchical DQN.
- StandaloneDQNAgent: Single DQN handling both placement and resource allocation
- SimpleHierarchicalAgent: Non-DQN hierarchical using heuristics
- RandomAllocationAgent: Random baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from fog_rl_medical.agents.base_agent import DQNNetwork, ReplayBuffer


class StandaloneDQNAgent:
    """Single DQN agent handling both placement and resource allocation.
    
    Action space: (num_nodes + 1) * 125 possible actions
    - First num_nodes+1 actions: placement to each node/cloud
    - For each placement, implicitly includes resource allocation (discretized)
    """
    
    def __init__(self, num_nodes, config=None):
        self.config = config or {}
        self.num_nodes = num_nodes
        
        # State: same as hierarchical (cluster state + task info)
        input_dim = 5 * num_nodes + 6  # cluster state features
        # Action: placement (num_nodes+1) X resource allocation (125)
        self.num_placements = num_nodes + 1
        self.num_allocations = 125
        output_dim = self.num_placements * self.num_allocations
        
        self.gamma = self.config.get('gamma', 0.99)
        self.lr = self.config.get('learning_rate', 0.0005)
        self.batch_size = self.config.get('batch_size', 64)
        
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.998)
        
        self.policy_net = DQNNetwork(input_dim, output_dim)
        self.target_net = DQNNetwork(input_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.config.get('memory_size', 50000))
        self.output_dim = output_dim
        
        # Action space mapping
        self.action_fractions = []
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        for c in fractions:
            for m in fractions:
                for b in fractions:
                    self.action_fractions.append([c, m, b])

    def act(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.output_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def decode_action(self, action_idx):
        """Decode combined action into (node_placement, resource_allocation)."""
        placement = action_idx // self.num_allocations
        allocation_idx = action_idx % self.num_allocations
        return placement, self.action_fractions[allocation_idx]

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class SimpleHierarchicalAgent:
    """Non-DQN hierarchical baseline using heuristics and simple Q-tables.
    
    High-level: Uses heuristic-based node selection (least loaded)
    Low-level: Uses heuristic-based resource allocation (proportional to task priority)
    """
    
    def __init__(self, num_nodes, config=None):
        self.config = config or {}
        self.num_nodes = num_nodes
        
        # Simple heuristics without learning
        self.action_fractions = []
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        for c in fractions:
            for m in fractions:
                for b in fractions:
                    self.action_fractions.append([c, m, b])

    def select_node(self, cluster_state, priority_tasks):
        """Select node based on least-loaded heuristic."""
        assignments = {}
        cpu = cluster_state.cpu_utilization
        mem = cluster_state.memory_utilization
        
        # Simple heuristic: select node with minimum (cpu + mem) / 2
        node_loads = (cpu + mem) / 2.0
        
        for task in priority_tasks:
            # High priority tasks go to least loaded node
            # Low priority tasks more likely to go to cloud (node 0)
            if task.priority >= 3:
                # High priority: select best fog node
                best_node = np.argmin(node_loads) + 1  # Nodes are 1-indexed
            else:
                # Low priority: 50% chance cloud, 50% least loaded
                if random.random() < 0.5:
                    best_node = 0  # Cloud
                else:
                    best_node = np.argmin(node_loads) + 1
            
            assignments[task.task_id] = best_node
        
        return assignments

    def allocate_resources(self, node_slice, task):
        """Allocate resources based on task priority."""
        # Higher priority = more resources
        priority_factor = task.priority / 4.0
        
        # Allocate more resources for higher priority
        cpu_frac = min(1.0, 0.2 + priority_factor * 0.8)
        mem_frac = min(1.0, 0.2 + priority_factor * 0.8)
        bw_frac = min(1.0, 0.2 + priority_factor * 0.8)
        
        return [cpu_frac, mem_frac, bw_frac]

    def replay(self):
        """No learning in this baseline."""
        pass

    def update_epsilon(self):
        """No exploration decay in this baseline."""
        pass


class RandomAllocationAgent:
    """Completely random baseline.
    
    Randomly assigns tasks to nodes and randomly allocates resources.
    """
    
    def __init__(self, num_nodes, config=None):
        self.config = config or {}
        self.num_nodes = num_nodes
        
        self.action_fractions = []
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        for c in fractions:
            for m in fractions:
                for b in fractions:
                    self.action_fractions.append([c, m, b])

    def select_node(self, cluster_state, priority_tasks):
        """Randomly assign tasks to any node (fog or cloud)."""
        assignments = {}
        for task in priority_tasks:
            # Random placement: nodes 0 to num_nodes (0 is cloud)
            random_node = random.randint(0, self.num_nodes)
            assignments[task.task_id] = random_node
        return assignments

    def allocate_resources(self, node_slice, task):
        """Randomly allocate resources."""
        return random.choice(self.action_fractions)

    def replay(self):
        """No learning in this baseline."""
        pass

    def update_epsilon(self):
        """No exploration decay in this baseline."""
        pass
