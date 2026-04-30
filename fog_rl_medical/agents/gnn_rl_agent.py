"""
Graph Neural Network enhanced RL Agent for Fog Computing
Learns topology-aware node embeddings for superior task routing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolutional Layer (no external dependencies)"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: [num_nodes, in_features]
            adj: [num_nodes, num_nodes] adjacency matrix
        """
        # Graph convolution: A * X * W
        out = torch.matmul(adj, x)  # [num_nodes, in_features]
        out = torch.matmul(out, self.weight)  # [num_nodes, out_features]
        out = out + self.bias
        return out


class GNNRLNetwork(nn.Module):
    """Graph Neural Network for task routing in fog networks"""
    
    def __init__(self, node_feature_dim, task_feature_dim, num_nodes, hidden_dim=128):
        """
        Args:
            node_feature_dim: Dimension of node features (CPU, memory, bandwidth, etc.)
            task_feature_dim: Dimension of task features (priority, deadline, etc.)
            num_nodes: Number of fog nodes
            hidden_dim: Hidden dimension for GNN layers
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.task_feature_dim = task_feature_dim
        
        # GNN layers for topology-aware embeddings
        self.gcn1 = SimpleGCNLayer(node_feature_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = SimpleGCNLayer(hidden_dim, 64)
        
        # Policy head: Q-values for each node
        self.policy_head = nn.Sequential(
            nn.Linear(64 + task_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_nodes + 1)  # num_nodes + cloud
        )
        
        # Value head: baseline for variance reduction
        self.value_head = nn.Sequential(
            nn.Linear(64 + task_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def build_adjacency_matrix(self, num_nodes):
        """Build adjacency matrix for fog network (fully connected)"""
        adj = torch.ones(num_nodes, num_nodes) * 0.2  # Weak connections
        for i in range(num_nodes):
            adj[i, i] = 1.0  # Self-loops
            # Neighbors get stronger connection
            if i > 0:
                adj[i, i-1] = 0.8
            if i < num_nodes - 1:
                adj[i, i+1] = 0.8
        
        # Normalize adjacency matrix
        rowsum = adj.sum(dim=1, keepdim=True)
        adj = adj / rowsum
        return adj
    
    def forward(self, node_features, task_features, device='cpu'):
        """
        Args:
            node_features: [num_nodes, node_feature_dim]
            task_features: [task_feature_dim]
        Returns:
            q_values: [num_nodes + 1] - Q-value for each possible action (node assignment)
            value: scalar
        """
        num_nodes = node_features.shape[0]
        
        # Build adjacency matrix
        adj = self.build_adjacency_matrix(num_nodes).to(device)
        node_features = node_features.to(device)
        task_features = task_features.to(device)
        
        # GNN forward pass - learn topology-aware embeddings
        x = node_features
        x = self.gcn1(x, adj)           # [num_nodes, hidden_dim]
        x = F.relu(x)
        x = self.gcn2(x, adj)            # [num_nodes, hidden_dim]
        x = F.relu(x)
        x = self.gcn3(x, adj)            # [num_nodes, 64]
        x = F.relu(x)
        
        # Aggregate node embeddings (average pooling)
        aggregated_embedding = x.mean(dim=0)  # [64]
        
        # Concatenate aggregated embedding with task features
        combined = torch.cat([aggregated_embedding, task_features])  # [64 + task_feature_dim]
        
        # Policy and value heads
        q_values = self.policy_head(combined.unsqueeze(0)).squeeze(0)  # [num_nodes + 1]
        value = self.value_head(combined.unsqueeze(0)).squeeze(0)  # Scalar
        
        return q_values, value


class GNNRLAgent:
    """Graph-aware RL agent for medical fog computing"""
    
    def __init__(self, num_nodes, config=None, device=None):
        """
        Args:
            num_nodes: Number of fog nodes
            config: Configuration dictionary
            device: torch device (cuda or cpu)
        """
        self.config = config or {}
        self.num_nodes = num_nodes
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Feature dimensions
        self.node_feature_dim = 5  # [cpu, memory, bandwidth, queue, sla]
        self.task_feature_dim = 3  # [priority, modality, deadline]
        
        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.lr = self.config.get('learning_rate', 0.0008)
        self.batch_size = self.config.get('batch_size', 32)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('epsilon_end', 0.05)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.998)
        self.update_target_freq = 200
        self.steps = 0
        
        # Networks
        self.policy_net = GNNRLNetwork(
            self.node_feature_dim, 
            self.task_feature_dim, 
            num_nodes,
            hidden_dim=128
        ).to(self.device)
        
        self.target_net = GNNRLNetwork(
            self.node_feature_dim, 
            self.task_feature_dim, 
            num_nodes,
            hidden_dim=128
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=10000)
    
    def extract_node_features(self, cluster_state, num_nodes):
        """Extract node features from cluster state"""
        features = []
        for i in range(num_nodes):
            node_features = [
                cluster_state.cpu_utilization[i],
                cluster_state.memory_utilization[i],
                cluster_state.bandwidth_utilization[i],
                cluster_state.queue_depths[i] / 10.0,  # Normalize
                cluster_state.sla_violations[i]
            ]
            features.append(node_features)
        return torch.FloatTensor(features)
    
    def extract_task_features(self, task):
        """Extract task features"""
        priority = getattr(task, 'priority', 1) / 4.0
        modality_id = getattr(task, 'modality_id', 0) / 5.0
        deadline = getattr(task, 'deadline', 100) / 200.0
        return torch.FloatTensor([priority, modality_id, deadline])
    
    def act(self, cluster_state, task, evaluate=False):
        """Select node for task using GNN-RL"""
        node_features = self.extract_node_features(cluster_state, self.num_nodes)
        task_features = self.extract_task_features(task)
        
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.num_nodes)  # Random action
        
        with torch.no_grad():
            q_values, _ = self.policy_net(node_features, task_features, self.device)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Process each sample in the batch (non-vectorized for simplicity)
        total_loss = 0.0
        for i in range(len(batch)):
            node_features = states[i][0].to(self.device)
            task_features = states[i][1].to(self.device)
            action = torch.LongTensor([actions[i]]).to(self.device)
            reward = torch.FloatTensor([rewards[i]]).to(self.device)
            done = torch.FloatTensor([dones[i]]).to(self.device)
            
            next_node_features = next_states[i][0].to(self.device)
            next_task_features = next_states[i][1].to(self.device)
            
            # Current Q-values
            q_values, _ = self.policy_net(node_features, task_features, self.device)
            q_selected = q_values[action.item()]
            
            # Target Q-values
            with torch.no_grad():
                next_q_values, _ = self.target_net(next_node_features, next_task_features, self.device)
                max_next_q = next_q_values.max()
                target_q = reward[0] + (1 - done[0]) * self.gamma * max_next_q
            
            # Loss - ensure both are scalars for smooth_l1_loss
            loss = F.smooth_l1_loss(q_selected, target_q)
            total_loss += loss.item()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon for exploration-exploitation trade-off"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
