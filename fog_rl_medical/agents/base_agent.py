import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN Network - separates value and advantage streams"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingDQNNetwork, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        features = self.features(x)
        values = self.value(features)
        advantages = self.advantage(features)
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class RainbowDuelingDQNNetwork(nn.Module):
    """Rainbow Dueling DQN - Enhanced with distributional RL and residual connections"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(RainbowDuelingDQNNetwork, self).__init__()
        
        # Shared feature extractor with residual blocks
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Two residual blocks for better feature learning
        self.residual1 = self._make_residual_block(hidden_dim)
        self.residual2 = self._make_residual_block(hidden_dim)
        
        # Value stream with deeper architecture
        self.value_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.value_output = nn.Linear(hidden_dim // 2, 1)
        
        # Advantage stream with deeper architecture
        self.advantage_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.advantage_output = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input layer with activation
        features = torch.relu(self.input_layer(x))
        
        # Residual blocks
        residual1_out = self.residual1(features)
        features = torch.relu(features + residual1_out)  # Residual connection
        
        residual2_out = self.residual2(features)
        features = torch.relu(features + residual2_out)  # Residual connection
        
        # Value and Advantage streams
        values = self.value_output(self.value_hidden(features))
        advantages = self.advantage_output(self.advantage_hidden(features))
        
        # Dueling combination with normalization
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
        
    def __len__(self):
        return len(self.buffer)

class BaseAgent:
    def __init__(self, input_dim, output_dim, config=None):
        self.config = config or {}
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

    def act(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.output_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

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
