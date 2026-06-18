"""
GNN-RL Trainer for Graph-aware task routing in fog networks
"""

from fog_rl_medical.simulation.workload_generator import WorkloadGenerator
from fog_rl_medical.environment.fog_cluster import FogClusterEnv
from fog_rl_medical.environment.yafs_wrapper import EnvironmentFactory
from fog_rl_medical.ingestion.stream_receiver import StreamReceiver
from fog_rl_medical.ingestion.modality_tagger import ModalityTagger
from fog_rl_medical.ingestion.normalizer import Normalizer
from fog_rl_medical.multimodal.fusion_engine import FusionEngine
from fog_rl_medical.priority.priority_engine import PriorityEngine
from fog_rl_medical.agents.gnn_rl_agent import GNNRLAgent
from fog_rl_medical.training.metrics import MetricsRecorder

import numpy as np
import torch


class GNNRLTrainer:
    """Trainer for Graph Neural Network RL agent"""
    
    def __init__(self, config=None, use_yafs=False):
        self.config = config or {}
        self.num_episodes = self.config.get('rl', {}).get('training', {}).get('num_episodes', 500)
        self.use_yafs = use_yafs
        self.env = EnvironmentFactory.create_environment(self.config, use_yafs=use_yafs)
        
        self.workload_gen = WorkloadGenerator(self.config)
        self.receiver = StreamReceiver()
        self.tagger = ModalityTagger()
        self.normalizer = Normalizer()
        self.fusion = FusionEngine()
        self.priority_engine = PriorityEngine()
        
        self.metrics = MetricsRecorder()
        
        num_nodes = self.env.num_nodes
        self.agent = GNNRLAgent(num_nodes, self.config)
        
        print(f"[GNN-RL] Initialized with {num_nodes} fog nodes")

    def run(self):
        print("[GNN-RL] Starting training with Graph Neural Network...")
        
        for ep in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_states = []
            episode_assignments = []
            episode_priorities = []
            
            for step in range(50):  # Steps per episode
                # Ingestion and preprocessing
                raw_tasks = [
                    self.workload_gen.generate_ecg_stream(f"p_{ep}_{step}"),
                    self.workload_gen.generate_vitals(f"p_{ep}_{step}")
                ]
                
                tasks = []
                for rt in raw_tasks:
                    t = self.receiver.receive(rt)
                    t = self.tagger.tag(t)
                    t = self.normalizer.normalize(t)
                    t['features'] = np.random.randn(32) if t['tagged_modality'] == 'ECG' else np.random.randn(16)
                    tasks.append(t)
                
                # Fusion and Priority Assignment
                fused = self.fusion.process_multimodal(tasks, f"p_{ep}_{step}")
                priority_task = self.priority_engine.assign(fused, task_id=f"t_{ep}_{step}")
                
                episode_priorities.append(priority_task.priority)
                
                # GNN-RL Agent Decision
                # Extract node and task features
                node_features = self.agent.extract_node_features(state, self.env.num_nodes)
                task_features = torch.FloatTensor(self.agent.extract_task_features(priority_task))
                
                # Get Q-values and select action
                q_values, value = self.agent.policy_net(
                    node_features, 
                    task_features,
                    device=self.agent.device
                )
                
                # Epsilon-greedy action selection
                if np.random.random() < self.agent.epsilon:
                    node_action = np.random.randint(0, self.env.num_nodes + 1)
                else:
                    node_action = torch.argmax(q_values).item()
                
                # Map action to node
                assignments = {priority_task.task_id: node_action}
                
                # Step environment
                next_state, reward, done = self.env.step(assignments)
                
                # Store in memory for learning
                self.agent.memory.append({
                    'state': state,
                    'node_features': node_features,
                    'task_features': task_features,
                    'action': node_action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })
                
                total_reward += reward
                episode_states.append(next_state)
                episode_assignments.append(node_action)
                
                # Train on batch
                if len(self.agent.memory) >= self.agent.batch_size:
                    self._train_batch()
                
                # Decay epsilon
                self.agent.epsilon = max(
                    self.agent.epsilon_min,
                    self.agent.epsilon * self.agent.epsilon_decay
                )
                
                # Update target network
                self.agent.steps += 1
                if self.agent.steps % self.agent.update_target_freq == 0:
                    self.agent.target_net.load_state_dict(
                        self.agent.policy_net.state_dict()
                    )
                
                state = next_state
                
                if done:
                    break
            
            # Compute metrics
            avg_sla = np.mean([np.mean(1 - s.sla_violations) for s in episode_states]) if episode_states else 0.85
            avg_latency = np.mean([100 + np.mean(s.cpu_utilization) * 50 for s in episode_states]) if episode_states else 120.0
            cloud_ratio = sum(1 for a in episode_assignments if a == 0) / len(episode_assignments) if episode_assignments else 0.1
            total_energy = episode_states[-1].energy_consumption if episode_states else 0.0
            
            # Record metrics
            if episode_priorities:
                priority_dist = np.bincount(episode_priorities, minlength=4)
                self.metrics.add_episode(
                    algo_name='GNN-RL',
                    episode=ep,
                    latency=avg_latency,
                    sla=avg_sla,
                    energy=total_energy,
                    cloud_offload_ratio=cloud_ratio,
                    reward=total_reward,
                    priority_distribution=priority_dist
                )
            
            if (ep + 1) % 50 == 0:
                print(f"[GNN-RL] Episode {ep + 1}: Latency={avg_latency:.2f}ms, "
                      f"SLA={avg_sla*100:.1f}%, Energy={total_energy:.4f}kWh, "
                      f"Reward={total_reward:.2f}")
    
    def _train_batch(self):
        """Train the GNN-RL agent on a batch from memory"""
        import random
        
        batch = random.sample(list(self.agent.memory), self.agent.batch_size)
        
        states = torch.stack([torch.FloatTensor(b['state'].cpu_utilization) for b in batch])
        actions = torch.LongTensor([b['action'] for b in batch])
        rewards = torch.FloatTensor([b['reward'] for b in batch])
        next_states = torch.stack([torch.FloatTensor(b['next_state'].cpu_utilization) for b in batch])
        dones = torch.FloatTensor([b['done'] for b in batch])
        
        node_features = torch.stack([b['node_features'] for b in batch])
        task_features = torch.stack([b['task_features'] for b in batch])
        
        # Forward pass
        q_values, values = [], []
        for i in range(len(batch)):
            q, v = self.agent.policy_net(
                node_features[i:i+1],
                task_features[i:i+1],
                device=self.agent.device
            )
            q_values.append(q)
            values.append(v)
        
        q_values = torch.stack(q_values).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = []
            for i in range(len(batch)):
                next_q, _ = self.agent.target_net(
                    node_features[i:i+1],
                    task_features[i:i+1],
                    device=self.agent.device
                )
                next_q_values.append(next_q)
            
            next_q_values = torch.stack(next_q_values).squeeze()
            target_q = rewards.to(self.agent.device) + self.agent.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones.to(self.agent.device))
        
        # Compute loss
        actions = actions.to(self.agent.device)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(q_selected, target_q)
        
        # Backward pass
        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 1.0)
        self.agent.optimizer.step()
