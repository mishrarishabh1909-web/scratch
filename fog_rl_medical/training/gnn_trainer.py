"""
GNN-RL Trainer - Graph Neural Network enhanced hierarchical training
Trains GNN agent alongside low-level policies for resource allocation
"""

from fog_rl_medical.simulation.workload_generator import WorkloadGenerator
from fog_rl_medical.environment.yafs_wrapper import EnvironmentFactory
from fog_rl_medical.ingestion.stream_receiver import StreamReceiver
from fog_rl_medical.ingestion.modality_tagger import ModalityTagger
from fog_rl_medical.ingestion.normalizer import Normalizer
from fog_rl_medical.multimodal.fusion_engine import FusionEngine
from fog_rl_medical.priority.priority_engine import PriorityEngine
from fog_rl_medical.agents.low_level_policy import LowLevelPolicy
from fog_rl_medical.agents.gnn_rl_agent import GNNRLAgent
from fog_rl_medical.training.metrics import MetricsRecorder

import numpy as np
import torch


class GNNRLTrainer:
    """Trainer for GNN-RL approach"""
    
    def __init__(self, config=None, use_yafs=False):
        """
        Initialize GNN-RL trainer
        
        Args:
            config: Configuration dictionary
            use_yafs: Whether to use YAFS simulator
        """
        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.config = config or {}
        self.num_episodes = self.config.get('rl', {}).get('training', {}).get('num_episodes', 500)
        self.env = EnvironmentFactory.create_environment(self.config, use_yafs=use_yafs)
        
        self.workload_gen = WorkloadGenerator(self.config)
        self.receiver = StreamReceiver()
        self.tagger = ModalityTagger()
        self.normalizer = Normalizer()
        self.fusion = FusionEngine()
        self.priority_engine = PriorityEngine()
        
        self.metrics = MetricsRecorder()
        
        num_nodes = self.env.num_nodes
        
        # GNN agent for high-level node selection
        self.gnn_agent = GNNRLAgent(num_nodes, self.config, device=self.device)
        
        # Low-level policies for resource allocation
        self.low_policies = {i: LowLevelPolicy() for i in range(num_nodes + 1)}
    
    def run(self):
        """Run training for GNN-RL"""
        print("[GNN-RL] Starting training...")
        
        for ep in range(self.num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            episode_states = []
            episode_assignments = []
            episode_priorities = []
            
            for step in range(50):
                # Generate and process tasks
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
                
                # Multimodal fusion and priority assignment
                fused = self.fusion.process_multimodal(tasks, f"p_{ep}_{step}")
                priority_task = self.priority_engine.assign(fused, task_id=f"t_{ep}_{step}")
                episode_priorities.append(priority_task.priority)
                
                # GNN agent selects best node
                action = self.gnn_agent.act(state, priority_task, evaluate=False)
                
                # Handle cloud action (action == num_nodes means use cloud)
                # Note: step() expects 1-based node_ids: 0=cloud, 1..num_nodes=fog nodes
                if action == self.env.num_nodes:
                    node_id = 0  # Use cloud
                    resource_alloc = np.array([0.5, 0.5, 0.5])  # Default cloud allocation
                else:
                    node_id = action + 1  # Convert 0-based action to 1-based node_id
                    node_idx = action
                    # Low-level policy allocates resources
                    node_slice = np.concatenate([
                        [state.cpu_utilization[node_idx]],
                        [state.memory_utilization[node_idx]],
                        [state.bandwidth_utilization[node_idx]],
                        [state.queue_depths[node_idx]],
                        [state.sla_violations[node_idx]]
                    ])
                    task_features = np.concatenate([
                        [priority_task.priority / 4.0],
                        [getattr(priority_task, 'modality_id', 0) / 5.0],
                        [priority_task.deadline / 200.0],
                        np.random.randn(29)  # Padding to 32 dims
                    ])
                    resource_alloc = self.low_policies[node_id].allocate(node_slice, task_features)
                
                # Step environment (note: step expects 1-based node_ids)
                assignments = {priority_task.task_id: node_id}
                allocations = {priority_task.task_id: resource_alloc}
                
                next_state, reward, done, info = self.env.step(assignments, allocations, [priority_task])
                total_reward += reward
                episode_states.append(next_state)
                episode_assignments.append(action)  # Store 0-based action
                
                # Store experience for GNN training
                experience_state = (
                    self.gnn_agent.extract_node_features(state, self.env.num_nodes),
                    self.gnn_agent.extract_task_features(priority_task)
                )
                next_experience_state = (
                    self.gnn_agent.extract_node_features(next_state, self.env.num_nodes),
                    self.gnn_agent.extract_task_features(priority_task)
                )
                
                self.gnn_agent.remember(experience_state, action, reward, next_experience_state, done)
                
                # Training
                self.gnn_agent.replay()
                self.gnn_agent.update_epsilon()
            
            # Record metrics
            self.metrics.record_episode(
                sla_comp=100.0 if all(np.sum(s.sla_violations) == 0 for s in episode_states) else 50.0,
                latency=np.mean([s.buffer_level for s in episode_states]) if episode_states else 0,
                offload_ratio=np.mean([1.0 if a == 0 else 0.0 for a in episode_assignments]) if episode_assignments else 0,
                energy=np.mean([s.energy_consumption for s in episode_states]) if episode_states else 0
            )
            
            if (ep + 1) % 50 == 0:
                print(f"  Episode {ep + 1}/{self.num_episodes} - Reward: {total_reward:.2f}, Epsilon: {self.gnn_agent.epsilon:.4f}")
        
        print("[GNN-RL] Training complete!")
