"""
Baseline trainers for comparison with Hierarchical DQN.
"""

from fog_rl_medical.simulation.workload_generator import WorkloadGenerator
from fog_rl_medical.environment.fog_cluster import FogClusterEnv
from fog_rl_medical.environment.yafs_wrapper import EnvironmentFactory
from fog_rl_medical.ingestion.stream_receiver import StreamReceiver
from fog_rl_medical.ingestion.modality_tagger import ModalityTagger
from fog_rl_medical.ingestion.normalizer import Normalizer
from fog_rl_medical.multimodal.fusion_engine import FusionEngine
from fog_rl_medical.priority.priority_engine import PriorityEngine
from fog_rl_medical.training.metrics import MetricsRecorder
from fog_rl_medical.agents.baseline_agents import StandaloneDQNAgent, SimpleHierarchicalAgent, RandomAllocationAgent

import numpy as np


class StandaloneDQNTrainer:
    """Trainer for standalone DQN baseline."""
    
    def __init__(self, config=None, use_yafs=False):
        self.config = config or {}
        # IMPROVED: Increased to 500 episodes for comprehensive comparison
        self.num_episodes = self.config.get('rl', {}).get('training', {}).get('num_episodes', 500)
        # YAFS Integration: Optional YAFS environment
        self.env = EnvironmentFactory.create_environment(self.config, use_yafs=use_yafs)
        
        self.workload_gen = WorkloadGenerator(self.config)
        self.receiver = StreamReceiver()
        self.tagger = ModalityTagger()
        self.normalizer = Normalizer()
        self.fusion = FusionEngine()
        self.priority_engine = PriorityEngine()
        
        self.metrics = MetricsRecorder()
        
        num_nodes = self.env.num_nodes
        self.agent = StandaloneDQNAgent(num_nodes, self.config)

    def run(self):
        print("[Standalone DQN] Starting training...")
        for ep in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_states = []
            episode_assignments = []
            episode_priorities = []
            
            for step in range(50):  # IMPROVED: Increased from 10 to 50 steps
                # Ingestion
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
                
                # Fusion and Priority
                fused = self.fusion.process_multimodal(tasks, f"p_{ep}_{step}")
                priority_task = self.priority_engine.assign(fused, task_id=f"t_{ep}_{step}")
                
                episode_priorities.append(priority_task.priority)
                
                # Single DQN action
                state_vec = np.concatenate([
                    state.cpu_utilization,
                    state.memory_utilization,
                    state.bandwidth_utilization,
                    state.queue_depths,
                    state.sla_violations,
                    state.priority_distribution,
                    np.array([state.buffer_level]),
                    np.array([state.timestep / 500.0])
                ])
                
                action_idx = self.agent.act(state_vec)
                node_placement, resource_alloc = self.agent.decode_action(action_idx)
                
                # Environment step
                assignments = {priority_task.task_id: node_placement}
                allocations = {priority_task.task_id: resource_alloc}
                
                next_state, reward, done, _ = self.env.step(assignments, allocations, [priority_task])
                total_reward += reward
                episode_states.append(next_state)
                episode_assignments.append(node_placement)
                
                # Store experience
                self.agent.memory.push(state_vec, action_idx, reward, np.zeros_like(state_vec), done)
                
                # Training
                self.agent.replay()
                self.agent.update_epsilon()
                
                if done:
                    break
            
            # Metrics
            avg_sla = np.mean([np.mean(1 - s.sla_violations) for s in episode_states]) if episode_states else 0.85
            avg_latency = np.mean([100 + np.mean(s.cpu_utilization) * 50 for s in episode_states]) if episode_states else 120.0
            cloud_ratio = sum(1 for a in episode_assignments if a == 0) / len(episode_assignments) if episode_assignments else 0.1
            total_energy = episode_states[-1].energy_consumption if episode_states else 0.0
            
            if episode_priorities:
                priority_counts = np.zeros(4)
                for pri in episode_priorities:
                    if 1 <= pri <= 4:
                        priority_counts[pri - 1] += 1
                priority_distribution = priority_counts / len(episode_priorities)
            else:
                priority_distribution = np.zeros(4)
            
            final_state = episode_states[-1] if episode_states else self.env.reset()
            final_state.priority_distribution = priority_distribution
            
            self.metrics.record_episode(avg_sla, avg_latency, cloud_ratio, total_energy, final_state)
            
            if (ep + 1) % 50 == 0:
                print(f"[Standalone DQN] Episode {ep+1}/{self.num_episodes}")


class SimpleHierarchicalTrainer:
    """Trainer for simple hierarchical (non-DQN) baseline."""
    
    def __init__(self, config=None, use_yafs=False):
        self.config = config or {}
        # IMPROVED: Increased to 500 episodes for fair comparison
        self.num_episodes = self.config.get('rl', {}).get('training', {}).get('num_episodes', 500)
        # YAFS Integration: Optional YAFS environment
        self.env = EnvironmentFactory.create_environment(self.config, use_yafs=use_yafs)
        
        self.workload_gen = WorkloadGenerator(self.config)
        self.receiver = StreamReceiver()
        self.tagger = ModalityTagger()
        self.normalizer = Normalizer()
        self.fusion = FusionEngine()
        self.priority_engine = PriorityEngine()
        
        self.metrics = MetricsRecorder()
        
        num_nodes = self.env.num_nodes
        self.agent = SimpleHierarchicalAgent(num_nodes, self.config)

    def run(self):
        print("[Simple Hierarchical] Starting training...")
        for ep in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_states = []
            episode_assignments = []
            episode_priorities = []
            
            for step in range(50):  # IMPROVED: Increased from 10 to 50 steps
                # Ingestion
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
                
                # Fusion and Priority
                fused = self.fusion.process_multimodal(tasks, f"p_{ep}_{step}")
                priority_task = self.priority_engine.assign(fused, task_id=f"t_{ep}_{step}")
                
                episode_priorities.append(priority_task.priority)
                
                # Heuristic-based assignment
                assignments = self.agent.select_node(state, [priority_task])
                alloc = self.agent.allocate_resources(None, priority_task)
                
                allocations = {priority_task.task_id: alloc}
                
                next_state, reward, done, _ = self.env.step(assignments, allocations, [priority_task])
                total_reward += reward
                episode_states.append(next_state)
                episode_assignments.append(assignments[priority_task.task_id])
                
                if done:
                    break
            
            # Metrics
            avg_sla = np.mean([np.mean(1 - s.sla_violations) for s in episode_states]) if episode_states else 0.85
            avg_latency = np.mean([100 + np.mean(s.cpu_utilization) * 50 for s in episode_states]) if episode_states else 120.0
            cloud_ratio = sum(1 for a in episode_assignments if a == 0) / len(episode_assignments) if episode_assignments else 0.1
            total_energy = episode_states[-1].energy_consumption if episode_states else 0.0
            
            if episode_priorities:
                priority_counts = np.zeros(4)
                for pri in episode_priorities:
                    if 1 <= pri <= 4:
                        priority_counts[pri - 1] += 1
                priority_distribution = priority_counts / len(episode_priorities)
            else:
                priority_distribution = np.zeros(4)
            
            final_state = episode_states[-1] if episode_states else self.env.reset()
            final_state.priority_distribution = priority_distribution
            
            self.metrics.record_episode(avg_sla, avg_latency, cloud_ratio, total_energy, final_state)
            
            if (ep + 1) % 50 == 0:
                print(f"[Simple Hierarchical] Episode {ep+1}/{self.num_episodes}")


class RandomAllocationTrainer:
    """Trainer for random allocation baseline."""
    
    def __init__(self, config=None, use_yafs=False):
        self.config = config or {}
        # IMPROVED: Increased to 500 episodes for fair comparison
        self.num_episodes = self.config.get('rl', {}).get('training', {}).get('num_episodes', 500)
        # YAFS Integration: Optional YAFS environment
        self.env = EnvironmentFactory.create_environment(self.config, use_yafs=use_yafs)
        
        self.workload_gen = WorkloadGenerator(self.config)
        self.receiver = StreamReceiver()
        self.tagger = ModalityTagger()
        self.normalizer = Normalizer()
        self.fusion = FusionEngine()
        self.priority_engine = PriorityEngine()
        
        self.metrics = MetricsRecorder()
        
        num_nodes = self.env.num_nodes
        self.agent = RandomAllocationAgent(num_nodes, self.config)

    def run(self):
        print("[Random Allocation] Starting training...")
        for ep in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_states = []
            episode_assignments = []
            episode_priorities = []
            
            for step in range(50):  # IMPROVED: Increased from 10 to 50 steps
                # Ingestion
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
                
                # Fusion and Priority
                fused = self.fusion.process_multimodal(tasks, f"p_{ep}_{step}")
                priority_task = self.priority_engine.assign(fused, task_id=f"t_{ep}_{step}")
                
                episode_priorities.append(priority_task.priority)
                
                # Random assignment
                assignments = self.agent.select_node(state, [priority_task])
                alloc = self.agent.allocate_resources(None, priority_task)
                
                allocations = {priority_task.task_id: alloc}
                
                next_state, reward, done, _ = self.env.step(assignments, allocations, [priority_task])
                total_reward += reward
                episode_states.append(next_state)
                episode_assignments.append(assignments[priority_task.task_id])
                
                if done:
                    break
            
            # Metrics
            avg_sla = np.mean([np.mean(1 - s.sla_violations) for s in episode_states]) if episode_states else 0.85
            avg_latency = np.mean([100 + np.mean(s.cpu_utilization) * 50 for s in episode_states]) if episode_states else 120.0
            cloud_ratio = sum(1 for a in episode_assignments if a == 0) / len(episode_assignments) if episode_assignments else 0.1
            total_energy = episode_states[-1].energy_consumption if episode_states else 0.0
            
            if episode_priorities:
                priority_counts = np.zeros(4)
                for pri in episode_priorities:
                    if 1 <= pri <= 4:
                        priority_counts[pri - 1] += 1
                priority_distribution = priority_counts / len(episode_priorities)
            else:
                priority_distribution = np.zeros(4)
            
            final_state = episode_states[-1] if episode_states else self.env.reset()
            final_state.priority_distribution = priority_distribution
            
            self.metrics.record_episode(avg_sla, avg_latency, cloud_ratio, total_energy, final_state)
            
            if (ep + 1) % 50 == 0:
                print(f"[Random Allocation] Episode {ep+1}/{self.num_episodes}")
