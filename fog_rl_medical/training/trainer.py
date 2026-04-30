from fog_rl_medical.simulation.workload_generator import WorkloadGenerator
from fog_rl_medical.environment.fog_cluster import FogClusterEnv
from fog_rl_medical.environment.yafs_wrapper import EnvironmentFactory
from fog_rl_medical.ingestion.stream_receiver import StreamReceiver
from fog_rl_medical.ingestion.modality_tagger import ModalityTagger
from fog_rl_medical.ingestion.normalizer import Normalizer
from fog_rl_medical.multimodal.fusion_engine import FusionEngine
from fog_rl_medical.priority.priority_engine import PriorityEngine
from fog_rl_medical.agents.high_level_policy import HighLevelPolicy
from fog_rl_medical.agents.low_level_policy import LowLevelPolicy
from fog_rl_medical.agents.hierarchical_trainer import HierarchicalTrainer
from fog_rl_medical.training.metrics import MetricsRecorder

import time
import numpy as np

class Trainer:
    def __init__(self, config=None, use_yafs=False):
        self.config = config or {}
        # IMPROVED: Increased training episodes from 50 to 500 for comprehensive analysis
        self.num_episodes = self.config.get('rl', {}).get('training', {}).get('num_episodes', 500) 
        # YAFS Integration: Create environment with factory (defaults to custom, can use YAFS)
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
        self.high_policy = HighLevelPolicy(num_nodes)
        self.low_policies = {i: LowLevelPolicy() for i in range(1, num_nodes + 1)}
        
        self.hrl_trainer = HierarchicalTrainer(self.high_policy, self.low_policies, self.env)

    def run(self):
        print("Starting training...")
        for ep in range(self.num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            episode_states = []
            episode_assignments = []
            episode_priorities = []  # Track priorities of tasks processed in this episode
            
            for step in range(50): # IMPROVED: Increased from 10 to 50 steps per episode
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
                print(f"Priority Task Assigned: {priority_task}")
                
                # Track the priority of this task
                episode_priorities.append(priority_task.priority)
                
                # HRL
                next_state, reward, done = self.hrl_trainer.train_step([priority_task])
                print(f"Fog Environment State: {next_state}, Reward: {reward}, Done: {done}")
                total_reward += reward
                episode_states.append(next_state)
                
                # Track assignments
                assignments = self.hrl_trainer.high_policy.select_node(state, [priority_task])
                episode_assignments.extend(assignments.values())
                
                self.high_policy.replay()
                for lp in self.low_policies.values():
                    lp.replay()
                    
                if done:
                    break
                    
            print(f"Episode {ep} completed. Reward: {total_reward}")
            # Compute actual metrics from episode
            avg_sla = np.mean([np.mean(1 - s.sla_violations) for s in episode_states]) if episode_states else 0.85
            avg_latency = np.mean([100 + np.mean(s.cpu_utilization) * 50 for s in episode_states]) if episode_states else 120.0
            cloud_ratio = sum(1 for a in episode_assignments if a == 0) / len(episode_assignments) if episode_assignments else 0.1
            # Get total energy consumption from final state
            total_energy = episode_states[-1].energy_consumption if episode_states else 0.0
            
            # Create episode state with priority distribution from processed tasks
            if episode_priorities:
                priority_counts = np.zeros(4)
                for pri in episode_priorities:
                    if 1 <= pri <= 4:
                        priority_counts[pri - 1] += 1
                # Normalize to get distribution
                total_tasks = len(episode_priorities)
                priority_distribution = priority_counts / total_tasks if total_tasks > 0 else np.zeros(4)
            else:
                priority_distribution = np.zeros(4)
            
            # Create a final state with the episode's priority distribution
            final_state = episode_states[-1] if episode_states else self.env.reset()
            final_state.priority_distribution = priority_distribution
            
            self.metrics.record_episode(avg_sla, avg_latency, cloud_ratio, total_energy, final_state)
            
        self.metrics.plot_metrics()
        print("Training completed. Metrics saved.")
