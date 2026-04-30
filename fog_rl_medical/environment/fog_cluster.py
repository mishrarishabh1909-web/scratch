from dataclasses import dataclass, field
import numpy as np

@dataclass
class ClusterState:
    cpu_utilization: np.ndarray
    memory_utilization: np.ndarray
    bandwidth_utilization: np.ndarray
    queue_depths: np.ndarray
    sla_violations: np.ndarray
    priority_distribution: np.ndarray  # shape (4,)
    buffer_level: float
    timestep: int
    energy_consumption: float = 0.0  # Energy consumed in kWh

class FogClusterEnv:
    def __init__(self, config=None):
        self.config = config or {}
        self.num_nodes = self.config.get('num_fog_nodes', 5)
        # Power specifications for each node type (in Watts)
        # Fog nodes: 40W idle + up to 60W active
        self.node_power_specs = {
            'fog': {'idle': 40.0, 'peak': 100.0},
            'cloud': {'idle': 20.0, 'peak': 100.0}
        }
        self.reset()
        
    def reset(self):
        self.timestep = 0
        self.state = ClusterState(
            cpu_utilization=np.zeros(self.num_nodes),
            memory_utilization=np.zeros(self.num_nodes),
            bandwidth_utilization=np.zeros(self.num_nodes),
            queue_depths=np.zeros(self.num_nodes),
            sla_violations=np.zeros(self.num_nodes),
            priority_distribution=np.zeros(4),  # Reset to zeros each episode
            buffer_level=0.0,
            timestep=self.timestep,
            energy_consumption=0.0
        )
        return self.state

    def step(self, actions, alloc_actions=None, priority_tasks=None):
        """
        actions: node assignment mapping for each incoming task
        alloc_actions: low_level_policy resource allocation per node
        priority_tasks: list of PriorityTask objects
        Return: (ClusterState, rewards, done, info)
        """
        self.timestep += 1
        
        # Update state based on assignments and allocations
        task_dict = {t.task_id: t for t in (priority_tasks or [])}
        
        # Reset priority distribution for this step (track current active tasks)
        current_priority_counts = np.zeros(4)
        
        for task_id, node_id in actions.items():
            task = task_dict.get(task_id)
            if node_id > 0:  # Fog node (not cloud)
                node_idx = node_id - 1
                if alloc_actions and task_id in alloc_actions:
                    alloc = alloc_actions[task_id]  # [cpu_frac, mem_frac, bw_frac]
                    # Update utilizations (cap at 1.0)
                    self.state.cpu_utilization[node_idx] = min(1.0, self.state.cpu_utilization[node_idx] + alloc[0])
                    self.state.memory_utilization[node_idx] = min(1.0, self.state.memory_utilization[node_idx] + alloc[1])
                    self.state.bandwidth_utilization[node_idx] = min(1.0, self.state.bandwidth_utilization[node_idx] + alloc[2])
                    # Update priority distribution - count active tasks by priority
                    if task:
                        pri_idx = task.priority - 1  # 1-4 -> 0-3
                        if 0 <= pri_idx < 4:
                            current_priority_counts[pri_idx] += 1
                else:
                    # If no alloc, just assign to queue
                    self.state.queue_depths[node_idx] += 1
        
        # Update the state with current priority distribution
        self.state.priority_distribution = current_priority_counts
        
        # Simulate processing: reduce utilizations slightly over time
        decay = 0.1
        self.state.cpu_utilization = np.maximum(0, self.state.cpu_utilization - decay)
        self.state.memory_utilization = np.maximum(0, self.state.memory_utilization - decay)
        self.state.bandwidth_utilization = np.maximum(0, self.state.bandwidth_utilization - decay)
        
        # Update SLA violations (mock: if cpu > 0.9, violate)
        self.state.sla_violations = (self.state.cpu_utilization > 0.9).astype(float)
        
        # Calculate energy consumption (Watt-hours per timestep)
        # Energy = Power * Time, assuming 1 timestep = 1 second = 1/3600 hour
        fog_specs = self.node_power_specs['fog']
        energy_per_step = 0.0
        for cpu_util in self.state.cpu_utilization:
            # Power = idle_power + (peak - idle) * cpu_utilization
            power = fog_specs['idle'] + (fog_specs['peak'] - fog_specs['idle']) * cpu_util
            # Convert to kWh for this timestep (1 second = 1/3600 hour)
            energy_per_step += power / 3600.0 / 1000.0  # Convert Watt-hours to kWh
        
        self.state.energy_consumption += energy_per_step
        
        self.state.timestep = self.timestep
        
        # Calculate Reward based on SLA violations and resource utilization
        sla_penalty = -10.0 * np.sum(self.state.sla_violations)
        overload_penalty = -5.0 * np.sum(self.state.cpu_utilization > 0.9)
        balance_reward = 1.0 / (np.var(self.state.cpu_utilization) + 0.1)
        
        reward = 1.0 + sla_penalty + overload_penalty + balance_reward
        reward = float(reward)
        
        done = self.timestep >= self.config.get('episode_length', 10)  # Match trainer's 10 steps
        info = {}
        
        return self.state, reward, done, info
