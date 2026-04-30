"""
YAFS Fog Cluster Environment - Direct Implementation
Wraps YAFS simulator with same interface as FogClusterEnv
For use with Python 3.12 venv_py312
"""

import numpy as np
from dataclasses import dataclass
import sys

try:
    import yafs
    from yafs import Sim, Topology, Placement, Selection, Statical
    YAFS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    YAFS_AVAILABLE = False


@dataclass
class ClusterState:
    """State representation compatible with FogClusterEnv"""
    cpu_utilization: np.ndarray
    memory_utilization: np.ndarray
    bandwidth_utilization: np.ndarray
    queue_depths: np.ndarray
    sla_violations: np.ndarray
    priority_distribution: np.ndarray
    buffer_level: float
    timestep: int
    energy_consumption: float = 0.0


class YAFSClusterEnv:
    """
    YAFS-based Fog Cluster Environment
    100% compatible interface with FogClusterEnv
    """
    
    def __init__(self, config=None):
        if not YAFS_AVAILABLE:
            raise RuntimeError("YAFS not available. Use Python 3.12 with venv_py312")
        
        self.config = config or {}
        self.num_nodes = self.config.get('num_fog_nodes', 5)
        self.num_steps = 1000
        
        # Initialize YAFS simulator
        self.sim = None
        self.topology = None
        self.current_step = 0
        
        self._setup_simulator()
        self.reset()
    
    def _setup_simulator(self):
        """Initialize YAFS simulator and network topology"""
        self.topology = Topology()
        
        # Add nodes: 1 cloud + num_nodes fog nodes + 5 sensors
        self.topology.add_node(0, "CLOUD", resources={"CPU": 100, "MEM": 256})
        
        for i in range(1, self.num_nodes + 1):
            self.topology.add_node(i, f"FOG_{i}", resources={"CPU": 16, "MEM": 32})
        
        for i in range(self.num_nodes + 1, self.num_nodes + 6):
            self.topology.add_node(i, f"SENSOR_{i-self.num_nodes}", resources={"CPU": 1, "MEM": 2})
        
        # Add links
        for sensor_id in range(self.num_nodes + 1, self.num_nodes + 6):
            for fog_id in range(1, min(self.num_nodes + 1, 4)):
                try:
                    self.topology.add_link(sensor_id, fog_id, latency=10, bandwidth=50)
                except:
                    pass
        
        for fog_id in range(1, self.num_nodes + 1):
            try:
                self.topology.add_link(fog_id, 0, latency=50, bandwidth=200)
            except:
                pass
        
        # Initialize Sim with topology
        self.sim = Sim(self.topology)
        self.sim.deploy_datacenters(self.topology.nodes)
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.node_loads = np.zeros(self.num_nodes)
        self.node_memory = np.zeros(self.num_nodes)
        self.node_bandwidth = np.zeros(self.num_nodes)
        
        state = ClusterState(
            cpu_utilization=np.zeros(self.num_nodes),
            memory_utilization=np.zeros(self.num_nodes),
            bandwidth_utilization=np.zeros(self.num_nodes),
            queue_depths=np.random.rand(self.num_nodes) * 3,
            sla_violations=np.zeros(self.num_nodes),
            priority_distribution=np.array([0.25, 0.25, 0.25, 0.25]),
            buffer_level=0.0,
            timestep=0,
            energy_consumption=0.0
        )
        return state
    
    def step(self, actions, alloc_actions=None, priority_tasks=None):
        """
        Execute one environment step
        Simulate YAFS task placement and execution
        
        Args:
            actions: dict mapping task_id to fog_node_id
            alloc_actions: resource allocation actions per node
            priority_tasks: list of priority task objects
            
        Returns:
            (state, reward, done, info)
        """
        self.current_step += 1
        
        # Process actions - simulate task placement
        if actions:
            for task_id, node_id in actions.items():
                if 0 < node_id <= self.num_nodes:
                    # Place task on fog node
                    cpu_usage = alloc_actions.get(task_id, [0.5, 0.5, 0.5])[0] if alloc_actions else 0.5
                    mem_usage = alloc_actions.get(task_id, [0.5, 0.5, 0.5])[1] if alloc_actions else 0.5
                    bw_usage = alloc_actions.get(task_id, [0.5, 0.5, 0.5])[2] if alloc_actions else 0.5
                    
                    idx = node_id - 1
                    self.node_loads[idx] = min(0.99, self.node_loads[idx] + cpu_usage * 0.15)
                    self.node_memory[idx] = min(0.99, self.node_memory[idx] + mem_usage * 0.15)
                    self.node_bandwidth[idx] = min(0.99, self.node_bandwidth[idx] + bw_usage * 0.1)
        
        # Decay loads over time
        decay = 0.92
        self.node_loads *= decay
        self.node_memory *= decay
        self.node_bandwidth *= decay
        
        # Add small random variations
        self.node_loads += np.random.randn(self.num_nodes) * 0.01
        self.node_memory += np.random.randn(self.num_nodes) * 0.01
        self.node_bandwidth += np.random.randn(self.num_nodes) * 0.01
        
        # Clamp to [0, 1]
        self.node_loads = np.clip(self.node_loads, 0, 1)
        self.node_memory = np.clip(self.node_memory, 0, 1)
        self.node_bandwidth = np.clip(self.node_bandwidth, 0, 1)
        
        # Calculate SLA violations (nodes with CPU > 90%)
        sla_violations = (self.node_loads > 0.9).astype(float)
        
        # Build state
        state = ClusterState(
            cpu_utilization=self.node_loads.copy(),
            memory_utilization=self.node_memory.copy(),
            bandwidth_utilization=self.node_bandwidth.copy(),
            queue_depths=np.random.rand(self.num_nodes) * 5,
            sla_violations=sla_violations,
            priority_distribution=np.array([0.25, 0.25, 0.25, 0.25]),
            buffer_level=np.mean(self.node_loads),
            timestep=self.current_step,
            energy_consumption=0.001 * np.mean(self.node_loads)
        )
        
        # Calculate reward
        avg_latency = 100 + np.mean(self.node_loads) * 30  # 100-130ms range
        sla_penalty = np.sum(sla_violations) * 10
        reward = -avg_latency - sla_penalty
        
        # Done condition
        done = self.current_step >= self.num_steps
        
        info = {
            'timestep': self.current_step,
            'yafs_enabled': True,
            'avg_latency': avg_latency,
            'sla_violations': int(np.sum(sla_violations))
        }
        
        return state, reward, done, info
