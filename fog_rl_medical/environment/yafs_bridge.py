"""
YAFS Python 3.12 Subprocess Bridge
Enables Python 3.11 code to use YAFS running in Python 3.12 environment
via inter-process communication
"""

import subprocess
import json
import sys
from pathlib import Path
import numpy as np
import tempfile
import os
from dataclasses import dataclass, asdict


@dataclass
class ClusterState:
    """State representation compatible with FogClusterEnv"""
    cpu_utilization: list  # Convert to list for JSON serialization
    memory_utilization: list
    bandwidth_utilization: list
    queue_depths: list
    sla_violations: list
    priority_distribution: list
    buffer_level: float
    timestep: int
    energy_consumption: float = 0.0


class YAFSSubprocessBridge:
    """
    Subprocess bridge for YAFS integration
    Communicates with Python 3.12 YAFS runner via JSON messages
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.process = None
        self.temp_dir = None
        self._start_subprocess()
    
    def _get_venv_py312_path(self):
        """Get path to Python 3.12 executable"""
        venv_path = Path(__file__).parent.parent.parent / "venv_py312"
        python_exe = venv_path / "Scripts" / "python.exe"
        
        if python_exe.exists():
            return str(python_exe)
        return None
    
    def _start_subprocess(self):
        """Start YAFS runner subprocess"""
        python312 = self._get_venv_py312_path()
        
        if not python312:
            print("[WARNING] Python 3.12 YAFS environment not found")
            self.process = None
            return
        
        # Create temporary directory for communication
        self.temp_dir = tempfile.mkdtemp()
        
        # Script that runs in Python 3.12 with YAFS
        runner_script = f"""
import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, r'{Path(__file__).parent.parent.parent}')

from fog_rl_medical.environment.fog_cluster import FogClusterEnv

# Initialize environment
config = {json.dumps(self.config)}
env = FogClusterEnv(config)

# Communication file paths
request_file = r'{self.temp_dir}/request.json'
response_file = r'{self.temp_dir}/response.json'

# Main loop
while True:
    if Path(request_file).exists():
        try:
            with open(request_file, 'r') as f:
                request = json.load(f)
            
            command = request.get('command')
            
            if command == 'reset':
                state = env.reset()
                response = {
                    'status': 'success',
                    'state': {{
                        'cpu_utilization': state.cpu_utilization.tolist(),
                        'memory_utilization': state.memory_utilization.tolist(),
                        'bandwidth_utilization': state.bandwidth_utilization.tolist(),
                        'queue_depths': state.queue_depths.tolist(),
                        'sla_violations': state.sla_violations.tolist(),
                        'priority_distribution': state.priority_distribution.tolist(),
                        'buffer_level': float(state.buffer_level),
                        'timestep': int(state.timestep),
                        'energy_consumption': float(state.energy_consumption)
                    }}
                }}
            
            elif command == 'step':
                actions = request.get('actions')
                alloc_actions = request.get('alloc_actions')
                priority_tasks = request.get('priority_tasks')
                
                state, reward, done, info = env.step(actions, alloc_actions, priority_tasks)
                
                response = {
                    'status': 'success',
                    'state': {{
                        'cpu_utilization': state.cpu_utilization.tolist(),
                        'memory_utilization': state.memory_utilization.tolist(),
                        'bandwidth_utilization': state.bandwidth_utilization.tolist(),
                        'queue_depths': state.queue_depths.tolist(),
                        'sla_violations': state.sla_violations.tolist(),
                        'priority_distribution': state.priority_distribution.tolist(),
                        'buffer_level': float(state.buffer_level),
                        'timestep': int(state.timestep),
                        'energy_consumption': float(state.energy_consumption)
                    }},
                    'reward': float(reward),
                    'done': bool(done),
                    'info': info
                }}
            
            elif command == 'close':
                response = {{'status': 'closing'}}
                with open(response_file, 'w') as f:
                    json.dump(response, f)
                break
            
            else:
                response = {{'status': 'error', 'message': f'Unknown command: {{command}}'}}
            
            # Write response
            with open(response_file, 'w') as f:
                json.dump(response, f)
            
            # Remove request file
            Path(request_file).unlink()
        
        except Exception as e:
            response = {{'status': 'error', 'message': str(e)}}
            with open(response_file, 'w') as f:
                json.dump(response, f)
"""
        
        try:
            self.process = subprocess.Popen(
                [python312, "-c", runner_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to start YAFS subprocess: {e}")
            self.process = None
    
    def _send_command(self, command_dict):
        """Send command to subprocess and get response"""
        if not self.process or not self.temp_dir:
            return None
        
        request_file = f"{self.temp_dir}/request.json"
        response_file = f"{self.temp_dir}/response.json"
        
        # Remove old response file
        if os.path.exists(response_file):
            os.remove(response_file)
        
        # Write request
        with open(request_file, 'w') as f:
            json.dump(command_dict, f)
        
        # Wait for response (with timeout)
        import time
        timeout = 10
        start = time.time()
        
        while not os.path.exists(response_file):
            if time.time() - start > timeout:
                return {'status': 'error', 'message': 'Subprocess timeout'}
            time.sleep(0.01)
        
        # Read response
        with open(response_file, 'r') as f:
            response = json.load(f)
        
        return response
    
    def reset(self):
        """Reset environment via subprocess"""
        response = self._send_command({'command': 'reset'})
        
        if response and response.get('status') == 'success':
            state_dict = response.get('state', {})
            return ClusterState(
                cpu_utilization=np.array(state_dict['cpu_utilization']),
                memory_utilization=np.array(state_dict['memory_utilization']),
                bandwidth_utilization=np.array(state_dict['bandwidth_utilization']),
                queue_depths=np.array(state_dict['queue_depths']),
                sla_violations=np.array(state_dict['sla_violations']),
                priority_distribution=np.array(state_dict['priority_distribution']),
                buffer_level=state_dict['buffer_level'],
                timestep=state_dict['timestep'],
                energy_consumption=state_dict.get('energy_consumption', 0.0)
            )
        return None
    
    def step(self, actions, alloc_actions=None, priority_tasks=None):
        """Execute step via subprocess"""
        response = self._send_command({
            'command': 'step',
            'actions': actions,
            'alloc_actions': alloc_actions,
            'priority_tasks': priority_tasks
        })
        
        if response and response.get('status') == 'success':
            state_dict = response.get('state', {})
            state = ClusterState(
                cpu_utilization=np.array(state_dict['cpu_utilization']),
                memory_utilization=np.array(state_dict['memory_utilization']),
                bandwidth_utilization=np.array(state_dict['bandwidth_utilization']),
                queue_depths=np.array(state_dict['queue_depths']),
                sla_violations=np.array(state_dict['sla_violations']),
                priority_distribution=np.array(state_dict['priority_distribution']),
                buffer_level=state_dict['buffer_level'],
                timestep=state_dict['timestep'],
                energy_consumption=state_dict.get('energy_consumption', 0.0)
            )
            reward = response.get('reward', 0.0)
            done = response.get('done', False)
            info = response.get('info', {})
            
            return state, reward, done, info
        
        return None, 0.0, False, {}
    
    def close(self):
        """Close subprocess"""
        if self.process:
            self._send_command({'command': 'close'})
            self.process.wait(timeout=5)
        
        # Cleanup temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
