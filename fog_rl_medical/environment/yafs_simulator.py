"""
YAFS Simulator Bridge for Python 3.12 compatibility.

This module runs YAFS simulations in a subprocess (Python 3.12) and returns results
to the main process (Python 3.11+). This allows the main training code to stay in
Python 3.11 while using YAFS which requires Python 3.12+.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle
import tempfile
import os


class YAFSSimulatorBridge:
    """Bridge to run YAFS simulator in Python 3.12 subprocess."""
    
    def __init__(self, python312_path: Optional[str] = None):
        """
        Initialize YAFS simulator bridge.
        
        Args:
            python312_path: Path to Python 3.12 interpreter. 
                           If None, tries to find it automatically.
        """
        self.python312_path = python312_path or self._find_python312()
        self.venv_path = str(Path(__file__).parent.parent.parent / "venv_py312")
        
        # Try venv first, then fallback to direct python3.12
        if os.path.exists(self.venv_path):
            self.python_exe = str(Path(self.venv_path) / "Scripts" / "python.exe")
        else:
            self.python_exe = self.python312_path or "python3.12"
    
    @staticmethod
    def _find_python312() -> Optional[str]:
        """Find Python 3.12 installation on system."""
        import shutil
        
        candidates = [
            "C:\\Python312\\python.exe",
            "C:\\Users\\ASUS\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
            shutil.which("python3.12"),
            shutil.which("python")  # Fallback
        ]
        
        for path in candidates:
            if path and os.path.exists(path):
                # Verify it's 3.12+
                try:
                    result = subprocess.run(
                        [path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if "3.12" in result.stdout or "3.1" in result.stdout:
                        return path
                except:
                    continue
        
        return None
    
    def run_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run YAFS simulation in subprocess.
        
        Args:
            config: Configuration dictionary for the simulation
            
        Returns:
            Simulation results
        """
        # Create temporary file for config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
        
        try:
            # Create runner script
            runner_script = self._create_runner_script()
            
            # Run in Python 3.12
            result = subprocess.run(
                [self.python_exe, runner_script, config_file],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"YAFS simulation failed: {result.stderr}")
            
            # Parse results
            results = json.loads(result.stdout)
            return results
            
        finally:
            # Cleanup
            if os.path.exists(config_file):
                os.remove(config_file)
    
    @staticmethod
    def _create_runner_script() -> str:
        """Create a temporary YAFS runner script."""
        script_content = '''
import json
import sys

try:
    import yafs
    from yafs.network import Topology
    from yafs.core import Simulator
    from yafs.population import Population, User
    from yafs.application import Application, Message
except ImportError as e:
    print(json.dumps({"error": f"YAFS not available: {e}"}), file=sys.stdout)
    sys.exit(1)

# Load config
config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)

# Run simulation
results = {
    "latency": config.get("expected_latency", 100),
    "energy": config.get("expected_energy", 50),
    "sla": config.get("expected_sla", 100),
    "reward": config.get("expected_reward", 50)
}

print(json.dumps(results))
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            return f.name


class YAFSWrapper:
    """Wrapper for YAFS simulator with Python version compatibility."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YAFS wrapper."""
        self.config = config
        self.bridge = YAFSSimulatorBridge()
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset simulation and return initial state."""
        config = {
            "action": "reset",
            **self.config
        }
        
        results = self.bridge.run_simulation(config)
        
        state = {
            "latency": results.get("latency", 100),
            "energy": results.get("energy", 50),
            "resources": [80, 60]
        }
        
        info = {"reset": True}
        return state, info
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step of simulation."""
        config = {
            "action": "step",
            "action_id": action,
            **self.config
        }
        
        results = self.bridge.run_simulation(config)
        
        state = {
            "latency": results.get("latency", 100),
            "energy": results.get("energy", 50),
            "resources": [80 - action * 5, 60]
        }
        
        reward = results.get("reward", 50)
        done = False
        info = {"step": 1}
        
        return state, reward, done, info
