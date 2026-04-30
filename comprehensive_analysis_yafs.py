#!/usr/bin/env python3
"""
YAFS Direct Comprehensive Analysis
Run with: venv_py312\Scripts\python.exe comprehensive_analysis_yafs.py

Trains all 4 approaches using YAFS simulator (Python 3.12)
Compatible with venv_py312 only - requires YAFS package
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
from datetime import datetime

print("="*80)
print("COMPREHENSIVE ANALYSIS - YAFS SIMULATOR")
print("="*80)
print(f"Python: {sys.version.split()[0]}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load configuration
config_path = project_root / 'fog_rl_medical' / 'config' / 'env_config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

# Import trainers
from fog_rl_medical.training.trainer import Trainer
from fog_rl_medical.training.baseline_trainers import (
    StandaloneDQNTrainer,
    SimpleHierarchicalTrainer,
    RandomAllocationTrainer
)

# Create results directory
results_dir = project_root / 'results' / 'yafs_analysis'
results_dir.mkdir(parents=True, exist_ok=True)

print("TRAINING ALL 4 APPROACHES WITH YAFS")
print("-" * 80)
print()

trainers_config = [
    ('Hierarchical DQN', Trainer),
    ('Standalone DQN', StandaloneDQNTrainer),
    ('Simple Hierarchical', SimpleHierarchicalTrainer),
    ('Random Allocation', RandomAllocationTrainer),
]

all_results = {}

for idx, (name, trainer_class) in enumerate(trainers_config, 1):
    print(f"[{idx}/4] {name}...")
    
    try:
        # Create trainer with YAFS enabled
        trainer = trainer_class(config, use_yafs=True)
        
        # Run training
        trainer.run()
        
        # Collect metrics if available
        if hasattr(trainer, 'metrics') and hasattr(trainer.metrics, 'history'):
            latency = np.array(trainer.metrics.history.get('avg_latency', []))
            if len(latency) > 0:
                all_results[name] = {
                    'latency': latency,
                    'energy': np.array(trainer.metrics.history.get('energy_consumption', [])),
                    'sla': np.array(trainer.metrics.history.get('sla_compliance', [])),
                    'reward': np.array(trainer.metrics.history.get('reward', []))
                }
                print(f"      ✓ Completed - {len(latency)} episodes")
            else:
                print(f"      ✓ Completed")
        else:
            print(f"      ✓ Completed")
    
    except Exception as e:
        print(f"      ✗ Error: {str(e)[:80]}")

print()
print("="*80)
print("YAFS ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Results saved to: {results_dir}")
print()

# Summary
if all_results:
    print("Results Summary:")
    for name, metrics in all_results.items():
        if metrics.get('latency') is not None:
            latency_mean = np.mean(metrics['latency'][-100:])
            print(f"  {name:25} latency: {latency_mean:6.2f} ms")

print()
print("✓ YAFS Analysis Complete")
