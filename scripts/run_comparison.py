#!/usr/bin/env python3
"""
Run comparison between Hierarchical DQN and baseline approaches.
Usage: python scripts/run_comparison.py
"""

import sys
import os

# Add project root to path (go up one level from scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml
from fog_rl_medical.training.comparison_runner import ComparisonRunner

if __name__ == '__main__':
    # Load configuration
    config = {}
    config_path = 'fog_rl_medical/config.yaml'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        print(f"✓ Loaded config from {config_path}")
    else:
        print(f"⚠ Config file not found at {config_path}, using defaults")
    
    # Run comparison
    print("\nInitializing comparison runner...")
    runner = ComparisonRunner(config)
    
    print("\nStarting comprehensive comparison of all approaches...\n")
    runner.run_comparison()
    
    print("\n✓ All comparisons complete! Check results/ for visualizations.")
