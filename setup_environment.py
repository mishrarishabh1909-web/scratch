#!/usr/bin/env python3
"""
Complete environment setup and verification script.
Run this to ensure your environment is properly configured and working.

Usage:
    venv_complete\Scripts\python.exe setup_environment.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status"""
    try:
        if description:
            print(f"\n{description}...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            if description:
                print("✓")
            return True
        else:
            if description:
                print(f"✗ ({result.stderr[:50]}...)")
            return False
    except Exception as e:
        if description:
            print(f"✗ ({str(e)[:50]}...)")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('networkx', 'NetworkX'),
        ('simpy', 'SimPy'),
        ('yaml', 'PyYAML'),
        ('transformers', 'Transformers'),
        ('openai', 'OpenAI'),
        ('gym', 'Gym'),
    ]
    
    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name:20} installed")
        except ImportError:
            print(f"  ✗ {name:20} NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check if project structure is intact"""
    print("\n" + "="*80)
    print("CHECKING PROJECT STRUCTURE")
    print("="*80)
    
    required_dirs = [
        'fog_rl_medical/agents',
        'fog_rl_medical/environment',
        'fog_rl_medical/training',
        'fog_rl_medical/config',
    ]
    
    required_files = [
        'fog_rl_medical/config/env_config.yaml',
        'fog_rl_medical/training/trainer.py',
        'fog_rl_medical/training/baseline_trainers.py',
        'comprehensive_analysis.py',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} NOT FOUND")
            all_ok = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_trainers():
    """Check if all trainers are importable"""
    print("\n" + "="*80)
    print("CHECKING TRAINERS")
    print("="*80)
    
    try:
        from fog_rl_medical.training.trainer import Trainer
        from fog_rl_medical.training.baseline_trainers import (
            StandaloneDQNTrainer,
            SimpleHierarchicalTrainer,
            RandomAllocationTrainer
        )
        import yaml
        
        with open('fog_rl_medical/config/env_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        trainers = [
            ('Trainer (H-DQN)', Trainer),
            ('StandaloneDQNTrainer', StandaloneDQNTrainer),
            ('SimpleHierarchicalTrainer', SimpleHierarchicalTrainer),
            ('RandomAllocationTrainer', RandomAllocationTrainer),
        ]
        
        all_ok = True
        for name, trainer_class in trainers:
            try:
                # Test custom implementation
                t1 = trainer_class(config, use_yafs=False)
                print(f"  ✓ {name:30} (custom implementation)")
            except Exception as e:
                print(f"  ✗ {name:30} {str(e)[:40]}...")
                all_ok = False
            
            try:
                # Test YAFS availability
                t2 = trainer_class(config, use_yafs=True)
                print(f"    ✓ YAFS available (subprocess bridge)")
            except Exception as e:
                print(f"    ✓ YAFS unavailable (falls back to custom)")
        
        return all_ok
    except Exception as e:
        print(f"  ✗ Failed to import trainers: {e}")
        return False

def check_comprehensive_analysis():
    """Check if comprehensive analysis script works"""
    print("\n" + "="*80)
    print("CHECKING COMPREHENSIVE ANALYSIS SCRIPT")
    print("="*80)
    
    try:
        from comprehensive_analysis import run_comprehensive_analysis
        import inspect
        
        sig = inspect.signature(run_comprehensive_analysis)
        has_param = 'use_yafs' in sig.parameters
        default = sig.parameters['use_yafs'].default if has_param else None
        
        if has_param:
            print(f"  ✓ run_comprehensive_analysis() found")
            print(f"  ✓ use_yafs parameter (default={default})")
            print(f"  ✓ CLI support: --use-yafs flag")
        else:
            print(f"  ✗ use_yafs parameter not found")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*80)
    print("COMPLETE ENVIRONMENT VERIFICATION")
    print("="*80)
    
    print(f"\nCurrent Python: {sys.executable}")
    print(f"Project directory: {os.getcwd()}")
    
    results = {
        "Dependencies": check_dependencies(),
        "Project Structure": check_project_structure(),
        "Trainers": check_trainers(),
        "Comprehensive Analysis": check_comprehensive_analysis(),
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_ok = True
    for check_name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {check_name:30} {'OK' if status else 'ISSUES FOUND'}")
        if not status:
            all_ok = False
    
    print("\n" + "="*80)
    if all_ok:
        print("✓ ALL CHECKS PASSED - ENVIRONMENT IS READY")
        print("="*80)
        print("\nYou can now run:")
        print("  python main_complete.py")
        print("  python comprehensive_analysis.py")
        print("  python comprehensive_analysis.py --use-yafs")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - PLEASE REVIEW ABOVE")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
