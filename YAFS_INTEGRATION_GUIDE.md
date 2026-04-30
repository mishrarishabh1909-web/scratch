# YAFS Integration Guide

## Overview

Your fog computing project now supports **dual-mode operation**:
1. **Custom Fog Cluster Environment** (default, existing)
2. **YAFS Simulator** (optional, industry-standard)

Both modes use the same code interface but can be switched on/off based on simulation needs.

## Quick Start

### Using Default Custom Implementation (No Changes Needed)

```python
from fog_rl_medical.training.trainer import Trainer

config = {...}
trainer = Trainer(config)  # Uses FogClusterEnv automatically
trainer.run()
```

### Using YAFS Simulator (When Available)

```python
from fog_rl_medical.training.trainer import Trainer

config = {...}
trainer = Trainer(config, use_yafs=True)  # Uses YAFSFogEnvironment
trainer.run()
```

## Supported Trainers

All trainers support the same `use_yafs` parameter:

```python
# Hierarchical DQN
trainer = Trainer(config, use_yafs=use_yafs)

# Standalone DQN
trainer = StandaloneDQNTrainer(config, use_yafs=use_yafs)

# Simple Hierarchical Baseline
trainer = SimpleHierarchicalTrainer(config, use_yafs=use_yafs)

# Random Allocation Baseline
trainer = RandomAllocationTrainer(config, use_yafs=use_yafs)
```

## Comprehensive Analysis

### Run with Custom Implementation (Default)
```bash
python comprehensive_analysis.py
```

### Run with YAFS Simulator
```bash
python comprehensive_analysis.py --use-yafs
```

## Installation (Optional)

To install YAFS simulator:

```bash
pip install git+https://github.com/acsicuib/YAFS.git
```

**Note**: YAFS installation is optional. The code gracefully falls back to the custom implementation if YAFS is not installed.

## Architecture Compatibility

Both environments maintain identical interfaces:

- **Return Type**: `ClusterState` dataclass
- **Method Signatures**: Same reset() and step() methods
- **State Format**: Compatible tensor shapes for DQN networks
- **Metrics**: Same metric collection interface

## Zero Breaking Changes

✅ **All existing code works unchanged** - no updates needed to your current implementation.

- Default behavior: Uses `FogClusterEnv` (your existing custom implementation)
- Opt-in behavior: Set `use_yafs=True` only where needed
- Graceful fallback: If YAFS not installed, automatically uses `FogClusterEnv`

## Implementation Details

### Factory Pattern

The magic happens in `fog_rl_medical/environment/yafs_wrapper.py`:

```python
class EnvironmentFactory:
    @staticmethod
    def create_environment(config, use_yafs=False):
        if use_yafs:
            try:
                return YAFSFogEnvironment(config)
            except ImportError:
                # Fallback if YAFS not installed
                return FogClusterEnv(config)
        else:
            return FogClusterEnv(config)  # Default
```

### Configuration

YAFS settings stored in `fog_rl_medical/config/yafs_config.yaml`:
- Simulator parameters
- Network topology configuration
- Resource specifications
- Task distribution settings

## Research Use Cases

### Scenario 1: Development & Prototyping
**Use Custom Implementation (Default)**
```python
trainer = Trainer(config)  # Fast iteration
```
- Faster training cycles
- Your existing tested implementation
- Perfect for algorithm development

### Scenario 2: Industry-Standard Validation
**Use YAFS Simulator**
```python
trainer = Trainer(config, use_yafs=True)  # Publication validation
```
- Compare against YAFS-based research papers
- Industry-standard simulator validation
- More credible for peer review

### Scenario 3: Comprehensive Comparison
**Use Both Modes**
```bash
# Run with custom implementation
python comprehensive_analysis.py

# Run with YAFS to compare
python comprehensive_analysis.py --use-yafs
```

## Troubleshooting

### Warning: "YAFS not installed"
This is normal if you haven't installed YAFS yet. The code automatically falls back to `FogClusterEnv`.

To fix:
```bash
pip install git+https://github.com/acsicuib/YAFS.git
```

### Different Results Between Modes
This is expected - YAFS and custom implementation have different:
- Timeout handling
- Network simulation fidelity
- Resource contention models

Both are valid for research - choose based on your needs.

## Summary

| Aspect | Custom | YAFS |
|--------|--------|------|
| **Speed** | ⚡ Faster | 🔧 Slower |
| **Install** | ✅ Built-in | 📦 Optional pip |
| **Industry** | 📊 Custom | 🏢 Standard |
| **Default** | ✅ Yes | ❌ No |
| **Change Code** | ❌ No | ❌ No |
| **Switching** | Via `use_yafs` parameter | Via `use_yafs` parameter |

Both modes deliver to your DQN agents through the same interface.
