# Fog RL Medical

A Priority-Aware Hierarchical RL system for Multimodal Fog Resource Allocation with LLM-Assisted Medical Reasoning.

## Project Structure
- `agents/`: Reinforcement Learning agents (high-level, low-level, hierarchical trainers)
- `cloud/`: Cloud capabilities, analytics, model stores
- `environment/`: Fog topological environment and SLA checking
- `fog/`: Fog cluster management, task dispatchers, load balancing
- `ingestion/`: Data streams, modal triage, queue managers
- `llm/`: Large Language Model integration for reasoning and caching
- `multimodal/`: Medical data processors (ECG, imaging, text, vitals)
- `priority/`: Priority engines and triage scoring
- `simulation/`: Scenario generation, benchmarks (YCSB)
- `training/`: Training routines, metrics and evaluation

## How to Run

1. **Install Dependencies**
   It's recommended to use an Anaconda environment or Python virtual environment.
   ```bash
   # Create a virtual environment (optional)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   
   # Install typical requirements
   pip install -r requirements.txt
   
   # Note: YAFS (Yet Another Fog Simulator) might need to be installed from GitHub if pip fails:
   pip install git+https://github.com/acsicuib/YAFS
   ```

2. **Run the Implementation**
   The main entry point to the system is `main.py`. This reads your `config.yaml` file, initializes the trainer, and executes the episodes.
   ```bash
   python main.py
   ```

3. **Check Output/Logs**
   Training metrics, episode stats, and execution logs will be handled by the modules in `training.metrics` and output to your console.
