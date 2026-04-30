# Analysis Scripts

Entry points for research analysis and visualization.

## Main Scripts

### `run_analysis.py`
Comprehensive analysis framework supporting both custom and YAFS environments.

**Usage:**
```bash
# 500 episodes with custom environment (default, fast)
python scripts/run_analysis.py

# Custom episode count
python scripts/run_analysis.py --episodes 200

# With YAFS simulator (Python 3.12 only)
venv_py312\Scripts\python.exe scripts/run_analysis.py --use-yafs
```

**Output:**
- `results/analysis/01_metrics_comparison.png` - 4-metric boxplot
- `results/analysis/02_convergence_trends.png` - Learning curves  
- `results/analysis/03_performance_heatmap.png` - Normalized metric comparison

**Features:**
- Trains all 4 approaches (H-DQN, Standalone DQN, Simple Hierarchical, Random)
- Generates statistical analysis (mean/std/range)
- Creates publication-ready visualizations
- Supports environment switching via --use-yafs flag

## Output Format

All results saved to `results/analysis/`:

```
results/analysis/
├── 01_metrics_comparison.png      # 4x1 boxplot: Latency, Energy, SLA, Reward
├── 02_convergence_trends.png      # Episode-by-episode latency trends
└── 03_performance_heatmap.png     # Normalized performance matrix
```

## Environment Requirements

### Python 3.11 (Custom Environment - Main)
```bash
venv_complete\Scripts\python.exe scripts/run_analysis.py
```

### Python 3.12 (YAFS - Optional)
```bash
venv_py312\Scripts\python.exe scripts/run_analysis.py --use-yafs
```

## Performance Output

Each run generates:
1. **Training Progress**: Episode-by-episode logs
2. **Statistical Summary**: Mean/Std/Range for last 100 episodes
3. **Research Summary**: Best performer by metric
4. **Visualizations**: 3 publication-quality PNG files (300 dpi)

## Interpreting Results

### Metrics Comparison
- Shows distribution of metrics across all 4 approaches
- Lower latency/energy better; Higher SLA% better
- Box width indicates variance across episodes

### Convergence Trends
- Shows learning progress over episodes
- Steeper downward slopes indicate faster convergence
- Plateau indicates convergence

### Performance Heatmap
- Color intensity: Green=Better, Red=Worse
- All metrics normalized to 0-9 scale
- Numbers show actual values (raw, not normalized)
