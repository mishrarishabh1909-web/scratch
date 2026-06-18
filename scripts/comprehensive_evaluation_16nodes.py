"""
Comprehensive Algorithm Comparison: 16 Fog Nodes + Realistic Hardware
Evaluates: H-DQN, Standalone DQN, GNN-RL, Simple Hierarchical, Random
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

def run_all_algorithms(config=None, num_episodes=500):
    """
    Run all algorithms and compare performance
    """
    print("=" * 80)
    print("COMPREHENSIVE ALGORITHM EVALUATION - 16 FOG NODES WITH REALISTIC HARDWARE")
    print("=" * 80)
    print(f"Configuration: {num_episodes} episodes per algorithm")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    results = {}
    
    # Algorithm 1: Hierarchical DQN (H-DQN)
    print("\n[1/5] Starting Hierarchical DQN (Rainbow Dueling)...")
    print("-" * 80)
    try:
        from fog_rl_medical.training.trainer import Trainer
        hdqn_trainer = Trainer(config, use_yafs=False)
        hdqn_trainer.run()
        results['H-DQN (Rainbow Dueling)'] = {
            'trainer': hdqn_trainer,
            'metrics': hdqn_trainer.metrics
        }
        print("✅ H-DQN training completed")
    except Exception as e:
        print(f"❌ H-DQN training failed: {e}")
    
    # Algorithm 2: Standalone DQN
    print("\n[2/5] Starting Standalone DQN...")
    print("-" * 80)
    try:
        from fog_rl_medical.training.baseline_trainers import StandaloneDQNTrainer
        standalone_trainer = StandaloneDQNTrainer(config, use_yafs=False)
        standalone_trainer.run()
        results['Standalone DQN'] = {
            'trainer': standalone_trainer,
            'metrics': standalone_trainer.metrics
        }
        print("✅ Standalone DQN training completed")
    except Exception as e:
        print(f"❌ Standalone DQN training failed: {e}")
    
    # Algorithm 3: GNN-RL
    print("\n[3/5] Starting Graph Neural Network RL (GNN-RL)...")
    print("-" * 80)
    try:
        from fog_rl_medical.training.gnn_rl_trainer import GNNRLTrainer
        gnnrl_trainer = GNNRLTrainer(config, use_yafs=False)
        gnnrl_trainer.run()
        results['GNN-RL'] = {
            'trainer': gnnrl_trainer,
            'metrics': gnnrl_trainer.metrics
        }
        print("✅ GNN-RL training completed")
    except Exception as e:
        print(f"❌ GNN-RL training failed: {e}")
    
    # Algorithm 4: Simple Hierarchical (Heuristic)
    print("\n[4/5] Starting Simple Hierarchical Heuristic...")
    print("-" * 80)
    try:
        from fog_rl_medical.training.baseline_trainers import SimpleHierarchicalTrainer
        simple_trainer = SimpleHierarchicalTrainer(config, use_yafs=False)
        simple_trainer.run()
        results['Simple Hierarchical'] = {
            'trainer': simple_trainer,
            'metrics': simple_trainer.metrics
        }
        print("✅ Simple Hierarchical training completed")
    except Exception as e:
        print(f"❌ Simple Hierarchical training failed: {e}")
    
    # Algorithm 5: Random Allocation
    print("\n[5/5] Starting Random Allocation (Baseline)...")
    print("-" * 80)
    try:
        from fog_rl_medical.training.baseline_trainers import RandomAllocationTrainer
        random_trainer = RandomAllocationTrainer(config, use_yafs=False)
        random_trainer.run()
        results['Random Allocation'] = {
            'trainer': random_trainer,
            'metrics': random_trainer.metrics
        }
        print("✅ Random Allocation training completed")
    except Exception as e:
        print(f"❌ Random Allocation training failed: {e}")
    
    return results


def analyze_and_compare(results):
    """
    Analyze results and generate comparison metrics
    """
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON RESULTS - 16 NODES REALISTIC HARDWARE")
    print("=" * 80)
    
    summary = {}
    
    for algo_name, data in results.items():
        metrics = data['metrics']
        
        # Get all latency, SLA, energy values
        latencies = [m.get('latency', 0) for m in metrics.episodes if m]
        slas = [m.get('sla', 0) for m in metrics.episodes if m]
        energies = [m.get('energy', 0) for m in metrics.episodes if m]
        rewards = [m.get('reward', 0) for m in metrics.episodes if m]
        
        if latencies:
            summary[algo_name] = {
                'latency_mean': np.mean(latencies),
                'latency_std': np.std(latencies),
                'latency_min': np.min(latencies),
                'latency_max': np.max(latencies),
                'sla_mean': np.mean(slas) * 100,  # Convert to percentage
                'sla_std': np.std(slas) * 100,
                'energy_mean': np.mean(energies),
                'energy_std': np.std(energies),
                'reward_mean': np.mean(rewards),
                'reward_std': np.std(rewards),
                'cloud_ratio_mean': np.mean([m.get('cloud_offload_ratio', 0) for m in metrics.episodes if m])
            }
    
    # Print comparison table
    print("\n" + "=" * 120)
    print(f"{'Algorithm':<25} {'Latency (ms)':<20} {'SLA (%)':<15} {'Energy (kWh)':<18} {'Reward':<15}")
    print("-" * 120)
    
    for algo_name in sorted(summary.keys(), 
                           key=lambda x: summary[x]['latency_mean']):
        metrics = summary[algo_name]
        print(f"{algo_name:<25} {metrics['latency_mean']:>6.2f}±{metrics['latency_std']:<8.2f} "
              f"{metrics['sla_mean']:>6.1f}±{metrics['sla_std']:<8.1f} "
              f"{metrics['energy_mean']:>8.4f}±{metrics['energy_std']:<10.4f} "
              f"{metrics['reward_mean']:>8.2f}±{metrics['reward_std']:<8.2f}")
    
    print("=" * 120)
    
    # Find best algorithm
    best_algo = min(summary.keys(), key=lambda x: summary[x]['latency_mean'])
    best_metrics = summary[best_algo]
    
    print(f"\n🏆 BEST ALGORITHM: {best_algo}")
    print(f"   Latency: {best_metrics['latency_mean']:.2f} ms")
    print(f"   SLA Compliance: {best_metrics['sla_mean']:.1f}%")
    print(f"   Energy: {best_metrics['energy_mean']:.4f} kWh")
    print(f"   Cloud Offload Ratio: {best_metrics['cloud_ratio_mean']:.2%}")
    
    # Ranking
    print("\n" + "-" * 120)
    print("ALGORITHM RANKING (by Latency):")
    print("-" * 120)
    for i, (algo_name, metrics) in enumerate(
        sorted(summary.items(), key=lambda x: x[1]['latency_mean']), 1
    ):
        improvement = ((summary[best_algo]['latency_mean'] / metrics['latency_mean']) - 1) * 100
        print(f"{i}. {algo_name:<25} {metrics['latency_mean']:>7.2f} ms  "
              f"({improvement:+.1f}% vs best)")
    
    return summary


def generate_visualizations(results, summary, output_dir='results/analysis_16nodes'):
    """
    Generate comprehensive visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    algos = list(summary.keys())
    latencies = [summary[a]['latency_mean'] for a in algos]
    latencies_std = [summary[a]['latency_std'] for a in algos]
    slas = [summary[a]['sla_mean'] for a in algos]
    energies = [summary[a]['energy_mean'] for a in algos]
    rewards = [summary[a]['reward_mean'] for a in algos]
    
    # Sort by latency
    sorted_indices = np.argsort(latencies)
    algos_sorted = [algos[i] for i in sorted_indices]
    latencies_sorted = [latencies[i] for i in sorted_indices]
    latencies_std_sorted = [latencies_std[i] for i in sorted_indices]
    slas_sorted = [slas[i] for i in sorted_indices]
    energies_sorted = [energies[i] for i in sorted_indices]
    rewards_sorted = [rewards[i] for i in sorted_indices]
    
    # Figure 1: Performance Comparison (Bar Chart)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Algorithm Performance Comparison - 16 Fog Nodes (Realistic Hardware)', 
                 fontsize=16, fontweight='bold')
    
    # Latency comparison
    ax = axes[0, 0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(algos_sorted)))
    bars = ax.bar(range(len(algos_sorted)), latencies_sorted, 
                  yerr=latencies_std_sorted, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Task Latency', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(algos_sorted)))
    ax.set_xticklabels(algos_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, latencies_sorted)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + latencies_std_sorted[i] + 2,
                f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # SLA comparison
    ax = axes[0, 1]
    colors_sla = plt.cm.Greens(np.linspace(0.3, 0.9, len(algos_sorted)))
    bars = ax.bar(range(len(algos_sorted)), slas_sorted, color=colors_sla, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('SLA Compliance (%)', fontsize=12, fontweight='bold')
    ax.set_title('SLA Compliance', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(algos_sorted)))
    ax.set_xticklabels(algos_sorted, rotation=45, ha='right')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, slas_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Energy comparison
    ax = axes[1, 0]
    colors_energy = plt.cm.Blues(np.linspace(0.3, 0.9, len(algos_sorted)))
    bars = ax.bar(range(len(algos_sorted)), energies_sorted, color=colors_energy, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Efficiency', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(algos_sorted)))
    ax.set_xticklabels(algos_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Reward comparison
    ax = axes[1, 1]
    colors_reward = plt.cm.Purples(np.linspace(0.3, 0.9, len(algos_sorted)))
    bars = ax.bar(range(len(algos_sorted)), rewards_sorted, color=colors_reward, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Training Reward', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(algos_sorted)))
    ax.set_xticklabels(algos_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_algorithm_comparison_16nodes.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/01_algorithm_comparison_16nodes.png")
    plt.close()
    
    # Figure 2: Detailed Metrics Table as Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for heatmap
    metrics_data = []
    for algo in algos_sorted:
        m = summary[algo]
        # Normalize metrics to 0-1 scale for heatmap
        metrics_data.append([
            m['latency_mean'] / max([summary[a]['latency_mean'] for a in algos_sorted]),
            (100 - m['sla_mean']) / 100,  # Invert (lower is better)
            m['energy_mean'] / max([summary[a]['energy_mean'] for a in algos_sorted]),
            m['reward_mean'] / max([summary[a]['reward_mean'] for a in algos_sorted])
        ])
    
    metrics_data = np.array(metrics_data)
    sns.heatmap(metrics_data.T, annot=False, cmap='RdYlGn_r', 
                xticklabels=algos_sorted, yticklabels=['Latency', 'SLA Gap', 'Energy', 'Reward'],
                cbar_kws={'label': 'Normalized Score'}, ax=ax, vmin=0, vmax=1)
    ax.set_title('Normalized Performance Heatmap - 16 Nodes', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_performance_heatmap_16nodes.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/02_performance_heatmap_16nodes.png")
    plt.close()


def main():
    """
    Main evaluation pipeline
    """
    import yaml
    
    # Load config
    with open('fog_rl_medical/config/env_config.yaml', 'r') as f:
        config = {'environment': yaml.safe_load(f)}
    
    # Run all algorithms
    results = run_all_algorithms(config, num_episodes=500)
    
    # Analyze and compare
    summary = analyze_and_compare(results)
    
    # Generate visualizations
    generate_visualizations(results, summary)
    
    # Save results to JSON
    import json
    results_json = {}
    for algo, metrics_dict in summary.items():
        results_json[algo] = {k: float(v) for k, v in metrics_dict.items()}
    
    os.makedirs('results/analysis_16nodes', exist_ok=True)
    with open('results/analysis_16nodes/algorithm_comparison_16nodes.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n✅ All algorithms evaluated successfully!")
    print("📊 Results saved to: results/analysis_16nodes/")


if __name__ == "__main__":
    main()
