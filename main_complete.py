#!/usr/bin/env python3
"""
Complete Fog RL Medical Project Runner
Executes everything: training, baselines, comparison, and generates all plots
Single entry point for the entire pipeline.

Usage: cd scratch && python main_complete.py
"""

import os
import sys
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import matplotlib.pyplot as plt
from fog_rl_medical.training.trainer import Trainer
from fog_rl_medical.training.baseline_trainers import (
    StandaloneDQNTrainer,
    SimpleHierarchicalTrainer,
    RandomAllocationTrainer
)

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  ► {title}")
    print("="*80 + "\n")

def load_config():
    """Load project configuration."""
    config = {}
    config_path = 'fog_rl_medical/config.yaml'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        print(f"✓ Config loaded from {config_path}")
    else:
        print(f"⚠ Config not found, using defaults")
    
    return config

def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs('results/comparison/', exist_ok=True)
    print("✓ Results directories ready")

def run_complete_pipeline():
    """Run the complete training and comparison pipeline."""
    
    print_section("FOG RL MEDICAL - COMPLETE PIPELINE")
    print("This will execute the full end-to-end workflow:")
    print("  1. Hierarchical DQN training")
    print("  2. Standalone DQN baseline")
    print("  3. Simple Hierarchical baseline")
    print("  4. Random Allocation baseline")
    print("  5. Generate comparison plots")
    print("  6. Generate improvement analysis")
    print("  7. Generate final metrics table\n")
    
    start_time = time.time()
    
    # Load config
    config = load_config()
    ensure_results_dir()
    
    # ========== PHASE 1: RUN ALL TRAINERS ==========
    print_section("PHASE 1: TRAINING ALL APPROACHES")
    
    results = {}
    
    # 1. Hierarchical DQN
    print("\n[1/4] Training Hierarchical DQN...")
    trainer_hdqn = Trainer(config)
    trainer_hdqn.run()
    results['Hierarchical DQN'] = {
        'sla': trainer_hdqn.metrics.history['sla_compliance'],
        'latency': trainer_hdqn.metrics.history['avg_latency'],
        'cloud_ratio': trainer_hdqn.metrics.history['cloud_offload_ratio'],
        'energy': trainer_hdqn.metrics.history['energy_consumption']
    }
    print("✓ Hierarchical DQN training complete")
    
    # 2. Standalone DQN
    print("\n[2/4] Training Standalone DQN...")
    trainer_sdqn = StandaloneDQNTrainer(config)
    trainer_sdqn.run()
    results['Standalone DQN'] = {
        'sla': trainer_sdqn.metrics.history['sla_compliance'],
        'latency': trainer_sdqn.metrics.history['avg_latency'],
        'cloud_ratio': trainer_sdqn.metrics.history['cloud_offload_ratio'],
        'energy': trainer_sdqn.metrics.history['energy_consumption']
    }
    print("✓ Standalone DQN training complete")
    
    # 3. Simple Hierarchical
    print("\n[3/4] Training Simple Hierarchical...")
    trainer_sh = SimpleHierarchicalTrainer(config)
    trainer_sh.run()
    results['Simple Hierarchical'] = {
        'sla': trainer_sh.metrics.history['sla_compliance'],
        'latency': trainer_sh.metrics.history['avg_latency'],
        'cloud_ratio': trainer_sh.metrics.history['cloud_offload_ratio'],
        'energy': trainer_sh.metrics.history['energy_consumption']
    }
    print("✓ Simple Hierarchical training complete")
    
    # 4. Random Allocation
    print("\n[4/4] Training Random Allocation...")
    trainer_ra = RandomAllocationTrainer(config)
    trainer_ra.run()
    results['Random Allocation'] = {
        'sla': trainer_ra.metrics.history['sla_compliance'],
        'latency': trainer_ra.metrics.history['avg_latency'],
        'cloud_ratio': trainer_ra.metrics.history['cloud_offload_ratio'],
        'energy': trainer_ra.metrics.history['energy_consumption']
    }
    print("✓ Random Allocation training complete")
    
    # ========== PHASE 2: GENERATE COMPARISON PLOTS ==========
    print_section("PHASE 2: GENERATING COMPARISON PLOTS")
    
    # Define colors and markers
    colors = {
        'Hierarchical DQN': '#1f77b4',
        'Standalone DQN': '#ff7f0e',
        'Simple Hierarchical': '#2ca02c',
        'Random Allocation': '#d62728'
    }
    markers = {
        'Hierarchical DQN': 'o',
        'Standalone DQN': 's',
        'Simple Hierarchical': '^',
        'Random Allocation': 'd'
    }
    
    # Plot 1: Multi-approach comparison
    print("\n[1/3] Generating multi-approach comparison plot...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hierarchical DQN vs Baselines - Performance Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    def downsample(data, step=20):
        return data[::step] if len(data) > step else data
    
    # SLA Compliance
    for approach, data in results.items():
        downsampled = downsample(data['sla'], step=20)
        episodes = np.arange(len(downsampled)) * 20
        axs[0, 0].plot(episodes, downsampled, label=approach, 
                      color=colors[approach], marker=markers[approach],
                      markersize=4, linewidth=2, alpha=0.8)
    axs[0, 0].set_title('SLA Compliance over Episodes', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Compliance Rate')
    axs[0, 0].legend(loc='best', fontsize=10)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Latency
    for approach, data in results.items():
        downsampled = downsample(data['latency'], step=20)
        episodes = np.arange(len(downsampled)) * 20
        axs[0, 1].plot(episodes, downsampled, label=approach,
                      color=colors[approach], marker=markers[approach],
                      markersize=4, linewidth=2, alpha=0.8)
    axs[0, 1].set_title('Average Latency over Episodes', fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Latency (ms)')
    axs[0, 1].legend(loc='best', fontsize=10)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Cloud Offload Ratio
    for approach, data in results.items():
        downsampled = downsample(data['cloud_ratio'], step=20)
        episodes = np.arange(len(downsampled)) * 20
        axs[1, 0].plot(episodes, downsampled, label=approach,
                      color=colors[approach], marker=markers[approach],
                      markersize=4, linewidth=2, alpha=0.8)
    axs[1, 0].set_title('Cloud Offload Ratio over Episodes', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Offload Ratio')
    axs[1, 0].legend(loc='best', fontsize=10)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Energy Consumption
    for approach, data in results.items():
        downsampled = downsample(data['energy'], step=20)
        episodes = np.arange(len(downsampled)) * 20
        axs[1, 1].plot(episodes, downsampled, label=approach,
                      color=colors[approach], marker=markers[approach],
                      markersize=4, linewidth=2, alpha=0.8)
    axs[1, 1].set_title('Energy Consumption over Episodes', fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Energy (kWh)')
    axs[1, 1].legend(loc='best', fontsize=10)
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison/multi_approach_comparison.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: multi_approach_comparison.png")
    
    # Plot 2: Improvement analysis
    print("\n[2/3] Generating improvement analysis plot...")
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Hierarchical DQN Performance Improvement over Baselines', 
                fontsize=14, fontweight='bold')
    
    approaches_to_compare = ['Standalone DQN', 'Simple Hierarchical', 'Random Allocation']
    hdqn_sla = np.mean(results['Hierarchical DQN']['sla'][-50:])
    hdqn_latency = np.mean(results['Hierarchical DQN']['latency'][-50:])
    hdqn_cloud = np.mean(results['Hierarchical DQN']['cloud_ratio'][-50:])
    hdqn_energy = np.mean(results['Hierarchical DQN']['energy'][-50:])
    
    metrics_data = {
        'SLA Compliance': (axs[0], 'SLA Improvement (%)', hdqn_sla, 'sla', True),
        'Latency (ms)': (axs[1], 'Latency Improvement (%)', hdqn_latency, 'latency', False),
        'Cloud Ratio': (axs[2], 'Cloud Ratio Improvement (%)', hdqn_cloud, 'cloud_ratio', False),
        'Energy (kWh)': (axs[3], 'Energy Improvement (%)', hdqn_energy, 'energy', False)
    }
    
    for metric_name, (ax, ylabel, hdqn_val, key, higher_is_better) in metrics_data.items():
        improvements = []
        labels = []
        
        for baseline in approaches_to_compare:
            baseline_val = np.mean(results[baseline][key][-50:])
            
            if higher_is_better:
                improvement = ((hdqn_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
            else:
                improvement = ((baseline_val - hdqn_val) / baseline_val * 100) if baseline_val != 0 else 0
            
            improvements.append(improvement)
            labels.append(baseline.replace(' ', '\n'))
        
        colors_list = ['#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(labels, improvements, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(min(improvements) - 10, max(improvements) + 15)
    
    plt.tight_layout()
    plt.savefig('results/comparison/improvement_analysis.png',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: improvement_analysis.png")
    
    # Plot 3: Final metrics table
    print("\n[3/3] Generating final metrics table...")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    approaches = list(results.keys())
    final_sla = [results[app]['sla'][-1] for app in approaches]
    final_latency = [results[app]['latency'][-1] for app in approaches]
    final_cloud_ratio = [results[app]['cloud_ratio'][-1] for app in approaches]
    final_energy = [results[app]['energy'][-1] for app in approaches]
    
    hdqn_sla = results['Hierarchical DQN']['sla'][-1]
    hdqn_latency = results['Hierarchical DQN']['latency'][-1]
    hdqn_cloud = results['Hierarchical DQN']['cloud_ratio'][-1]
    hdqn_energy = results['Hierarchical DQN']['energy'][-1]
    
    table_data = [['Metric', 'Hierarchical DQN', 'Standalone DQN', 'Simple Hierarchical', 'Random Allocation']]
    
    # SLA row
    sla_row = ['SLA Compliance']
    for app in approaches:
        val = results[app]['sla'][-1]
        if app == 'Hierarchical DQN':
            sla_row.append(f"{val:.4f}")
        else:
            improvement = ((val - hdqn_sla) / hdqn_sla * 100) if hdqn_sla != 0 else 0
            sla_row.append(f"{val:.4f}\n({improvement:+.1f}%)")
    table_data.append(sla_row)
    
    # Latency row
    latency_row = ['Avg Latency (ms)']
    for app in approaches:
        val = results[app]['latency'][-1]
        if app == 'Hierarchical DQN':
            latency_row.append(f"{val:.2f}")
        else:
            improvement = ((hdqn_latency - val) / val * 100) if val != 0 else 0
            latency_row.append(f"{val:.2f}\n({improvement:+.1f}%)")
    table_data.append(latency_row)
    
    # Cloud ratio row
    cloud_row = ['Cloud Offload Ratio']
    for app in approaches:
        val = results[app]['cloud_ratio'][-1]
        cloud_row.append(f"{val:.4f}")
    table_data.append(cloud_row)
    
    # Energy row
    energy_row = ['Energy Consumption (kWh)']
    for app in approaches:
        val = results[app]['energy'][-1]
        if app == 'Hierarchical DQN':
            energy_row.append(f"{val:.4f}")
        else:
            improvement = ((hdqn_energy - val) / val * 100) if val != 0 else 0
            energy_row.append(f"{val:.4f}\n({improvement:+.1f}%)")
    table_data.append(energy_row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style metric names
    for i in range(1, 5):
        table[(i, 0)].set_facecolor('#D9E1F2')
        table[(i, 0)].set_text_props(weight='bold')
    
    # Highlight H-DQN column
    for i in range(1, 5):
        table[(i, 1)].set_facecolor('#E2EFDA')
    
    plt.title('Final Performance Metrics Comparison\n(Values relative to Hierarchical DQN)', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('results/comparison/final_metrics_table.png',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: final_metrics_table.png")
    
    # ========== PHASE 3: COMPLETION ==========
    elapsed = time.time() - start_time
    
    print_section("✅ COMPLETE PIPELINE FINISHED")
    print(f"Total execution time: {elapsed:.1f} seconds\n")
    print("📊 Generated Files:")
    print("   • results/comparison/multi_approach_comparison.png")
    print("   • results/comparison/improvement_analysis.png")
    print("   • results/comparison/final_metrics_table.png")
    print("   • results/training_curves.png (H-DQN)")
    print("   • results/priority_distribution_heatmap.png\n")
    print("📈 All visualizations are ready for analysis!")
    print("="*80 + "\n")

if __name__ == '__main__':
    try:
        run_complete_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
