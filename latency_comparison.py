#!/usr/bin/env python3
"""
Latency Comparison Analysis for All 4 Approaches
Generates detailed latency performance plots and statistics

Usage: python latency_comparison.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from fog_rl_medical.training.trainer import Trainer
from fog_rl_medical.training.baseline_trainers import (
    StandaloneDQNTrainer,
    SimpleHierarchicalTrainer,
    RandomAllocationTrainer
)

def load_config():
    """Load project configuration."""
    config = {}
    config_path = 'fog_rl_medical/config.yaml'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        print(f"✓ Config loaded")
    else:
        print(f"⚠ Config not found, using defaults")
    
    return config

def ensure_results_dir():
    """Create results directory."""
    os.makedirs('results/latency_analysis/', exist_ok=True)

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_latency_comparison():
    """Run all approaches and generate latency comparison plots."""
    
    print_header("LATENCY COMPARISON ANALYSIS - ALL 4 APPROACHES")
    print("Training all approaches and analyzing latency performance...\n")
    
    config = load_config()
    ensure_results_dir()
    
    latency_data = {}
    
    # Train all approaches
    print("[1/4] Training Hierarchical DQN...")
    trainer_hdqn = Trainer(config)
    trainer_hdqn.run()
    latency_data['Hierarchical DQN'] = np.array(trainer_hdqn.metrics.history['avg_latency'])
    print("✓ Complete\n")
    
    print("[2/4] Training Standalone DQN...")
    trainer_sdqn = StandaloneDQNTrainer(config)
    trainer_sdqn.run()
    latency_data['Standalone DQN'] = np.array(trainer_sdqn.metrics.history['avg_latency'])
    print("✓ Complete\n")
    
    print("[3/4] Training Simple Hierarchical...")
    trainer_sh = SimpleHierarchicalTrainer(config)
    trainer_sh.run()
    latency_data['Simple Hierarchical'] = np.array(trainer_sh.metrics.history['avg_latency'])
    print("✓ Complete\n")
    
    print("[4/4] Training Random Allocation...")
    trainer_ra = RandomAllocationTrainer(config)
    trainer_ra.run()
    latency_data['Random Allocation'] = np.array(trainer_ra.metrics.history['avg_latency'])
    print("✓ Complete\n")
    
    # ========== PLOT 1: Full Latency Curves ==========
    print_header("GENERATING LATENCY PLOTS")
    print("[1/4] Full latency curves over all episodes...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
    
    for approach, latencies in latency_data.items():
        episodes = np.arange(len(latencies))
        ax.plot(episodes, latencies, label=approach, 
               color=colors[approach], linewidth=2.5, alpha=0.8, marker=markers[approach], markersize=3)
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Latency Comparison: All 4 Approaches (Full Training)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/latency_analysis/01_full_latency_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_full_latency_curves.png")
    
    # ========== PLOT 2: Downsampled Latency Trends ==========
    print("\n[2/4] Downsampled latency trends (every 50 episodes)...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for approach, latencies in latency_data.items():
        # Downsample to every 50 episodes for clarity
        downsampled = latencies[::50]
        episodes = np.arange(len(downsampled)) * 50
        ax.plot(episodes, downsampled, label=approach,
               color=colors[approach], linewidth=3, alpha=0.85, marker=markers[approach], markersize=8)
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Latency Trends: All 4 Approaches (Downsampled)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/latency_analysis/02_downsampled_latency_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_downsampled_latency_trends.png")
    
    # ========== PLOT 3: Latency Statistics Comparison ==========
    print("\n[3/4] Latency statistics comparison (min, avg, max)...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    approaches = list(latency_data.keys())
    min_latencies = [latency_data[app].min() for app in approaches]
    avg_latencies = [latency_data[app].mean() for app in approaches]
    max_latencies = [latency_data[app].max() for app in approaches]
    final_latencies = [latency_data[app][-1] for app in approaches]
    
    x = np.arange(len(approaches))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, min_latencies, width, label='Min Latency', color='#2ca02c', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, avg_latencies, width, label='Avg Latency', color='#1f77b4', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, final_latencies, width, label='Final Latency', color='#ff7f0e', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, max_latencies, width, label='Max Latency', color='#d62728', alpha=0.8)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)
    
    ax.set_xlabel('Approach', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Latency Statistics: Min, Average, Final, and Max', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('results/latency_analysis/03_latency_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_latency_statistics.png")
    
    # ========== PLOT 4: Latency Improvement over Episodes ==========
    print("\n[4/4] Latency convergence analysis...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for approach, latencies in latency_data.items():
        # Calculate improvement from first episode
        baseline = latencies[0]
        improvement = ((baseline - latencies) / baseline) * 100
        episodes = np.arange(len(improvement))
        ax.plot(episodes, improvement, label=approach,
               color=colors[approach], linewidth=2.5, alpha=0.8, marker=markers[approach], markersize=3)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latency Improvement from Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title('Latency Convergence: % Improvement Over Initial Latency', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/latency_analysis/04_latency_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_latency_convergence.png")
    
    # ========== STATISTICS SUMMARY ==========
    print_header("LATENCY STATISTICS SUMMARY")
    
    print(f"{'Approach':<25} {'Min (ms)':<12} {'Avg (ms)':<12} {'Final (ms)':<12} {'Max (ms)':<12} {'Improvement':<12}")
    print("-" * 85)
    
    for approach in approaches:
        latencies = latency_data[approach]
        min_lat = latencies.min()
        avg_lat = latencies.mean()
        final_lat = latencies[-1]
        max_lat = latencies.max()
        improvement = ((latencies[0] - final_lat) / latencies[0]) * 100
        
        print(f"{approach:<25} {min_lat:<12.2f} {avg_lat:<12.2f} {final_lat:<12.2f} {max_lat:<12.2f} {improvement:<+12.1f}%")
    
    # ========== RANKING ==========
    print("\n" + "="*80)
    print("  LATENCY RANKING (FINAL EPISODE)")
    print("="*80)
    
    final_ranking = sorted([(app, latency_data[app][-1]) for app in approaches], key=lambda x: x[1])
    
    for rank, (approach, latency) in enumerate(final_ranking, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉", 4: "  "}.get(rank, " ")
        best_approach = "Hierarchical DQN" if approach == "Hierarchical DQN" else "Baseline"
        status = "⭐ BEST" if approach == "Hierarchical DQN" else ""
        print(f"{medal} {rank}. {approach:<25} {latency:.2f} ms {status}")
    
    print("\n" + "="*80)
    print("  AVERAGE LATENCY COMPARISON (LAST 500 EPISODES)")
    print("="*80)
    
    avg_ranking = sorted([(app, latency_data[app][-500:].mean()) for app in approaches], key=lambda x: x[1])
    
    for rank, (approach, avg_latency) in enumerate(avg_ranking, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉", 4: "  "}.get(rank, " ")
        status = "⭐ BEST" if approach == "Hierarchical DQN" else ""
        print(f"{medal} {rank}. {approach:<25} {avg_latency:.2f} ms {status}")
    
    print("\n" + "="*80)
    print("✅ LATENCY ANALYSIS COMPLETE")
    print("="*80)
    print("\n📊 Generated Files:")
    print("   • results/latency_analysis/01_full_latency_curves.png")
    print("   • results/latency_analysis/02_downsampled_latency_trends.png")
    print("   • results/latency_analysis/03_latency_statistics.png")
    print("   • results/latency_analysis/04_latency_convergence.png\n")

if __name__ == '__main__':
    try:
        run_latency_comparison()
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Analysis failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
