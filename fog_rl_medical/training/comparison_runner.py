"""
Comparison runner for Hierarchical DQN vs baselines.
Runs all approaches and compares the results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from fog_rl_medical.training.trainer import Trainer
from fog_rl_medical.training.baseline_trainers import (
    StandaloneDQNTrainer,
    SimpleHierarchicalTrainer,
    RandomAllocationTrainer
)


class ComparisonRunner:
    """Run all approaches and compare metrics."""
    
    def __init__(self, config=None, num_runs=1):
        self.config = config or {}
        self.num_runs = num_runs
        self.results = {
            'Hierarchical DQN': {'sla': [], 'latency': [], 'cloud_ratio': [], 'energy': []},
            'Standalone DQN': {'sla': [], 'latency': [], 'cloud_ratio': [], 'energy': []},
            'Simple Hierarchical': {'sla': [], 'latency': [], 'cloud_ratio': [], 'energy': []},
            'Random Allocation': {'sla': [], 'latency': [], 'cloud_ratio': [], 'energy': []}
        }
        self.comparison_dir = 'results/comparison/'
        os.makedirs(self.comparison_dir, exist_ok=True)

    def run_all(self):
        """Run all approaches."""
        print("\n" + "="*80)
        print("STARTING MULTI-APPROACH COMPARISON")
        print("="*80 + "\n")
        
        approaches = [
            ('Hierarchical DQN', Trainer),
            ('Standalone DQN', StandaloneDQNTrainer),
            ('Simple Hierarchical', SimpleHierarchicalTrainer),
            ('Random Allocation', RandomAllocationTrainer)
        ]
        
        for approach_name, trainer_class in approaches:
            print(f"\n{'='*80}")
            print(f"Running: {approach_name}")
            print(f"{'='*80}\n")
            
            trainer = trainer_class(self.config)
            trainer.run()
            
            # Collect results
            self.results[approach_name]['sla'] = trainer.metrics.history['sla_compliance']
            self.results[approach_name]['latency'] = trainer.metrics.history['avg_latency']
            self.results[approach_name]['cloud_ratio'] = trainer.metrics.history['cloud_offload_ratio']
            self.results[approach_name]['energy'] = trainer.metrics.history['energy_consumption']
            
            # Plot individual approach metrics
            trainer.metrics.plot_metrics()
            
            print(f"\n{approach_name} completed!")
            print(f"  Final SLA Compliance: {trainer.metrics.history['sla_compliance'][-1]:.4f}")
            print(f"  Final Avg Latency: {trainer.metrics.history['avg_latency'][-1]:.2f} ms")
            print(f"  Final Cloud Ratio: {trainer.metrics.history['cloud_offload_ratio'][-1]:.4f}")
            print(f"  Final Energy: {trainer.metrics.history['energy_consumption'][-1]:.4f}")

    def generate_comparison_plots(self):
        """Generate comparison plots across all approaches."""
        print("\n" + "="*80)
        print("GENERATING COMPARISON PLOTS")
        print("="*80 + "\n")
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hierarchical DQN vs Baselines - Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
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
        
        # Plot 1: SLA Compliance
        for approach, data in self.results.items():
            downsampled = self._downsample(data['sla'], step=20)
            episodes = np.arange(len(downsampled)) * 20
            axs[0, 0].plot(episodes, downsampled, label=approach, 
                          color=colors[approach], marker=markers[approach],
                          markersize=4, linewidth=2, alpha=0.8)
        
        axs[0, 0].set_title('SLA Compliance over Episodes', fontsize=12, fontweight='bold')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Compliance Rate')
        axs[0, 0].legend(loc='best', fontsize=10)
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Latency
        for approach, data in self.results.items():
            downsampled = self._downsample(data['latency'], step=20)
            episodes = np.arange(len(downsampled)) * 20
            axs[0, 1].plot(episodes, downsampled, label=approach,
                          color=colors[approach], marker=markers[approach],
                          markersize=4, linewidth=2, alpha=0.8)
        
        axs[0, 1].set_title('Average Latency over Episodes', fontsize=12, fontweight='bold')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Latency (ms)')
        axs[0, 1].legend(loc='best', fontsize=10)
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cloud Offload Ratio
        for approach, data in self.results.items():
            downsampled = self._downsample(data['cloud_ratio'], step=20)
            episodes = np.arange(len(downsampled)) * 20
            axs[1, 0].plot(episodes, downsampled, label=approach,
                          color=colors[approach], marker=markers[approach],
                          markersize=4, linewidth=2, alpha=0.8)
        
        axs[1, 0].set_title('Cloud Offload Ratio over Episodes', fontsize=12, fontweight='bold')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Offload Ratio')
        axs[1, 0].legend(loc='best', fontsize=10)
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Energy Consumption
        for approach, data in self.results.items():
            downsampled = self._downsample(data['energy'], step=20)
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
        plt.savefig(os.path.join(self.comparison_dir, 'multi_approach_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: multi_approach_comparison.png")

    def generate_final_metrics_table(self):
        """Generate a comparison table of final metrics."""
        print("\n" + "="*80)
        print("FINAL METRICS COMPARISON")
        print("="*80 + "\n")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        approaches = list(self.results.keys())
        final_sla = [self.results[app]['sla'][-1] for app in approaches]
        final_latency = [self.results[app]['latency'][-1] for app in approaches]
        final_cloud_ratio = [self.results[app]['cloud_ratio'][-1] for app in approaches]
        final_energy = [self.results[app]['energy'][-1] for app in approaches]
        
        # Calculate improvements over baselines
        hdqn_sla = self.results['Hierarchical DQN']['sla'][-1]
        hdqn_latency = self.results['Hierarchical DQN']['latency'][-1]
        hdqn_cloud = self.results['Hierarchical DQN']['cloud_ratio'][-1]
        hdqn_energy = self.results['Hierarchical DQN']['energy'][-1]
        
        # Build table data
        table_data = []
        table_data.append(['Metric', 'Hierarchical DQN', 'Standalone DQN', 'Simple Hierarchical', 'Random Allocation'])
        
        # SLA Compliance
        sla_row = ['SLA Compliance']
        for app in approaches:
            val = self.results[app]['sla'][-1]
            if app == 'Hierarchical DQN':
                sla_row.append(f"{val:.4f}")
            else:
                improvement = ((val - hdqn_sla) / hdqn_sla * 100) if hdqn_sla != 0 else 0
                sla_row.append(f"{val:.4f}\n({improvement:+.1f}%)")
        table_data.append(sla_row)
        
        # Latency
        latency_row = ['Avg Latency (ms)']
        for app in approaches:
            val = self.results[app]['latency'][-1]
            if app == 'Hierarchical DQN':
                latency_row.append(f"{val:.2f}")
            else:
                improvement = ((hdqn_latency - val) / val * 100) if val != 0 else 0
                latency_row.append(f"{val:.2f}\n({improvement:+.1f}%)")
        table_data.append(latency_row)
        
        # Cloud Ratio
        cloud_row = ['Cloud Offload Ratio']
        for app in approaches:
            val = self.results[app]['cloud_ratio'][-1]
            if app == 'Hierarchical DQN':
                cloud_row.append(f"{val:.4f}")
            else:
                cloud_row.append(f"{val:.4f}")
        table_data.append(cloud_row)
        
        # Energy
        energy_row = ['Energy Consumption (kWh)']
        for app in approaches:
            val = self.results[app]['energy'][-1]
            if app == 'Hierarchical DQN':
                energy_row.append(f"{val:.4f}")
            else:
                improvement = ((hdqn_energy - val) / val * 100) if val != 0 else 0
                energy_row.append(f"{val:.4f}\n({improvement:+.1f}%)")
        table_data.append(energy_row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style metric names
        for i in range(1, 5):
            table[(i, 0)].set_facecolor('#D9E1F2')
            table[(i, 0)].set_text_props(weight='bold')
        
        # Highlight Hierarchical DQN column
        for i in range(1, 5):
            table[(i, 1)].set_facecolor('#E2EFDA')
        
        plt.title('Final Performance Metrics Comparison\n(Values relative to Hierarchical DQN)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(self.comparison_dir, 'final_metrics_table.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: final_metrics_table.png")
        
        # Print to console
        print("\n" + "="*120)
        print(f"{'Metric':<25} {'Hierarchical DQN':<20} {'Standalone DQN':<20} {'Simple Hierarchical':<25} {'Random Allocation':<20}")
        print("="*120)
        
        print(f"{'SLA Compliance':<25} {final_sla[0]:<20.4f} {final_sla[1]:<20.4f} {final_sla[2]:<25.4f} {final_sla[3]:<20.4f}")
        print(f"{'Avg Latency (ms)':<25} {final_latency[0]:<20.2f} {final_latency[1]:<20.2f} {final_latency[2]:<25.2f} {final_latency[3]:<20.2f}")
        print(f"{'Cloud Offload Ratio':<25} {final_cloud_ratio[0]:<20.4f} {final_cloud_ratio[1]:<20.4f} {final_cloud_ratio[2]:<25.4f} {final_cloud_ratio[3]:<20.4f}")
        print(f"{'Energy Consumption':<25} {final_energy[0]:<20.4f} {final_energy[1]:<20.4f} {final_energy[2]:<25.4f} {final_energy[3]:<20.4f}")
        print("="*120)

    def generate_avg_improvement_plot(self):
        """Generate a bargraph showing average metric improvements."""
        print("\nGenerating improvement analysis...")
        
        fig, axs = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle('Hierarchical DQN Performance Improvement over Baselines', 
                    fontsize=14, fontweight='bold')
        
        approaches_to_compare = ['Standalone DQN', 'Simple Hierarchical', 'Random Allocation']
        hdqn_sla = np.mean(self.results['Hierarchical DQN']['sla'][-50:])
        hdqn_latency = np.mean(self.results['Hierarchical DQN']['latency'][-50:])
        hdqn_cloud = np.mean(self.results['Hierarchical DQN']['cloud_ratio'][-50:])
        hdqn_energy = np.mean(self.results['Hierarchical DQN']['energy'][-50:])
        
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
                baseline_val = np.mean(self.results[baseline][key][-50:])
                
                if higher_is_better:
                    # For SLA: higher is better
                    improvement = ((hdqn_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
                else:
                    # For latency, energy, etc.: lower is better
                    improvement = ((baseline_val - hdqn_val) / baseline_val * 100) if baseline_val != 0 else 0
                
                improvements.append(improvement)
                labels.append(baseline.replace(' ', '\n'))
            
            # Create bar plot
            colors_list = ['#ff7f0e', '#2ca02c', '#d62728']
            bars = ax.bar(labels, improvements, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
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
        plt.savefig(os.path.join(self.comparison_dir, 'improvement_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: improvement_analysis.png")

    def _downsample(self, data, step=20):
        """Downsample data for cleaner plots."""
        return data[::step] if len(data) > step else data

    def run_comparison(self):
        """Run full comparison workflow."""
        self.run_all()
        self.generate_comparison_plots()
        self.generate_final_metrics_table()
        self.generate_avg_improvement_plot()
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE!")
        print(f"Results saved to: {self.comparison_dir}")
        print("="*80 + "\n")


if __name__ == '__main__':
    # Load config
    import yaml
    config = {}
    if os.path.exists('fog_rl_medical/config.yaml'):
        with open('fog_rl_medical/config.yaml', 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Run comparison
    runner = ComparisonRunner(config)
    runner.run_comparison()
