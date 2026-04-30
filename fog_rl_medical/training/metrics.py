import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import numpy as np

class MetricsRecorder:
    def __init__(self, save_dir='results/'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.history = {
            'sla_compliance': [],
            'avg_latency': [],
            'cloud_offload_ratio': [],
            'energy_consumption': []
        }
        # Store detailed per-episode states for heatmaps
        self.episode_states = []  # List of episode state snapshots
        self.num_nodes = 5

    def record_episode(self, sla_comp, latency, offload_ratio, energy=0.0, episode_state=None):
        self.history['sla_compliance'].append(sla_comp)
        self.history['avg_latency'].append(latency)
        self.history['cloud_offload_ratio'].append(offload_ratio)
        self.history['energy_consumption'].append(energy)
        if episode_state is not None:
            self.episode_states.append(episode_state)

    def plot_metrics(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Downsample data to every 20th value
        def downsample(data, step=20):
            return data[::step] if len(data) > step else data
        
        episodes = np.arange(len(self.history['sla_compliance']))
        sampled_episodes = downsample(episodes)
        
        axs[0, 0].plot(sampled_episodes, downsample(self.history['sla_compliance']), linewidth=2, marker='o')
        axs[0, 0].set_title('SLA Compliance over Episodes', fontsize=12, fontweight='bold')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Compliance Rate')
        axs[0, 0].grid(True, alpha=0.3)
        
        axs[0, 1].plot(sampled_episodes, downsample(self.history['avg_latency']), linewidth=2, marker='s', color='orange')
        axs[0, 1].set_title('Average Latency over Episodes', fontsize=12, fontweight='bold')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Latency (ms)')
        axs[0, 1].grid(True, alpha=0.3)
        
        axs[1, 0].plot(sampled_episodes, downsample(self.history['cloud_offload_ratio']), linewidth=2, marker='^', color='green')
        axs[1, 0].set_title('Cloud Offload Ratio over Episodes', fontsize=12, fontweight='bold')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Offload Ratio')
        axs[1, 0].grid(True, alpha=0.3)
        
        axs[1, 1].plot(sampled_episodes, downsample(self.history['energy_consumption']), linewidth=2, marker='d', color='red')
        axs[1, 1].set_title('Energy Consumption over Episodes', fontsize=12, fontweight='bold')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Energy (kWh)')
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=150)
        plt.close()
        
        # Generate priority distribution heatmap
        self.plot_priority_heatmap()

    def plot_priority_heatmap(self):
        """Create heatmap for average priority task distribution per 100-episode windows."""
        if not self.episode_states:
            return
        
        # Group episodes into 100-episode windows and calculate averages
        window_size = 100
        num_windows = (len(self.episode_states) + window_size - 1) // window_size  # Ceiling division
        
        priority_averages = np.zeros((4, num_windows))  # 4 priority levels × number of windows
        
        for window_idx in range(num_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, len(self.episode_states))
            
            # Collect all priority distributions in this window
            window_data = []
            for ep in range(start_idx, end_idx):
                window_data.append(self.episode_states[ep].priority_distribution)
            
            if window_data:
                # Calculate average priority distribution for this window
                window_array = np.array(window_data)  # Shape: (num_episodes_in_window, 4)
                priority_averages[:, window_idx] = np.mean(window_array, axis=0)
        
        # Priority distributions are already normalized (0-1), convert to percentages
        priority_percentages = priority_averages * 100
        
        # Create window labels (e.g., "0-99", "100-199", etc.)
        window_labels = [f"{i*window_size}-{(i+1)*window_size-1}" for i in range(num_windows)]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Create custom colormap with better contrast
        colors = ['#FFF3CD', '#FF6B6B', '#4ECDC4', '#2C3E50']  # Light yellow, Red, Teal, Dark blue
        cmap = plt.cm.colors.ListedColormap(colors)
        
        # Create heatmap with better formatting
        heatmap = sns.heatmap(priority_percentages, annot=True, fmt='.1f', cmap=cmap, ax=ax, 
                             cbar_kws={'label': 'Average Percentage (%)', 'shrink': 0.8},
                             linewidths=0.5, linecolor='white', square=False,
                             vmin=0, vmax=100)
        
        ax.set_title('Average Priority Task Distribution per 100-Episode Window', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Episode Window', fontsize=12, fontweight='bold')
        ax.set_ylabel('Priority Level', fontsize=12, fontweight='bold')
        
        # Better y-axis labels
        priority_labels = ['Priority 1\n(Lowest)', 'Priority 2', 'Priority 3', 'Priority 4\n(Highest)']
        ax.set_yticklabels(priority_labels, rotation=0, fontsize=10)
        
        # Set x-axis ticks and labels
        ax.set_xticks(np.arange(len(window_labels)))
        ax.set_xticklabels(window_labels, fontsize=9)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid lines
        ax.grid(False)  # Disable default grid, we'll add our own
        for i in range(len(window_labels)):
            ax.axvline(i + 0.5, color='white', linewidth=1, alpha=0.3)
        for i in range(4):
            ax.axhline(i + 0.5, color='white', linewidth=1, alpha=0.3)
        
        # Add color legend with better positioning
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], edgecolor='black', linewidth=0.5) 
                          for i in range(4)]
        legend_labels = ['Priority 1 (Avg %)', 'Priority 2 (Avg %)', 'Priority 3 (Avg %)', 'Priority 4 (Avg %)']
        ax.legend(legend_elements, legend_labels, 
                 loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9,
                 title='Priority Levels', title_fontsize=10)
        
        # Add text annotation for interpretation
        ax.text(1.02, 0.5, 'Shows average task\ndistribution over\neach 100-episode\ntraining window', 
               transform=ax.transAxes, fontsize=8, verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'priority_distribution_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
