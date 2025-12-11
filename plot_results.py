

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_learning_curve(exp_id, log_dir, save_dir="plots"):
    """
    Plot learning curve from training logs
    
    Args:
        exp_id: Experiment ID
        log_dir: Directory containing training logs
        save_dir: Directory to save plots
    """
    print(f"Plotting learning curve for Experiment {exp_id}...")
    
    try:
        # Load results
        results = load_results(log_dir)
        
        # Extract data
        x, y = ts2xy(results, 'timesteps')
        
        # Smooth the curve
        window_size = 50
        if len(y) > window_size:
            y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
        else:
            y_smooth = y
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw data (semi-transparent)
        ax.plot(x, y, alpha=0.3, color='blue', label='Raw')
        
        # Plot smoothed data
        ax.plot(x, y_smooth, color='darkblue', linewidth=2, label=f'Smoothed (window={window_size})')
        
        ax.set_xlabel('Training Episodes', fontsize=14)
        ax.set_ylabel('Mean Episode Reward', fontsize=14)
        ax.set_title(f'Experiment {exp_id}: Learning Curve', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/exp_{exp_id}_learning_curve.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error plotting learning curve for exp {exp_id}: {e}")
        return False


def plot_performance_test(exp_id, results_file, save_dir="plots"):
    
    print(f"Plotting performance test for Experiment {exp_id}...")
    
    try:
        # Load results
        data = np.load(results_file)
        rewards = data['rewards']
        mean_reward = data['mean_reward']
        std_reward = data['std_reward']
        
        # Create dataframe for seaborn
        df = pd.DataFrame({
            'Experiment': [f'Exp {exp_id}'] * len(rewards),
            'Episode Reward': rewards
        })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Violin plot
        sns.violinplot(data=df, y='Experiment', x='Episode Reward', 
                      orient='h', inner='box', ax=ax, color='skyblue')
        
        # Add mean line
        ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_reward:.2f} ± {std_reward:.2f}')
        
        ax.set_xlabel('Mean Episode Reward', fontsize=14)
        ax.set_ylabel('', fontsize=14)
        ax.set_title(f'Experiment {exp_id}: Performance Test (500 Episodes)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add statistics text
        stats_text = f'Episodes: {len(rewards)}\n'
        stats_text += f'Mean: {mean_reward:.2f}\n'
        stats_text += f'Std: {std_reward:.2f}\n'
        stats_text += f'Min: {np.min(rewards):.2f}\n'
        stats_text += f'Max: {np.max(rewards):.2f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/exp_{exp_id}_performance_test.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error plotting performance test for exp {exp_id}: {e}")
        return False


def plot_all_learning_curves(exp_ids, save_dir="plots"):
   
    print("Plotting all learning curves...")
    
    n_plots = len(exp_ids)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, exp_id in enumerate(exp_ids):
        ax = axes[idx]
        log_dir = f"logs/exp_{exp_id}"
        
        try:
            results = load_results(log_dir)
            x, y = ts2xy(results, 'timesteps')
            
            # Smooth
            window_size = 50
            if len(y) > window_size:
                y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
            else:
                y_smooth = y
            
            ax.plot(x, y, alpha=0.2, color='blue')
            ax.plot(x, y_smooth, color='darkblue', linewidth=2)
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Mean Reward')
            ax.set_title(f'Exp {exp_id}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'No data for Exp {exp_id}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Exp {exp_id}', fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('All Learning Curves', fontsize=20, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = f"{save_dir}/all_learning_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_comparison(exp_ids, labels, save_dir="plots"):
    
    print(f"Plotting comparison for experiments: {exp_ids}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_ids)))
    
    for idx, (exp_id, label) in enumerate(zip(exp_ids, labels)):
        log_dir = f"logs/exp_{exp_id}"
        
        try:
            results = load_results(log_dir)
            x, y = ts2xy(results, 'timesteps')
            
            # Smooth
            window_size = 50
            if len(y) > window_size:
                y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
            else:
                y_smooth = y
            
            ax.plot(x, y_smooth, color=colors[idx], linewidth=2, label=label)
            
        except Exception as e:
            print(f"Could not load data for exp {exp_id}: {e}")
    
    ax.set_xlabel('Training Episodes', fontsize=14)
    ax.set_ylabel('Mean Episode Reward', fontsize=14)
    ax.set_title('Learning Curves Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    save_path = f"{save_dir}/comparison_learning_curves.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_stage_progress(log_dir, save_dir="plots"):
    
    print("Plotting stage progress...")
    
    # This is a placeholder - you'll need to log stage info during training
    # For now, create a sample visualization
    
    stages = ['City', 'Curved', 'School Zone', 'Intersection', 
             'Merge', 'Highway', 'Construction', 'Exit']
    
    # Sample data (replace with actual logged data)
    completion_rates = np.random.uniform(0.5, 1.0, len(stages))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(stages, completion_rates, color='skyblue', edgecolor='darkblue', linewidth=2)
    
    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, completion_rates)):
        ax.text(rate + 0.02, i, f'{rate*100:.1f}%', 
               va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Completion Rate', fontsize=14)
    ax.set_ylabel('Stage', fontsize=14)
    ax.set_title('Stage Completion Progress', fontsize=16, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='x')
    
    save_path = f"{save_dir}/stage_progress.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Multi-Stage Highway Results")
    
    parser.add_argument("--exp-id", type=str, required=True,
                       help="Experiment ID (1-16) or 'all'")
    parser.add_argument("--plot-type", type=str, default="both",
                       choices=["learning", "performance", "both", "comparison"],
                       help="Type of plot to generate")
    parser.add_argument("--save-dir", type=str, default="plots",
                       help="Directory to save plots")
    
    # For comparison plots
    parser.add_argument("--compare-ids", type=str, default=None,
                       help="Comma-separated experiment IDs to compare (e.g., '1,3,5')")
    parser.add_argument("--compare-labels", type=str, default=None,
                       help="Comma-separated labels for comparison (e.g., 'Lidar,Grayscale')")
    
    args = parser.parse_args()
    
    # Handle 'all' experiments
    if args.exp_id.lower() == "all":
        exp_ids = list(range(1, 17))  # 1-16
    else:
        exp_ids = [int(args.exp_id)]
    
    # Generate plots
    if args.plot_type in ["learning", "both"]:
        if args.exp_id.lower() == "all":
            plot_all_learning_curves(exp_ids, args.save_dir)
        else:
            for exp_id in exp_ids:
                log_dir = f"logs/exp_{exp_id}"
                if os.path.exists(log_dir):
                    plot_learning_curve(exp_id, log_dir, args.save_dir)
                else:
                    print(f"Warning: Log directory not found for exp {exp_id}")
    
    if args.plot_type in ["performance", "both"]:
        for exp_id in exp_ids:
            results_file = f"results/exp_{exp_id}/test_results_exp_{exp_id}.npz"
            if os.path.exists(results_file):
                plot_performance_test(exp_id, results_file, args.save_dir)
            else:
                print(f"Warning: Results file not found for exp {exp_id}")
    
    if args.plot_type == "comparison":
        if args.compare_ids is None:
            print("Error: --compare-ids required for comparison plots")
            return
        
        compare_ids = [int(x.strip()) for x in args.compare_ids.split(',')]
        
        if args.compare_labels:
            compare_labels = [x.strip() for x in args.compare_labels.split(',')]
        else:
            compare_labels = [f'Exp {i}' for i in compare_ids]
        
        plot_comparison(compare_ids, compare_labels, args.save_dir)
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
