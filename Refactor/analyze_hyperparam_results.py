"""
Analyze Hyperparameter Sweep Results
=====================================

Generates plots and statistical analysis from sweep results.

Usage:
    python analyze_hyperparam_results.py hyperparam_sweep/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def load_results(sweep_dir):
    """Load summary CSV."""
    summary_path = os.path.join(sweep_dir, 'summary.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.csv found in {sweep_dir}")
    
    df = pd.read_csv(summary_path)
    print(f"Loaded {len(df)} results from {summary_path}")
    return df

def plot_algorithm_comparison(df, output_dir):
    """Compare algorithms across all configs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Win rate distribution by algorithm
    ax = axes[0]
    for algo in df['algorithm'].unique():
        data = df[df['algorithm'] == algo]['final_win_rate_second']
        ax.hist(data, alpha=0.6, label=algo, bins=10)
    ax.set_xlabel('Win Rate (Second Position)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Performance Distribution by Algorithm', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    ax = axes[1]
    df.boxplot(column='final_win_rate_second', by='algorithm', ax=ax)
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (Second Position)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Algorithm', fontsize=13, fontweight='bold')
    plt.suptitle('')  # Remove default title
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: algorithm_comparison.png")
    plt.close()

def plot_alpha_effect(df, output_dir):
    """Plot effect of learning rate."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, algo in enumerate(df['algorithm'].unique()):
        ax = axes[i]
        algo_data = df[df['algorithm'] == algo]
        
        # Group by alpha and get mean/std
        grouped = algo_data.groupby('alpha')['final_win_rate_second'].agg(['mean', 'std', 'count'])
        
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                    marker='o', markersize=8, linewidth=2, capsize=5)
        ax.set_xlabel('Learning Rate (α)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Win Rate (Second Position)', fontsize=12, fontweight='bold')
        ax.set_title(f'{algo.upper()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_effect.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: alpha_effect.png")
    plt.close()

def plot_epsilon_effect(df, output_dir):
    """Plot effect of epsilon schedule."""
    # Extract epsilon label from run_name (format: run_XXX_algo_aX.X_gX.X_qX.X_eEPSLABEL)
    df['epsilon_label'] = df['run_name'].str.extract(r'_e([^_]+)$')[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, algo in enumerate(df['algorithm'].unique()):
        ax = axes[i]
        algo_data = df[df['algorithm'] == algo]
        
        # Group by epsilon schedule
        grouped = algo_data.groupby('epsilon_label')['final_win_rate_second'].agg(['mean', 'std', 'count'])
        
        x_pos = np.arange(len(grouped))
        ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.set_xlabel('Epsilon Schedule', fontsize=12, fontweight='bold')
        ax.set_ylabel('Win Rate (Second Position)', fontsize=12, fontweight='bold')
        ax.set_title(f'{algo.upper()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epsilon_effect.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: epsilon_effect.png")
    plt.close()

def plot_convergence_speed(df, output_dir):
    """Plot convergence speed comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Episodes to 80% by algorithm
    ax = axes[0]
    df_clean = df.dropna(subset=['convergence_to_80_second'])
    df_clean.boxplot(column='convergence_to_80_second', by='algorithm', ax=ax)
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episodes to 80% (Second Position)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Speed', fontsize=13, fontweight='bold')
    plt.suptitle('')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Scatter - convergence vs final performance
    ax = axes[1]
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo].dropna(subset=['convergence_to_80_second'])
        ax.scatter(algo_data['convergence_to_80_second'], 
                  algo_data['final_win_rate_second'],
                  label=algo, s=50, alpha=0.6)
    ax.set_xlabel('Episodes to 80%', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Win Rate (Second)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence vs Final Performance', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: convergence_analysis.png")
    plt.close()

def print_best_configs(df):
    """Print best configuration for each algorithm."""
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70 + "\n")
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        best = algo_data.loc[algo_data['final_win_rate_second'].idxmax()]
        
        print(f"{algo.upper()}:")
        print(f"  Best run: {best['run_name']}")
        print(f"  Win rate (first): {best['final_win_rate_first']:.1%}")
        print(f"  Win rate (second): {best['final_win_rate_second']:.1%}")
        print(f"  Alpha: {best['alpha']}")
        print(f"  Gamma: {best['gamma']}")
        print(f"  Epsilon: start={best['epsilon_start']}, end={best['epsilon_end']}, decay={best['epsilon_decay']}")
        print(f"  Initial Q: {best['initial_val']}")
        if not pd.isna(best['convergence_to_80_second']):
            print(f"  Converged to 80% at: {int(best['convergence_to_80_second'])} episodes")
        print()

def print_statistical_summary(df):
    """Print statistical analysis."""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70 + "\n")
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        
        print(f"{algo.upper()} ({len(algo_data)} runs):")
        print(f"  Mean (second): {algo_data['final_win_rate_second'].mean():.1%} ± {algo_data['final_win_rate_second'].std():.1%}")
        print(f"  Median (second): {algo_data['final_win_rate_second'].median():.1%}")
        print(f"  Min (second): {algo_data['final_win_rate_second'].min():.1%}")
        print(f"  Max (second): {algo_data['final_win_rate_second'].max():.1%}")
        print()

def main():
    """Main analysis."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_hyperparam_results.py <sweep_directory>")
        sys.exit(1)
    
    sweep_dir = sys.argv[1]
    
    # Create analysis output directory
    analysis_dir = os.path.join(sweep_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load results
    df = load_results(sweep_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_algorithm_comparison(df, analysis_dir)
    plot_alpha_effect(df, analysis_dir)
    plot_epsilon_effect(df, analysis_dir)
    plot_convergence_speed(df, analysis_dir)
    
    # Print summaries
    print_best_configs(df)
    print_statistical_summary(df)
    
    print("\n" + "="*70)
    print(f"✓ Analysis complete. Results saved to: {analysis_dir}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
