"""
Hyperparameter Sweep for RL Agents
===================================

Configurable grid search over hyperparameters.
Edit the HYPERPARAMETER_GRID section to control what gets tested.

Usage:
    python run_hyperparam_sweep.py
    
Output:
    hyperparam_sweep/
        run_001_q_learning_a0.1_e0.01_g0.9/
        run_002_monte_carlo_a0.1_e0.01_g0.9/
        ...
        summary.csv
"""

import os
import subprocess
import itertools
import pandas as pd
from datetime import datetime
import json

# ============================================================================
# HYPERPARAMETER GRID CONFIGURATION
# ============================================================================
# Edit these dictionaries to control what gets tested

BOARD_CONFIGS = [
    {
        'rows': 3,
        'cols': 4,
        'connect_n': 3,
        'episodes': 20000,
        'eval_interval': 1000,
        'checkpoint_interval': 1000,
        'label': '3x4'
    },
    {
        'rows': 4,
        'cols': 4,
        'connect_n': 3,
        'episodes': 50000,
        'eval_interval': 2000,
        'checkpoint_interval': 1000,
        'label': '4x4'
    }
]

# Which algorithms to test
ALGORITHMS = [
    'q_learning',
    'monte_carlo',
    'sarsa'
]

# Learning rates to test
ALPHA_VALUES = [
    0.05,
    0.1,
    0.2,
    0.5
]

# Discount factors to test
GAMMA_VALUES = [
    0.9,
    0.95
]

# Initial Q-values to test
INITIAL_Q_VALUES = [
    0.0,    # Neutral
    0.5     # Optimistic
]

# Epsilon decay schedules to test
# Format: (epsilon_start, epsilon_end, epsilon_decay, label)
EPSILON_SCHEDULES = [
    (1.0, 0.01, 0.9995, 'fast'),           # Reaches 0.01 at ~10K episodes
    (1.0, 0.01, 0.9997, 'medium'),         # Reaches 0.01 at ~20K episodes
    (1.0, 0.05, 0.9997, 'high_min'),       # Keeps exploration at 0.05
    (1.0, 0.001, 0.9997, 'very_low_min'),  # Very low minimum (for SARSA)
]

# Evaluation games (same for all boards)
EVAL_GAMES = 100

# ============================================================================
# FILTERING RULES (Optional - comment out to test everything)
# ============================================================================
def should_skip_config(config):
    """
    Return True if this config should be skipped.
    Use this to avoid testing irrelevant combinations.
    """
    
    # Example 1: Skip very_low_min epsilon for non-SARSA algorithms
    if config['epsilon_label'] == 'very_low_min' and config['algorithm'] != 'sarsa':
        return True
    
    # Example 2: Only test optimistic initialization (initial_q=0.5) with Q-Learning
    # if config['initial_val'] == 0.5 and config['algorithm'] != 'q_learning':
    #     return True
    
    # Example 3: Only test high alpha (0.5) with one epsilon schedule
    # if config['alpha'] == 0.5 and config['epsilon_label'] != 'medium':
    #     return True
    
    return False

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def generate_configs():
    """Generate all hyperparameter combinations."""
    configs = []
    
    for board_config in BOARD_CONFIGS:
        for algo in ALGORITHMS:
            for alpha in ALPHA_VALUES:
                for gamma in GAMMA_VALUES:
                    for initial_q in INITIAL_Q_VALUES:
                        for eps_start, eps_end, eps_decay, eps_label in EPSILON_SCHEDULES:
                            
                            config = {
                                'board': board_config,
                                'algorithm': algo,
                                'alpha': alpha,
                                'gamma': gamma,
                                'initial_val': initial_q,
                                'epsilon_start': eps_start,
                                'epsilon_end': eps_end,
                                'epsilon_decay': eps_decay,
                                'epsilon_label': eps_label
                            }
                            
                            # Apply filtering rules
                            if should_skip_config(config):
                                continue
                            
                            configs.append(config)
    
    return configs

def create_run_name(config, run_id):
    """Create descriptive run name."""
    name = f"run_{run_id:03d}_{config['board']['label']}"
    name += f"_{config['algorithm']}"
    name += f"_a{config['alpha']}"
    name += f"_g{config['gamma']}"
    name += f"_q{config['initial_val']}"
    name += f"_e{config['epsilon_label']}"
    return name

def run_single_config(config, run_id, output_base_dir):
    """Execute training for a single configuration."""
    
    run_name = create_run_name(config, run_id)
    board = config['board']
    
    # Build command
    cmd = [
        'python', 'train_single_config.py',
        '--rows', str(board['rows']),
        '--cols', str(board['cols']),
        '--connect_n', str(board['connect_n']),
        '--algorithm', config['algorithm'],
        '--alpha', str(config['alpha']),
        '--gamma', str(config['gamma']),
        '--epsilon_start', str(config['epsilon_start']),
        '--epsilon_end', str(config['epsilon_end']),
        '--epsilon_decay', str(config['epsilon_decay']),
        '--initial_val', str(config['initial_val']),
        '--episodes', str(board['episodes']),
        '--eval_interval', str(board['eval_interval']),
        '--checkpoint_interval', str(board['checkpoint_interval']),
        '--eval_games', str(EVAL_GAMES),
        '--output_dir', output_base_dir,
        '--run_name', run_name
    ]
    
    print(f"\n{'='*70}")
    print(f"Running: {run_name}")
    print(f"Board: {board['rows']}×{board['cols']}, {board['episodes']} episodes")
    print(f"Config: algo={config['algorithm']}, α={config['alpha']}, γ={config['gamma']}, ε={config['epsilon_label']}")
    print(f"{'='*70}\n")
    
    # Execute
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"⚠️  Run {run_name} failed with return code {result.returncode}")
        return False
    
    return True

def collect_results(output_base_dir):
    """Collect all results into summary CSV."""
    
    results = []
    
    # Find all run directories
    for run_dir in os.listdir(output_base_dir):
        run_path = os.path.join(output_base_dir, run_dir)
        
        if not os.path.isdir(run_path):
            continue
        
        # Load config
        config_path = os.path.join(run_path, 'config.json')
        if not os.path.exists(config_path):
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load metrics
        metrics_path = os.path.join(run_path, 'metrics.csv')
        if not os.path.exists(metrics_path):
            continue
        
        df = pd.read_csv(metrics_path)
        
        # Extract final performance
        final = df.iloc[-1]
        
        # Find convergence episode (first time second position hits 90%)
        convergence_90 = df[df['win_rate_vs_random_second'] >= 0.9]['episode'].min()
        convergence_80 = df[df['win_rate_vs_random_second'] >= 0.8]['episode'].min()
        
        # Compile result
        result = {
            'run_name': run_dir,
            'board_size': f"{config['environment']['rows']}x{config['environment']['cols']}",
            'connect_n': config['environment']['connect_n'],
            'algorithm': config['algorithm'],
            'alpha': config['hyperparameters']['alpha'],
            'gamma': config['hyperparameters']['gamma'],
            'epsilon_start': config['hyperparameters']['epsilon_start'],
            'epsilon_end': config['hyperparameters']['epsilon_end'],
            'epsilon_decay': config['hyperparameters']['epsilon_decay'],
            'initial_val': config['hyperparameters']['initial_val'],
            'episodes': config['training']['episodes'],
            'final_win_rate_first': final['win_rate_vs_random_first'],
            'final_win_rate_second': final['win_rate_vs_random_second'],
            'final_states_visited': final['states_visited'],
            'convergence_to_90_second': convergence_90,
            'convergence_to_80_second': convergence_80,
        }
        
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by board size, algorithm, and performance
    results_df = results_df.sort_values(['board_size', 'algorithm', 'final_win_rate_second'], 
                                        ascending=[True, True, False])
    
    # Save
    summary_path = os.path.join(output_base_dir, 'summary.csv')
    results_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Results summary saved to: {summary_path}")
    print(f"{'='*70}\n")
    
    return results_df

def print_summary_stats(results_df):
    """Print summary statistics."""
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70 + "\n")
    
    for algo in results_df['algorithm'].unique():
        algo_data = results_df[results_df['algorithm'] == algo]
        
        print(f"{algo.upper()}:")
        print(f"  Runs: {len(algo_data)}")
        print(f"  Best second position: {algo_data['final_win_rate_second'].max():.1%}")
        print(f"  Best config: {algo_data.loc[algo_data['final_win_rate_second'].idxmax(), 'run_name']}")
        print(f"  Mean second position: {algo_data['final_win_rate_second'].mean():.1%}")
        print()

def main():
    """Main execution."""
    
    # Configuration
    output_base_dir = 'hyperparam_sweep'
    
    # Generate all configs
    configs = generate_configs()
    
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"Board configurations:")
    for board in BOARD_CONFIGS:
        print(f"  - {board['rows']}×{board['cols']} connect-{board['connect_n']}: {board['episodes']} episodes")
    print(f"Total configurations: {len(configs)}")
    
    # Estimate time (different for different boards)
    board_counts = {}
    for config in configs:
        label = config['board']['label']
        board_counts[label] = board_counts.get(label, 0) + 1
    
    total_time = 0
    for board in BOARD_CONFIGS:
        count = board_counts.get(board['label'], 0)
        time_per_run = board['episodes'] / 10000 * 2  # Roughly 2 min per 10K episodes
        total_time += count * time_per_run
    
    print(f"Estimated time: ~{total_time:.0f} minutes ({total_time/60:.1f} hours)")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*70}\n")
    
    # Show first few configs
    print("Sample configurations:")
    for i, config in enumerate(configs[:3]):
        print(f"  {i+1}. {config}")
    if len(configs) > 3:
        print(f"  ... and {len(configs) - 3} more")
    print()
    
    # Confirm
    response = input("Proceed with sweep? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Run all configs
    start_time = datetime.now()
    successful = 0
    
    for i, config in enumerate(configs, 1):
        success = run_single_config(config, i, output_base_dir)
        if success:
            successful += 1
        
        print(f"\nProgress: {i}/{len(configs)} runs completed")
        print(f"Successful: {successful}/{i}")
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"Elapsed time: {elapsed:.1f} minutes")
        if i < len(configs):
            remaining = (elapsed / i) * (len(configs) - i)
            print(f"Estimated remaining: {remaining:.1f} minutes")
    
    # Collect and summarize results
    print("\n" + "="*70)
    print("COLLECTING RESULTS...")
    print("="*70)
    
    results_df = collect_results(output_base_dir)
    print_summary_stats(results_df)
    
    # Final summary
    total_time = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "="*70)
    print("SWEEP COMPLETE")
    print("="*70)
    print(f"Total runs: {len(configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(configs) - successful}")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Results saved to: {output_base_dir}/summary.csv")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
