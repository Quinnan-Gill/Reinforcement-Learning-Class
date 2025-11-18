"""
Master Training and Evaluation Pipeline for Connect Four RL Agents

This script orchestrates complete training and evaluation workflows:
- Trains multiple RL algorithms (Q-Learning, SARSA, Monte Carlo)
- Supports multiple training modes (self-play, vs-random, curriculum)
- Runs comprehensive evaluations across all trained agents
- Generates organized outputs with meaningful names
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List, Dict
import numpy as np

from connect_four_env import ConnectFourEnv
from q_learning import QLearning
from sarsa import Sarsa
from monte_carlo import MonteCarlo

from evaluator import Evaluator
from metrics import generate_statistical_report
from game_analyzer import GameAnalyzer, generate_game_quality_report
from visualizations import Visualizer
from advanced_metrics import TrainingTimer

# Import modular training methods
from training_methods import train_vs_random, train_self_play, train_curriculum


# =============================================================================
# AGENT CREATION
# =============================================================================

def create_agent(algorithm: str, env: ConnectFourEnv, opts: Namespace):
    """
    Factory function to create RL agents.
    
    Args:
        algorithm: Algorithm name ('q-learning', 'sarsa', 'monte-carlo')
        env: ConnectFour environment
        opts: Training options
    
    Returns:
        Instantiated agent
    """
    if algorithm == "q-learning":
        return QLearning(env, opts)
    elif algorithm == "sarsa":
        return Sarsa(env, opts)
    elif algorithm == "monte-carlo":
        return MonteCarlo(env, opts)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# =============================================================================
# TRAINING ORCHESTRATION
# =============================================================================

def train_agent(
    algorithm: str,
    training_mode: str,
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> str:
    """
    Train a single agent using specified method.
    
    Args:
        algorithm: Algorithm name
        training_mode: 'self-play', 'vs-random', or 'curriculum'
        env: ConnectFour environment
        opts: Training options
        output_dir: Base output directory
    
    Returns:
        Path to trained agent workspace
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup workspace FIRST (before creating agent)
    if training_mode == "curriculum":
        workspace = Path(output_dir) / f"{algorithm}_curriculum_{timestamp}"
    else:
        workspace = Path(output_dir) / f"{algorithm}_{training_mode}_{timestamp}"
    
    workspace.mkdir(parents=True, exist_ok=True)
    opts.workspace = str(workspace)
    opts.overwrite = True
    
    # NOW create agent (after workspace is set)
    agent = create_agent(algorithm, env, opts)
    
    print(f"\n{'='*70}")
    print(f"Training {algorithm.upper()} - {training_mode.upper()} Mode")
    print(f"{'='*70}")
    print(f"Output: {workspace}")
    print(f"Episodes: {opts.episodes}")
    
    # Train using appropriate method
    if training_mode == "vs-random":
        print(f"Mode: Bidirectional (both colors train equally)")
        print(f"Total games: {opts.episodes * 2}")
        
        stats = train_vs_random(agent, env, opts.episodes, 
                               bidirectional=True, workspace=workspace)
        
        # Save agent
        np.save(workspace / "best_red_agent", dict(agent.q['red']))
        np.save(workspace / "best_black_agent", dict(agent.q['black']))
        
        print(f"✓ Saved to: {workspace}")
        print(f"Red Q-table states: {stats['red_states']}")
        print(f"Black Q-table states: {stats['black_states']}")
        
        return str(workspace)
    
    elif training_mode == "self-play":
        print(f"Mode: Self-play (both colors learn simultaneously)")
        
        stats = train_self_play(agent, env, opts.episodes, workspace=workspace)
        
        # Save agent
        np.save(workspace / "best_red_agent", dict(agent.q['red']))
        np.save(workspace / "best_black_agent", dict(agent.q['black']))
        
        print(f"✓ Saved to: {workspace}")
        print(f"Red Q-table states: {stats['red_states']}")
        print(f"Black Q-table states: {stats['black_states']}")
        
        return str(workspace)
    
    elif training_mode == "curriculum":
        print(f"Mode: Curriculum learning (progressive training)")
        print(f"Episodes per phase: {opts.episodes}")
        print(f"Curriculum iterations: {opts.curriculum_iterations}")
        
        # Build curriculum config
        curriculum_config = {
            'vs_random_episodes': opts.episodes * 2,  # 2× for foundation
            'selfplay_episodes': opts.episodes * 2,   # 2× for strategy
            'vs_checkpoint_episodes': opts.episodes,  # 1× per iteration
            'iterations': opts.curriculum_iterations
        }
        
        result = train_curriculum(agent, env, curriculum_config, workspace)
        
        # Final checkpoint is already saved in curriculum function
        print(f"✓ Final checkpoint: {result['final_checkpoint']}")
        print(f"Red Q-table states: {result['red_states']}")
        print(f"Black Q-table states: {result['black_states']}")
        
        # Return final checkpoint path for evaluation
        if opts.curriculum_iterations > 0:
            return str(workspace / f"iteration{opts.curriculum_iterations}")
        else:
            return str(workspace / "phase3_vscheckpoint")
    
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def run_pairwise_evaluations(
    agent_paths: Dict[str, str],
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> Dict:
    """Run all pairwise evaluations between trained agents."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path(output_dir) / f"pairwise_evaluations_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running Pairwise Evaluations")
    print(f"{'='*70}")
    print(f"Agents: {list(agent_paths.keys())}")
    print(f"Games per matchup: {opts.eval_games}")
    
    evaluator = Evaluator(env, output_dir=str(eval_dir))
    
    # Load all agents
    for name, path in agent_paths.items():
        evaluator.load_agent(path, name)
    
    # Run all pairwise matchups
    results = {}
    agent_names = list(agent_paths.keys())
    
    for i, agent1 in enumerate(agent_names):
        for j, agent2 in enumerate(agent_names):
            if i >= j:  # Skip self-play and duplicates
                continue
            
            matchup_name = f"{agent1}_vs_{agent2}"
            print(f"\n--- Evaluating: {matchup_name} ---")
            
            # Run matchup
            matchup = evaluator.evaluate_matchup(
                red_agent_name=agent1,
                black_agent_name=agent2,
                num_games=opts.eval_games,
                show_progress=True
            )
            
            # Analyze games
            game_analyzer = GameAnalyzer(env)
            game_analyses = game_analyzer.analyze_multiple_games(matchup.games, progress=False)
            
            # Generate reports
            stat_report = generate_statistical_report(matchup, confidence=0.95)
            quality_report = generate_game_quality_report(game_analyses, agent1, agent2)
            
            # Save reports
            matchup_dir = eval_dir / matchup_name
            matchup_dir.mkdir(exist_ok=True)
            
            with open(matchup_dir / "statistical_report.txt", 'w') as f:
                f.write(stat_report)
            with open(matchup_dir / "quality_report.txt", 'w') as f:
                f.write(quality_report)
            
            # Generate visualizations
            viz = Visualizer(output_dir=str(matchup_dir / "plots"))
            viz.create_comprehensive_report(matchup, game_analyses, env.cols)
            
            results[matchup_name] = {
                'matchup': matchup,
                'analyses': game_analyses
            }
            
            print(f"✓ {matchup_name} complete")
    
    # Save all results
    evaluator.save_results(filename="all_pairwise_results.json")
    
    print(f"\n✓ All pairwise evaluations complete")
    print(f"✓ Results saved to: {eval_dir}")
    
    return results


def run_tournament(
    agent_paths: Dict[str, str],
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> Dict:
    """Run round-robin tournament between all agents."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_dir = Path(output_dir) / f"tournament_{timestamp}"
    tournament_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running Tournament")
    print(f"{'='*70}")
    print(f"Agents: {list(agent_paths.keys())}")
    print(f"Games per matchup: {opts.eval_games}")
    
    evaluator = Evaluator(env, output_dir=str(tournament_dir))
    
    # Load all agents
    for name, path in agent_paths.items():
        evaluator.load_agent(path, name)
    
    # Run tournament
    tournament_results = evaluator.run_tournament(
        agent_names=list(agent_paths.keys()),
        games_per_matchup=opts.eval_games,
        bidirectional=True
    )
    
    # Generate visualizations
    viz = Visualizer(output_dir=str(tournament_dir / "plots"))
    viz.plot_tournament_matrix(tournament_results)
    
    # Save results
    evaluator.save_results(filename="tournament_results.json")
    
    print(f"\n✓ Tournament complete")
    print(f"✓ Results saved to: {tournament_dir}")
    
    return tournament_results


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(
    agent_paths: Dict[str, str],
    training_times: Dict[str, float],
    output_dir: str
):
    """Generate text summary of all results."""
    summary_path = Path(output_dir) / "SUMMARY_REPORT.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CONNECT FOUR RL TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("TRAINED AGENTS:\n")
        f.write("-"*70 + "\n")
        for agent_name, agent_path in agent_paths.items():
            train_time = training_times.get(agent_name, 0)
            f.write(f"\n{agent_name}:\n")
            f.write(f"  Path: {agent_path}\n")
            f.write(f"  Training time: {train_time:.2f}s ({train_time/60:.1f} min)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("See evaluation directories for detailed results\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ Summary report saved to: {summary_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main pipeline orchestrating training and evaluation."""
    
    # Parse arguments
    parser = ArgumentParser(description="Connect Four RL Training Pipeline")
    
    # Agent selection
    parser.add_argument(
        '--agents',
        nargs='+',
        choices=['q-learning', 'sarsa', 'monte-carlo', 'all'],
        default=['all'],
        help='Which agents to train (default: all)'
    )
    
    # Training mode
    parser.add_argument(
        '--training-mode',
        choices=['self-play', 'vs-random', 'curriculum'],
        default='self-play',
        help='Training mode (default: self-play)'
    )
    
    # Training options
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training, only run evaluations on existing agents'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation, only run training'
    )
    
    parser.add_argument(
        '--agent-dirs',
        nargs='+',
        help='Directories of pre-trained agents (for --skip-training mode)'
    )
    
    # Environment configuration
    parser.add_argument('-r', '--rows', type=int, default=6, help='Board rows (default: 6)')
    parser.add_argument('-c', '--columns', type=int, default=7, help='Board columns (default: 7)')
    parser.add_argument('-n', '--connect-n', type=int, default=4, help='Connect-N to win (default: 4)')
    
    # Reward structure
    parser.add_argument('--reward', type=float, default=1.0, help='Win reward (default: 1.0)')
    parser.add_argument('--penalty', type=float, default=0.0, help='Loss/draw penalty (default: 0.0)')
    parser.add_argument('--move-cost', type=float, default=0.0, help='Cost per move (default: 0.0)')
    
    # Training hyperparameters
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate (default: 0.1)')
    parser.add_argument('--episodes', type=int, default=5000, help='Training episodes (default: 5000)')
    parser.add_argument('--initial-val', type=float, default=0.0, help='Initial Q-value (default: 0.0)')
    parser.add_argument('--num-agents', type=int, default=5, help='Ensemble size (default: 5)')
    parser.add_argument('--curriculum-iterations', type=int, default=3, 
                       help='Curriculum iterations (default: 3)')
    
    # Evaluation
    parser.add_argument('--eval-games', type=int, default=100, help='Games per evaluation (default: 100)')
    
    # Output
    parser.add_argument('-o', '--output-dir', type=str, default='results', 
                       help='Output directory (default: results)')
    
    opts = parser.parse_args()
    
    # Expand 'all' to all agents
    if 'all' in opts.agents:
        opts.agents = ['q-learning', 'sarsa', 'monte-carlo']
    
    # Create output directories
    training_dir = Path(opts.output_dir) / "training"
    eval_dir = Path(opts.output_dir) / "evaluations"
    training_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*70}")
    print("CONNECT FOUR RL MASTER PIPELINE")
    print(f"{'='*70}")
    print(f"Environment: {opts.rows}x{opts.columns} Connect-{opts.connect_n}")
    print(f"Agents: {', '.join(opts.agents)}")
    print(f"Training mode: {opts.training_mode}")
    print(f"Episodes: {opts.episodes}")
    print(f"Output directory: {opts.output_dir}")
    print(f"{'='*70}")
    
    # Create environment
    env = ConnectFourEnv(
        rows=opts.rows,
        cols=opts.columns,
        connect_n=opts.connect_n,
        reward=opts.reward,
        penalty=opts.penalty,
        move_cost=opts.move_cost
    )
    
    # Dictionary to store agent paths
    agent_paths = {}
    training_times = {}
    
    # =================================================================
    # TRAINING PHASE
    # =================================================================
    if not opts.skip_training:
        print(f"\n{'='*70}")
        print("PHASE 1: TRAINING")
        print(f"{'='*70}")
        
        for agent_name in opts.agents:
            print(f"\nStarting training: {agent_name}")
            
            with TrainingTimer() as timer:
                agent_path = train_agent(
                    agent_name,
                    opts.training_mode,
                    env,
                    opts,
                    str(training_dir)
                )
            
            elapsed = timer.elapsed
            training_times[agent_name] = elapsed
            
            agent_paths[agent_name] = agent_path
            print(f"✓ {agent_name} training completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        for agent_name, agent_path in agent_paths.items():
            train_time = training_times[agent_name]
            print(f"  {agent_name}: {agent_path}")
            print(f"    Training time: {train_time:.2f}s ({train_time/60:.1f} min)")
    
    else:
        # Load pre-trained agents
        if not opts.agent_dirs:
            print("ERROR: --agent-dirs required when using --skip-training")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("USING PRE-TRAINED AGENTS")
        print(f"{'='*70}")
        
        for agent_dir in opts.agent_dirs:
            agent_name = Path(agent_dir).name
            # Try to extract algorithm from directory name
            for algo in ['q-learning', 'sarsa', 'monte-carlo']:
                if algo in agent_name.lower():
                    agent_paths[algo] = agent_dir
                    print(f"  {algo}: {agent_dir}")
                    break
            else:
                # Fallback: use directory name as agent name
                agent_paths[agent_name] = agent_dir
                print(f"  {agent_name}: {agent_dir}")
    
    # =================================================================
    # EVALUATION PHASE
    # =================================================================
    if not opts.skip_evaluation:
        if len(agent_paths) < 2:
            print("\nWarning: Need at least 2 agents for evaluation, skipping")
        else:
            print(f"\n{'='*70}")
            print("PHASE 2: EVALUATION")
            print(f"{'='*70}")
            
            # Run pairwise evaluations
            pairwise_results = run_pairwise_evaluations(
                agent_paths, env, opts, str(eval_dir)
            )
            
            # Run tournament
            tournament_results = run_tournament(
                agent_paths, env, opts, str(eval_dir)
            )
            
            # Generate summary report
            generate_summary_report(agent_paths, training_times, opts.output_dir)
            
            print(f"\n{'='*70}")
            print("EVALUATION COMPLETE")
            print(f"{'='*70}")
    
    # =================================================================
    # PIPELINE COMPLETE
    # =================================================================
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults directory: {opts.output_dir}")
    print(f"  Training outputs: {training_dir}")
    print(f"  Evaluation outputs: {eval_dir}")
    print(f"  Summary report: {opts.output_dir}/SUMMARY_REPORT.txt")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
