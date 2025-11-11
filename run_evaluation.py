"""
Main Evaluation Script

Demonstrates comprehensive evaluation workflow using all modules:
- Load agents
- Run matchups/tournaments
- Perform statistical analysis
- Analyze game quality
- Generate visualizations
- Save results
"""

from argparse import ArgumentParser
import sys
from pathlib import Path

from connect_four_env import ConnectFourEnv
from evaluator import Evaluator
from metrics import StatisticalAnalyzer, generate_statistical_report
from game_analyzer import GameAnalyzer, generate_game_quality_report
from visualizations import Visualizer


def run_comprehensive_evaluation(
    red_agent_workspace: str,
    black_agent_workspace: str,
    num_games: int = 100,
    output_dir: str = "evaluation_results"
):
    """
    Run complete evaluation pipeline.
    
    Args:
        red_agent_workspace: Path to red agent workspace
        black_agent_workspace: Path to black agent workspace
        num_games: Number of games to play
        output_dir: Directory for outputs
    """
    print("="*70)
    print("COMPREHENSIVE CONNECT FOUR AGENT EVALUATION")
    print("="*70)
    
    # Initialize environment (using standard 6x7 board)
    print("\n1. Initializing environment...")
    env = ConnectFourEnv(rows=6, cols=7, connect_n=4)
    
    # Initialize evaluator
    print("2. Initializing evaluator...")
    evaluator = Evaluator(env, output_dir=output_dir)
    
    # Load agents
    print("3. Loading agents...")
    red_name = evaluator.load_agent(red_agent_workspace, "red_agent")
    black_name = evaluator.load_agent(black_agent_workspace, "black_agent")
    
    # Run matchup
    print(f"\n4. Running matchup ({num_games} games)...")
    matchup = evaluator.evaluate_matchup(
        red_agent_name=red_name,
        black_agent_name=black_name,
        num_games=num_games,
        verbose=False,
        show_progress=True
    )
    
    # Statistical analysis
    print("\n5. Performing statistical analysis...")
    stat_report = generate_statistical_report(matchup, confidence=0.95)
    print(stat_report)
    
    # Game quality analysis
    print("\n6. Analyzing game quality...")
    game_analyzer = GameAnalyzer(env)
    game_analyses = game_analyzer.analyze_multiple_games(matchup.games, progress=True)
    
    quality_report = generate_game_quality_report(
        game_analyses,
        red_agent=matchup.red_agent,
        black_agent=matchup.black_agent
    )
    print(quality_report)
    
    # Generate visualizations
    print("\n7. Generating visualizations...")
    visualizer = Visualizer(output_dir=f"{output_dir}/plots")
    visualizer.create_comprehensive_report(matchup, game_analyses, env.cols)
    
    # Save results
    print("\n8. Saving results...")
    results_file = evaluator.save_results()
    
    # Save text reports
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    with open(reports_dir / "statistical_report.txt", 'w') as f:
        f.write(stat_report)
    
    with open(reports_dir / "quality_report.txt", 'w') as f:
        f.write(quality_report)
    
    print(f"\nReports saved to: {reports_dir}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults directory: {output_dir}")
    print(f"  - JSON results: {results_file}")
    print(f"  - Text reports: {reports_dir}")
    print(f"  - Visualizations: {output_dir}/plots")
    
    return matchup, game_analyses


def run_tournament_evaluation(
    agent_workspaces: list,
    games_per_matchup: int = 100,
    output_dir: str = "tournament_results"
):
    """
    Run tournament between multiple agents.
    
    Args:
        agent_workspaces: List of paths to agent workspaces
        games_per_matchup: Games per pairwise matchup
        output_dir: Directory for outputs
    """
    print("="*70)
    print("TOURNAMENT EVALUATION")
    print("="*70)
    
    # Initialize
    env = ConnectFourEnv(rows=6, cols=7, connect_n=4)
    evaluator = Evaluator(env, output_dir=output_dir)
    
    # Load all agents
    print(f"\nLoading {len(agent_workspaces)} agents...")
    agent_names = []
    for workspace in agent_workspaces:
        name = evaluator.load_agent(workspace)
        agent_names.append(name)
    
    # Run tournament
    print(f"\nRunning tournament...")
    tournament_results = evaluator.run_tournament(
        agent_names=agent_names,
        games_per_matchup=games_per_matchup,
        bidirectional=True
    )
    
    # Visualize tournament matrix
    print("\nGenerating tournament visualization...")
    visualizer = Visualizer(output_dir=f"{output_dir}/plots")
    visualizer.plot_tournament_matrix(tournament_results)
    
    # Save results
    evaluator.save_results(filename="tournament_results.json")
    
    print("\n" + "="*70)
    print("TOURNAMENT COMPLETE")
    print("="*70)
    
    return tournament_results


def compare_checkpoints(
    workspace_base: str,
    checkpoint_numbers: list,
    opponent_workspace: str,
    games_per_checkpoint: int = 100,
    output_dir: str = "checkpoint_comparison"
):
    """
    Compare performance across training checkpoints.
    
    Args:
        workspace_base: Base path for checkpoints (e.g., "agent_training")
        checkpoint_numbers: List of checkpoint numbers to compare
        opponent_workspace: Fixed opponent for comparison
        games_per_checkpoint: Games per checkpoint
        output_dir: Directory for outputs
    """
    print("="*70)
    print("CHECKPOINT PROGRESSION ANALYSIS")
    print("="*70)
    
    # Initialize
    env = ConnectFourEnv(rows=6, cols=7, connect_n=4)
    evaluator = Evaluator(env, output_dir=output_dir)
    
    # Load opponent
    print("\nLoading opponent agent...")
    opponent_name = evaluator.load_agent(opponent_workspace, "opponent")
    
    # Evaluate each checkpoint
    results = []
    for ckpt in checkpoint_numbers:
        ckpt_workspace = f"{workspace_base}/checkpoint_{ckpt}"
        print(f"\n--- Evaluating Checkpoint {ckpt} ---")
        
        try:
            agent_name = evaluator.load_agent(ckpt_workspace, f"ckpt_{ckpt}")
            
            matchup = evaluator.evaluate_matchup(
                red_agent_name=agent_name,
                black_agent_name=opponent_name,
                num_games=games_per_checkpoint,
                show_progress=True
            )
            
            results.append({
                'checkpoint': ckpt,
                'win_rate': matchup.red_win_rate,
                'matchup': matchup
            })
        except Exception as e:
            print(f"Error loading checkpoint {ckpt}: {e}")
            continue
    
    # Plot learning curve
    if results:
        print("\nGenerating learning curve...")
        import matplotlib.pyplot as plt
        
        checkpoints = [r['checkpoint'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(checkpoints, win_rates, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
        ax.set_ylabel('Win Rate vs Opponent', fontsize=12, fontweight='bold')
        ax.set_title('Learning Progression Across Checkpoints', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plot_dir = Path(output_dir) / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_dir / "learning_curve.png", dpi=300, bbox_inches='tight')
        print(f"Saved learning curve: {plot_dir}/learning_curve.png")
    
    # Save results
    evaluator.save_results(filename="checkpoint_comparison.json")
    
    print("\n" + "="*70)
    print("CHECKPOINT ANALYSIS COMPLETE")
    print("="*70)
    
    return results


def main():
    parser = ArgumentParser(description="Comprehensive Connect Four Agent Evaluation")
    
    # Evaluation mode
    parser.add_argument(
        '--mode',
        choices=['matchup', 'tournament', 'checkpoints'],
        required=True,
        help='Evaluation mode'
    )
    
    # Common arguments
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games per matchup')
    
    # Matchup mode arguments
    parser.add_argument('--red-agent', type=str,
                       help='Path to red agent workspace (matchup mode)')
    parser.add_argument('--black-agent', type=str,
                       help='Path to black agent workspace (matchup mode)')
    
    # Tournament mode arguments
    parser.add_argument('--agents', nargs='+',
                       help='List of agent workspace paths (tournament mode)')
    
    # Checkpoint mode arguments
    parser.add_argument('--workspace-base', type=str,
                       help='Base workspace path (checkpoint mode)')
    parser.add_argument('--checkpoints', nargs='+', type=int,
                       help='Checkpoint numbers to compare (checkpoint mode)')
    parser.add_argument('--opponent', type=str,
                       help='Fixed opponent workspace (checkpoint mode)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'matchup':
            if not args.red_agent or not args.black_agent:
                print("Error: --red-agent and --black-agent required for matchup mode")
                sys.exit(1)
            
            run_comprehensive_evaluation(
                red_agent_workspace=args.red_agent,
                black_agent_workspace=args.black_agent,
                num_games=args.num_games,
                output_dir=args.output_dir
            )
        
        elif args.mode == 'tournament':
            if not args.agents or len(args.agents) < 2:
                print("Error: --agents requires at least 2 agent workspaces for tournament mode")
                sys.exit(1)
            
            run_tournament_evaluation(
                agent_workspaces=args.agents,
                games_per_matchup=args.num_games,
                output_dir=args.output_dir
            )
        
        elif args.mode == 'checkpoints':
            if not args.workspace_base or not args.checkpoints or not args.opponent:
                print("Error: --workspace-base, --checkpoints, and --opponent required for checkpoint mode")
                sys.exit(1)
            
            compare_checkpoints(
                workspace_base=args.workspace_base,
                checkpoint_numbers=args.checkpoints,
                opponent_workspace=args.opponent,
                games_per_checkpoint=args.num_games,
                output_dir=args.output_dir
            )
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
