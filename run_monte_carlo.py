import os
import glob
from argparse import ArgumentParser, Namespace
from connect_four_env import ConnectFourEnv
from monte_carlo import MonteCarlo
from random_agent import RandomAgent
from frozen_agent import FrozenAgent


def train_vs_random(env: ConnectFourEnv, opts: Namespace):
    """Phase 1: Train Monte Carlo agent against random opponent"""
    print("\n" + "="*60)
    print("PHASE 1: Training vs Random Opponent")
    print("="*60)
    
    agent = MonteCarlo(env, opts)
    opponent = RandomAgent(env)
    agent.train(opponent=opponent, phase_name="Phase1-Random")


def train_self_play(env: ConnectFourEnv, opts: Namespace):
    """Phase 2: Train Monte Carlo agent via self-play"""
    print("\n" + "="*60)
    print("PHASE 2: Training via Self-Play")
    print("="*60)
    
    agent = MonteCarlo(env, opts)
    agent.train(opponent=None, phase_name="Phase2-SelfPlay")


def train_curriculum(env: ConnectFourEnv, opts: Namespace):
    """Phase 3: Train new agent against progressively stronger opponents"""
    print("\n" + "="*60)
    print("PHASE 3: Curriculum Learning")
    print("="*60)
    
    if not opts.checkpoint_dir or not os.path.exists(opts.checkpoint_dir):
        print("ERROR: Checkpoint directory required for curriculum learning.")
        print("Please run Phase 1 and Phase 2 first to generate checkpoints.")
        return
    
    # Find Phase 1 and Phase 2 checkpoints
    phase1_checkpoints = sorted(glob.glob(os.path.join(opts.checkpoint_dir, "monte-carlo--phase1-random--*.save")))
    phase2_checkpoints = sorted(glob.glob(os.path.join(opts.checkpoint_dir, "monte-carlo--phase2-selfplay--*.save")))
    
    if not phase1_checkpoints:
        print("ERROR: No Phase 1 checkpoint found. Run Phase 1 first:")
        print("  python train_monte_carlo.py --mode random --episodes 1000 -o checkpoints/")
        return
    
    if not phase2_checkpoints:
        print("ERROR: No Phase 2 checkpoint found. Run Phase 2 first:")
        print("  python train_monte_carlo.py --mode self-play --episodes 1000 -o checkpoints/")
        return
    
    # Use most recent checkpoints
    phase1_checkpoint = phase1_checkpoints[-1]
    phase2_checkpoint = phase2_checkpoints[-1]
    
    print(f"\nUsing Phase 1 checkpoint: {os.path.basename(phase1_checkpoint)}")
    print(f"Using Phase 2 checkpoint: {os.path.basename(phase2_checkpoint)}")
    
    # Create new agent for curriculum learning
    agent = MonteCarlo(env, opts)
    
    # Stage 1: Train against Phase 1 agent (easier opponent)
    print(f"\n--- Curriculum Stage 1: Training vs Phase 1 Agent ---")
    phase1_opponent = CheckpointAgent(env, phase1_checkpoint, player=-1)
    agent.train(opponent=phase1_opponent, phase_name="Phase3-Curriculum-Stage1")
    
    # Stage 2: Continue training same agent against Phase 2 agent (harder opponent)
    print(f"\n--- Curriculum Stage 2: Training vs Phase 2 Agent ---")
    phase2_opponent = CheckpointAgent(env, phase2_checkpoint, player=-1)
    agent.train(opponent=phase2_opponent, phase_name="Phase3-Curriculum-Stage2")


def train_iterative_curriculum(env: ConnectFourEnv, opts: Namespace):
    """Iterative Curriculum: Train N generations, each against the previous best"""
    print("\n" + "="*60)
    print(f"ITERATIVE CURRICULUM LEARNING: {opts.curriculum_iterations} Iterations")
    print("="*60)
    
    if not opts.checkpoint_dir or not os.path.exists(opts.checkpoint_dir):
        print("ERROR: Checkpoint directory required for iterative curriculum.")
        return
    
    # Check for Phase 1 and Phase 2 checkpoints
    phase1_checkpoints = sorted(glob.glob(os.path.join(opts.checkpoint_dir, "monte-carlo--phase1-random--*.save")))
    phase2_checkpoints = sorted(glob.glob(os.path.join(opts.checkpoint_dir, "monte-carlo--phase2-selfplay--*.save")))
    
    if not phase1_checkpoints:
        print("ERROR: No Phase 1 checkpoint found. Run Phase 1 first:")
        print("  python train_monte_carlo.py --mode random --episodes 1000 -o checkpoints/")
        return
    
    if not phase2_checkpoints:
        print("ERROR: No Phase 2 checkpoint found. Run Phase 2 first:")
        print("  python train_monte_carlo.py --mode self-play --episodes 1000 -o checkpoints/")
        return
    
    phase1_checkpoint = phase1_checkpoints[-1]
    phase2_checkpoint = phase2_checkpoints[-1]
    
    print(f"\nBaseline checkpoints:")
    print(f"  Phase 1 (Random): {os.path.basename(phase1_checkpoint)}")
    print(f"  Phase 2 (Self-Play): {os.path.basename(phase2_checkpoint)}")
    print()
    
    # Iteration 1: Train against Phase 1 and Phase 2 (special first iteration)
    print("="*60)
    print("ITERATION 1: Curriculum v1 (Train vs Phase 1 → Phase 2)")
    print("="*60)
    
    agent_v1 = MonteCarlo(env, opts)
    
    # Stage 1: vs Phase 1
    print(f"\n--- Stage 1: Training vs Phase 1 Agent ---")
    phase1_opponent = CheckpointAgent(env, phase1_checkpoint, player=-1)
    agent_v1.train(opponent=phase1_opponent, phase_name="Curriculum-v1-Stage1")
    
    # Stage 2: vs Phase 2
    print(f"\n--- Stage 2: Training vs Phase 2 Agent ---")
    phase2_opponent = CheckpointAgent(env, phase2_checkpoint, player=-1)
    agent_v1.train(opponent=phase2_opponent, phase_name="Curriculum-v1")
    
    # Find the most recent curriculum-v1 checkpoint (the Stage 2 one)
    curriculum_checkpoints = sorted(glob.glob(os.path.join(opts.checkpoint_dir, "monte-carlo--curriculum-v1--*.save")))
    if not curriculum_checkpoints:
        print("ERROR: Failed to find curriculum-v1 checkpoint after training")
        return
    
    previous_checkpoint = curriculum_checkpoints[-1]
    print(f"\n✓ Curriculum v1 saved: {os.path.basename(previous_checkpoint)}")
    
    # Iterations 2 through N: Each trains against the previous
    for iteration in range(2, opts.curriculum_iterations + 1):
        print("\n" + "="*60)
        print(f"ITERATION {iteration}: Curriculum v{iteration} (Train vs Curriculum v{iteration-1})")
        print("="*60)
        
        # Create new agent
        new_agent = MonteCarlo(env, opts)
        
        # Train against previous curriculum agent
        print(f"\nTraining vs: {os.path.basename(previous_checkpoint)}")
        previous_opponent = CheckpointAgent(env, previous_checkpoint, player=-1)
        new_agent.train(opponent=previous_opponent, phase_name=f"Curriculum-v{iteration}")
        
        # Find the newly created checkpoint
        new_checkpoints = sorted(glob.glob(os.path.join(opts.checkpoint_dir, f"monte-carlo--curriculum-v{iteration}--*.save")))
        if not new_checkpoints:
            print(f"ERROR: Failed to find curriculum-v{iteration} checkpoint after training")
            return
        
        previous_checkpoint = new_checkpoints[-1]
        print(f"✓ Curriculum v{iteration} saved: {os.path.basename(previous_checkpoint)}")
    
    print("\n" + "="*60)
    print(f"ITERATIVE CURRICULUM COMPLETE: {opts.curriculum_iterations} generations trained")
    print("="*60)
    
    # Summary
    print("\nCheckpoints created:")
    print(f"  Baseline - Phase 1 (Random): {os.path.basename(phase1_checkpoints[-1])}")
    print(f"  Baseline - Phase 2 (Self-Play): {os.path.basename(phase2_checkpoints[-1])}")
    for v in range(1, opts.curriculum_iterations + 1):
        v_checkpoints = sorted(glob.glob(os.path.join(opts.checkpoint_dir, f"monte-carlo--curriculum-v{v}--*.save")))
        if v_checkpoints:
            print(f"  Curriculum v{v}: {os.path.basename(v_checkpoints[-1])}")


def run_all_phases(env: ConnectFourEnv, opts: Namespace):
    """Run complete training pipeline: Random → Self-Play → Curriculum"""
    print("\n" + "="*60)
    print("RUNNING COMPLETE TRAINING PIPELINE")
    print("="*60)
    
    # Phase 1: vs Random
    train_vs_random(env, opts)
    
    # Phase 2: Self-Play
    train_self_play(env, opts)
    
    # Phase 3: Curriculum
    train_curriculum(env, opts)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)


def run_full_pipeline(env: ConnectFourEnv, opts: Namespace):
    """Run complete pipeline: Random → Self-Play → Iterative Curriculum"""
    
    if opts.training_runs == 1:
        # Single run - original behavior
        print("\n" + "="*60)
        print(f"RUNNING FULL PIPELINE (with {opts.curriculum_iterations} curriculum iterations)")
        print("="*60)
        
        # Phase 1: vs Random
        train_vs_random(env, opts)
        
        # Phase 2: Self-Play
        train_self_play(env, opts)
        
        # Iterative Curriculum
        train_iterative_curriculum(env, opts)
        
        print("\n" + "="*60)
        print("FULL PIPELINE COMPLETE")
        print("="*60)
    else:
        # Multiple runs
        print("\n" + "="*70)
        print(f"RUNNING FULL PIPELINE: {opts.training_runs} runs × {opts.curriculum_iterations} curriculum iterations")
        print("="*70)
        
        original_checkpoint_dir = opts.checkpoint_dir
        
        for run in range(1, opts.training_runs + 1):
            print("\n" + "="*70)
            print(f"RUN {run} / {opts.training_runs}")
            print("="*70)
            
            # Create run-specific checkpoint directory
            run_checkpoint_dir = os.path.join(original_checkpoint_dir, f"run{run}")
            if not os.path.exists(run_checkpoint_dir):
                os.makedirs(run_checkpoint_dir)
            opts.checkpoint_dir = run_checkpoint_dir
            
            # Phase 1: vs Random
            train_vs_random(env, opts)
            
            # Phase 2: Self-Play
            train_self_play(env, opts)
            
            # Iterative Curriculum
            train_iterative_curriculum(env, opts)
            
            print(f"\n✓ Run {run} complete - checkpoints saved to {run_checkpoint_dir}")
        
        # Restore original checkpoint directory
        opts.checkpoint_dir = original_checkpoint_dir
        
        print("\n" + "="*70)
        print(f"ALL {opts.training_runs} RUNS COMPLETE")
        print("="*70)
        print(f"\nCheckpoints organized in: {original_checkpoint_dir}/run1/ through run{opts.training_runs}/")


def main():
    parser = ArgumentParser(description="Train Monte Carlo agent with different strategies")
    
    # Training mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--mode", choices=["random", "self-play", "curriculum", "iterative-curriculum"],
                           help="Training mode: random, self-play, curriculum, or iterative-curriculum")
    mode_group.add_argument("--run-all-phases", action="store_true",
                           help="Run all three phases sequentially (random → self-play → curriculum)")
    mode_group.add_argument("--run-full-pipeline", action="store_true",
                           help="Run full pipeline (random → self-play → iterative curriculum)")
    
    # Environment configuration
    parser.add_argument('-r', "--rows", type=int, default=3,
                       help="Number of rows in the board (default: 3)")
    parser.add_argument('-c', "--columns", type=int, default=4,
                       help="Number of columns in the board (default: 4)")
    parser.add_argument('-n', "--connect-n", type=int, default=3,
                       help="Number in a row needed to win (default: 3)")
    
    # Rewards
    parser.add_argument("--reward", type=float, default=1.0,
                       help="Reward for winning (default: 1.0)")
    parser.add_argument("--penalty", type=float, default=0.0,
                       help="Penalty for losing/draw (default: 0.0)")
    parser.add_argument("--move-cost", type=float, default=0.0,
                       help="Cost per move (default: 0.0)")
    
    # Hyperparameters
    parser.add_argument("--gamma", type=float, default=0.8,
                       help="Discount factor (default: 0.8)")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Learning rate (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=0.2,
                       help="Exploration rate (default: 0.2)")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes per phase/iteration (default: 1000)")
    parser.add_argument("--initial-val", type=float, default=0.0,
                       help="Initial Q-value (default: 0.0)")
    
    # Iterative curriculum specific
    parser.add_argument("--curriculum-iterations", type=int, default=5,
                       help="Number of curriculum iterations (default: 5, for iterative-curriculum mode)")
    
    # Multiple runs for statistical significance
    parser.add_argument("--training-runs", type=int, default=1,
                       help="Number of independent training runs (default: 1, for statistical significance use 30)")
    
    # Checkpointing
    parser.add_argument('-o', "--checkpoint-dir", default="checkpoints", type=str,
                       help="Directory to save/load checkpoints (default: checkpoints)")
    
    opts = parser.parse_args()
    
    # Create environment
    env = ConnectFourEnv(
        opts.rows,
        opts.columns,
        opts.connect_n,
        opts.reward,
        opts.penalty,
        opts.move_cost
    )
    
    # Route to appropriate training function
    if opts.run_all_phases:
        run_all_phases(env, opts)
    elif opts.run_full_pipeline:
        run_full_pipeline(env, opts)
    elif opts.mode == "random":
        train_vs_random(env, opts)
    elif opts.mode == "self-play":
        train_self_play(env, opts)
    elif opts.mode == "curriculum":
        train_curriculum(env, opts)
    elif opts.mode == "iterative-curriculum":
        train_iterative_curriculum(env, opts)


if __name__ == '__main__':
    main()
