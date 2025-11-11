"""
Master Training and Evaluation Pipeline for Connect Four RL Agents

This script orchestrates complete training and evaluation workflows:
- Trains multiple RL algorithms (Q-Learning, SARSA, Monte Carlo, DQN*)
- Supports multiple training modes (self-play, vs-random, curriculum)
- Runs comprehensive evaluations across all trained agents
- Generates organized outputs with meaningful names

*DQN support to be added
"""

import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict
from copy import copy

from connect_four_env import ConnectFourEnv
from rl_agent import RLModel, random_argmax
from q_learning import QLearning
from sarsa import Sarsa
from expected_sarsa import ExpectedSarsa
from monte_carlo import MonteCarlo
from random_agent import RandomAgent
from frozen_agent import FrozenAgent
from checkpoints import save_learning_curve

from evaluator import Evaluator
from metrics import generate_statistical_report
from game_analyzer import GameAnalyzer, generate_game_quality_report
from visualizations import Visualizer
from advanced_metrics import TrainingTimer, generate_q_table_report


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_tabular_agent_selfplay(
    agent_class,
    agent_name: str,
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> str:
    """
    Train Q-Learning or SARSA via self-play with ensemble.
    
    Returns:
        Path to trained agent workspace
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = Path(output_dir) / f"{agent_name}_selfplay_{timestamp}"
    
    print(f"\n{'='*70}")
    print(f"Training {agent_name.upper()} - Self-Play Mode")
    print(f"{'='*70}")
    print(f"Output: {workspace}")
    print(f"Episodes: {opts.episodes}")
    print(f"Ensemble size: {opts.num_agents}")
    
    # Create workspace
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Update opts with workspace
    opts.workspace = str(workspace)
    opts.overwrite = True
    
    # Create ensemble of agents
    agents = []
    base_agent = agent_class(env, opts)
    for _ in range(opts.num_agents):
        agents.append(copy(base_agent))
    
    # Train ensemble
    learning_curve_red = []
    learning_curve_black = []
    
    for i in range(opts.episodes):
        print(f"\rEpisode {i+1}/{opts.episodes}", end="")
        
        total_rewards = defaultdict(list)
        for agent in agents:
            total_reward = agent.train_step(i)
            for player, val in total_reward.items():
                total_rewards[player].append(val)
        
        learning_curve_red.append(np.array(total_rewards['red']))
        learning_curve_black.append(np.array(total_rewards['black']))
    
    print(f"\n✓ Training complete")
    
    # Save learning curves
    learning_curve_red = np.array(learning_curve_red)
    learning_curve_black = np.array(learning_curve_black)
    save_learning_curve(opts.workspace, player='red', data=learning_curve_red)
    save_learning_curve(opts.workspace, player='black', data=learning_curve_black)
    
    # Save best agents from ensemble
    best_red_idx = np.argmax(np.sum(learning_curve_red[-10:,], axis=0))
    best_red_agent = agents[best_red_idx].q['red']
    
    best_black_idx = np.argmax(np.sum(learning_curve_black[-10:,], axis=0))
    best_black_agent = agents[best_black_idx].q['black']
    
    np.save(workspace / "best_red_agent", dict(best_red_agent))
    np.save(workspace / "best_black_agent", dict(best_black_agent))
    
    print(f"✓ Saved to: {workspace}")
    
    return str(workspace)


def train_tabular_agent_vs_random(
    agent_class,
    agent_name: str,
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> str:
    """
    Train Q-Learning or SARSA against random opponent.
    Agent plays as Red (player 1), random plays as Black (player -1).
    
    Returns:
        Path to trained agent workspace
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = Path(output_dir) / f"{agent_name}_vsrandom_{timestamp}"
    
    print(f"\n{'='*70}")
    print(f"Training {agent_name.upper()} - vs Random Opponent")
    print(f"{'='*70}")
    print(f"Output: {workspace}")
    print(f"Episodes: {opts.episodes}")
    
    # Create workspace
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Update opts with workspace
    opts.workspace = str(workspace)
    opts.overwrite = True
    
    # Create agent
    agent = agent_class(env, opts)
    
    # Create random opponent
    random_opponent = RandomAgent(env)
    
    # Train agent vs random
    learning_curve_red = []
    
    for i in range(opts.episodes):
        print(f"\rEpisode {i+1}/{opts.episodes}", end="")
        
        env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            current_player = env.current_player
            state = env.get_state_key()
            
            # Select action
            if current_player == 1:  # Agent (Red)
                action = agent.select_action(state, current_player)
            else:  # Random opponent (Black)
                action = random_opponent.make_move()
            
            # Execute move
            _, reward, done, _ = env.make_move(action)
            next_state = env.get_state_key()
            
            # Update agent's Q-values (only when agent acts)
            if current_player == 1:
                best_next_action = random_argmax(agent.get_q(current_player, next_state))
                td_target = reward + agent.gamma * agent.get_q(current_player, next_state, best_next_action)
                td_error = td_target - agent.get_q(current_player, state, action)
                new_q = agent.get_q(current_player, state, action) + agent.alpha * td_error
                agent.set_q(current_player, state, action, new_q)
                episode_reward += reward
        
        learning_curve_red.append(episode_reward)
    
    print(f"\n✓ Training complete")
    
    # Save learning curve
    learning_curve_red = np.array(learning_curve_red)
    save_learning_curve(opts.workspace, player='red', data=learning_curve_red.reshape(-1, 1))
    
    # Save agent Q-tables
    np.save(workspace / "best_red_agent", dict(agent.q['red']))
    np.save(workspace / "best_black_agent", dict(agent.q['black']))
    
    print(f"✓ Saved to: {workspace}")
    
    return str(workspace)


def train_tabular_agent_curriculum(
    agent_class,
    agent_name: str,
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> str:
    """
    Train Q-Learning or SARSA with curriculum learning:
    Phase 1: vs Random opponent
    Phase 2: Self-play
    Phase 3: vs Phase 1 checkpoint (frozen)
    Iterations: Continue training vs previous best (N times)
    
    Returns:
        Path to final trained agent workspace
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_workspace = Path(output_dir) / f"{agent_name}_curriculum_{timestamp}"
    base_workspace.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training {agent_name.upper()} - Curriculum Mode")
    print(f"{'='*70}")
    print(f"Output: {base_workspace}")
    print(f"Episodes per phase: {opts.episodes}")
    print(f"Curriculum iterations: {opts.curriculum_iterations}")
    
    # ===== Phase 1: vs Random =====
    print(f"\n--- Phase 1: Training vs Random Opponent ---")
    phase1_workspace = base_workspace / "phase1_vsrandom"
    phase1_workspace.mkdir(exist_ok=True)
    
    opts_phase1 = copy(opts)
    opts_phase1.workspace = str(phase1_workspace)
    opts_phase1.overwrite = True
    
    agent_phase1 = agent_class(env, opts_phase1)
    random_opponent = RandomAgent(env)
    
    for i in range(opts.episodes):
        print(f"\rPhase 1 Episode {i+1}/{opts.episodes}", end="")
        
        env.reset()
        done = False
        
        while not done:
            current_player = env.current_player
            state = env.get_state_key()
            
            if current_player == 1:  # Agent
                action = agent_phase1.select_action(state, current_player)
            else:  # Random
                action = random_opponent.make_move()
            
            _, reward, done, _ = env.make_move(action)
            next_state = env.get_state_key()
            
            if current_player == 1:
                best_next_action = random_argmax(agent_phase1.get_q(current_player, next_state))
                td_target = reward + agent_phase1.gamma * agent_phase1.get_q(current_player, next_state, best_next_action)
                td_error = td_target - agent_phase1.get_q(current_player, state, action)
                new_q = agent_phase1.get_q(current_player, state, action) + agent_phase1.alpha * td_error
                agent_phase1.set_q(current_player, state, action, new_q)
    
    # Save Phase 1 checkpoint
    np.save(phase1_workspace / "best_red_agent", dict(agent_phase1.q['red']))
    np.save(phase1_workspace / "best_black_agent", dict(agent_phase1.q['black']))
    print(f"\n✓ Phase 1 complete")
    
    # ===== Phase 2: Self-Play =====
    print(f"\n--- Phase 2: Training via Self-Play ---")
    phase2_workspace = base_workspace / "phase2_selfplay"
    phase2_workspace.mkdir(exist_ok=True)
    
    opts_phase2 = copy(opts)
    opts_phase2.workspace = str(phase2_workspace)
    opts_phase2.overwrite = True
    
    # Create new agent for phase 2
    agent_phase2 = agent_class(env, opts_phase2)
    
    for i in range(opts.episodes):
        print(f"\rPhase 2 Episode {i+1}/{opts.episodes}", end="")
        agent_phase2.train_step(i)
    
    # Save Phase 2 checkpoint
    np.save(phase2_workspace / "best_red_agent", dict(agent_phase2.q['red']))
    np.save(phase2_workspace / "best_black_agent", dict(agent_phase2.q['black']))
    print(f"\n✓ Phase 2 complete")
    
    # ===== Phase 3: vs Phase 1 Checkpoint =====
    print(f"\n--- Phase 3: Training vs Phase 1 Checkpoint ---")
    phase3_workspace = base_workspace / "phase3_vscheckpoint"
    phase3_workspace.mkdir(exist_ok=True)
    
    opts_phase3 = copy(opts)
    opts_phase3.workspace = str(phase3_workspace)
    opts_phase3.overwrite = True
    
    # Create new agent for phase 3
    agent_phase3 = agent_class(env, opts_phase3)
    
    # Load Phase 1 as frozen opponent
    frozen_opponent = FrozenAgent(env, str(phase1_workspace), player=-1)
    
    for i in range(opts.episodes):
        print(f"\rPhase 3 Episode {i+1}/{opts.episodes}", end="")
        
        env.reset()
        done = False
        
        while not done:
            current_player = env.current_player
            state = env.get_state_key()
            
            if current_player == 1:  # Learning agent
                action = agent_phase3.select_action(state, current_player)
            else:  # Frozen opponent
                action = frozen_opponent.make_move()
            
            _, reward, done, _ = env.make_move(action)
            next_state = env.get_state_key()
            
            if current_player == 1:
                best_next_action = random_argmax(agent_phase3.get_q(current_player, next_state))
                td_target = reward + agent_phase3.gamma * agent_phase3.get_q(current_player, next_state, best_next_action)
                td_error = td_target - agent_phase3.get_q(current_player, state, action)
                new_q = agent_phase3.get_q(current_player, state, action) + agent_phase3.alpha * td_error
                agent_phase3.set_q(current_player, state, action, new_q)
    
    # Save Phase 3 checkpoint  
    np.save(phase3_workspace / "best_red_agent", dict(agent_phase3.q['red']))
    np.save(phase3_workspace / "best_black_agent", dict(agent_phase3.q['black']))
    print(f"\n✓ Phase 3 complete")
    
    # ===== Iterative Curriculum: Train vs Previous Best =====
    previous_workspace = phase3_workspace
    
    for iteration in range(1, opts.curriculum_iterations + 1):
        print(f"\n--- Iteration {iteration}: Training vs Previous Best ---")
        iter_workspace = base_workspace / f"iteration{iteration}"
        iter_workspace.mkdir(exist_ok=True)
        
        opts_iter = copy(opts)
        opts_iter.workspace = str(iter_workspace)
        opts_iter.overwrite = True
        
        # Create new agent
        agent_iter = agent_class(env, opts_iter)
        
        # Load previous best as frozen opponent
        frozen_opponent = FrozenAgent(env, str(previous_workspace), player=-1)
        
        for i in range(opts.episodes):
            print(f"\rIteration {iteration} Episode {i+1}/{opts.episodes}", end="")
            
            env.reset()
            done = False
            
            while not done:
                current_player = env.current_player
                state = env.get_state_key()
                
                if current_player == 1:  # Learning agent
                    action = agent_iter.select_action(state, current_player)
                else:  # Frozen previous best
                    action = frozen_opponent.make_move()
                
                _, reward, done, _ = env.make_move(action)
                next_state = env.get_state_key()
                
                if current_player == 1:
                    best_next_action = random_argmax(agent_iter.get_q(current_player, next_state))
                    td_target = reward + agent_iter.gamma * agent_iter.get_q(current_player, next_state, best_next_action)
                    td_error = td_target - agent_iter.get_q(current_player, state, action)
                    new_q = agent_iter.get_q(current_player, state, action) + agent_iter.alpha * td_error
                    agent_iter.set_q(current_player, state, action, new_q)
        
        # Save iteration checkpoint
        np.save(iter_workspace / "best_red_agent", dict(agent_iter.q['red']))
        np.save(iter_workspace / "best_black_agent", dict(agent_iter.q['black']))
        print(f"\n✓ Iteration {iteration} complete")
        
        # Update previous workspace for next iteration
        previous_workspace = iter_workspace
    
    print(f"\n✓ Full curriculum training complete ({3 + opts.curriculum_iterations} phases total)")
    print(f"✓ Final checkpoint: {previous_workspace}")
    
    # Return final iteration workspace
    return str(previous_workspace)


def train_monte_carlo_curriculum(
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> str:
    """
    Train Monte Carlo with curriculum - EXACTLY like Q-Learning/SARSA.
    Each phase gets its own workspace subdirectory.
    
    Returns:
        Path to final trained agent workspace
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_workspace = Path(output_dir) / f"monte-carlo_curriculum_{timestamp}"
    base_workspace.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training MONTE CARLO - Curriculum Mode")
    print(f"{'='*70}")
    print(f"Output: {base_workspace}")
    print(f"Episodes per phase: {opts.episodes}")
    print(f"Curriculum iterations: {opts.curriculum_iterations}")
    
    # ===== Phase 1: vs Random =====
    print(f"\n--- Phase 1: Training vs Random Opponent ---")
    phase1_workspace = base_workspace / "phase1_vsrandom"
    phase1_workspace.mkdir(exist_ok=True)
    
    opts_phase1 = copy(opts)
    opts_phase1.workspace = str(phase1_workspace)
    opts_phase1.checkpoint_dir = str(phase1_workspace)
    opts_phase1.overwrite = True
    
    agent_phase1 = MonteCarlo(env, opts_phase1)
    opponent_random = RandomAgent(env)
    agent_phase1.train(opponent=opponent_random, phase_name="Phase1-Random")
    
    # Save in workspace format
    np.save(phase1_workspace / "best_red_agent", dict(agent_phase1.q['red']))
    np.save(phase1_workspace / "best_black_agent", dict(agent_phase1.q['black']))
    print(f"✓ Phase 1 complete")
    
    # ===== Phase 2: Self-Play =====
    print(f"\n--- Phase 2: Training via Self-Play ---")
    phase2_workspace = base_workspace / "phase2_selfplay"
    phase2_workspace.mkdir(exist_ok=True)
    
    opts_phase2 = copy(opts)
    opts_phase2.workspace = str(phase2_workspace)
    opts_phase2.checkpoint_dir = str(phase2_workspace)
    opts_phase2.overwrite = True
    
    agent_phase2 = MonteCarlo(env, opts_phase2)
    agent_phase2.train(opponent=None, phase_name="Phase2-SelfPlay")
    
    # Save in workspace format
    np.save(phase2_workspace / "best_red_agent", dict(agent_phase2.q['red']))
    np.save(phase2_workspace / "best_black_agent", dict(agent_phase2.q['black']))
    print(f"✓ Phase 2 complete")
    
    # ===== Phase 3: vs Phase 1 Checkpoint =====
    print(f"\n--- Phase 3: Training vs Phase 1 Checkpoint ---")
    phase3_workspace = base_workspace / "phase3_vscheckpoint"
    phase3_workspace.mkdir(exist_ok=True)
    
    opts_phase3 = copy(opts)
    opts_phase3.workspace = str(phase3_workspace)
    opts_phase3.checkpoint_dir = str(phase3_workspace)
    opts_phase3.overwrite = True
    
    agent_phase3 = MonteCarlo(env, opts_phase3)
    
    # Load Phase 1 as frozen opponent
    frozen_opponent = FrozenAgent(env, str(phase1_workspace), player=-1)
    agent_phase3.train(opponent=frozen_opponent, phase_name="Phase3-VsCheckpoint")
    
    # Save in workspace format
    np.save(phase3_workspace / "best_red_agent", dict(agent_phase3.q['red']))
    np.save(phase3_workspace / "best_black_agent", dict(agent_phase3.q['black']))
    print(f"✓ Phase 3 complete")
    
    # ===== Iterative Curriculum: Train vs Previous Best =====
    previous_workspace = phase3_workspace
    
    for iteration in range(1, opts.curriculum_iterations + 1):
        print(f"\n--- Iteration {iteration}: Training vs Previous Best ---")
        iter_workspace = base_workspace / f"iteration{iteration}"
        iter_workspace.mkdir(exist_ok=True)
        
        opts_iter = copy(opts)
        opts_iter.workspace = str(iter_workspace)
        opts_iter.checkpoint_dir = str(iter_workspace)
        opts_iter.overwrite = True
        
        agent_iter = MonteCarlo(env, opts_iter)
        
        # Load previous best as frozen opponent
        frozen_opponent = FrozenAgent(env, str(previous_workspace), player=-1)
        agent_iter.train(opponent=frozen_opponent, phase_name=f"Iteration{iteration}")
        
        # Save in workspace format
        np.save(iter_workspace / "best_red_agent", dict(agent_iter.q['red']))
        np.save(iter_workspace / "best_black_agent", dict(agent_iter.q['black']))
        print(f"✓ Iteration {iteration} complete")
        
        # Update for next iteration
        previous_workspace = iter_workspace
    
    print(f"\n✓ Full curriculum training complete ({3 + opts.curriculum_iterations} phases total)")
    print(f"✓ Final checkpoint: {previous_workspace}")
    
    # Return final workspace
    return str(previous_workspace)


def train_agent(
    algorithm: str,
    training_mode: str,
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> str:
    """
    Train a single agent with specified algorithm and mode.
    
    Returns:
        Path to trained agent workspace/checkpoint
    """
    if algorithm == "q-learning":
        if training_mode == "self-play":
            return train_tabular_agent_selfplay(QLearning, "q-learning", env, opts, output_dir)
        elif training_mode == "vs-random":
            return train_tabular_agent_vs_random(QLearning, "q-learning", env, opts, output_dir)
        elif training_mode == "curriculum":
            return train_tabular_agent_curriculum(QLearning, "q-learning", env, opts, output_dir)
    
    elif algorithm == "sarsa":
        if training_mode == "self-play":
            return train_tabular_agent_selfplay(Sarsa, "sarsa", env, opts, output_dir)
        elif training_mode == "vs-random":
            return train_tabular_agent_vs_random(Sarsa, "sarsa", env, opts, output_dir)
        elif training_mode == "curriculum":
            return train_tabular_agent_curriculum(Sarsa, "sarsa", env, opts, output_dir)
    
    elif algorithm == "expected-sarsa":
        if training_mode == "self-play":
            return train_tabular_agent_selfplay(ExpectedSarsa, "expected-sarsa", env, opts, output_dir)
        elif training_mode == "vs-random":
            return train_tabular_agent_vs_random(ExpectedSarsa, "expected-sarsa", env, opts, output_dir)
        elif training_mode == "curriculum":
            return train_tabular_agent_curriculum(ExpectedSarsa, "expected-sarsa", env, opts, output_dir)
    
    elif algorithm == "monte-carlo":
        # Monte Carlo only supports curriculum mode properly
        return train_monte_carlo_curriculum(env, opts, output_dir)
    
    elif algorithm == "dqn":
        print(f"ERROR: DQN not yet implemented")
        sys.exit(1)
    
    else:
        print(f"ERROR: Unknown algorithm: {algorithm}")
        sys.exit(1)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def run_pairwise_evaluations(
    agent_paths: Dict[str, str],
    env: ConnectFourEnv,
    opts: Namespace,
    output_dir: str
) -> Dict:
    """
    Run all pairwise evaluations between trained agents.
    
    Returns:
        Dictionary of evaluation results
    """
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
    """
    Run round-robin tournament between all agents.
    
    Returns:
        Tournament results dictionary
    """
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


def generate_summary_report(
    training_results: Dict[str, str],
    pairwise_results: Dict,
    tournament_results: Dict,
    output_dir: str
):
    """Generate a summary report of all training and evaluation."""
    report_path = Path(output_dir) / "SUMMARY_REPORT.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CONNECT FOUR RL TRAINING & EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Training summary
        f.write("TRAINED AGENTS:\n")
        f.write("-"*70 + "\n")
        for agent_name, path in training_results.items():
            f.write(f"  {agent_name:20s}: {path}\n")
        f.write("\n")
        
        # Pairwise results summary
        f.write("PAIRWISE MATCHUP RESULTS:\n")
        f.write("-"*70 + "\n")
        for matchup_name, data in pairwise_results.items():
            matchup = data['matchup']
            f.write(f"\n  {matchup_name}:\n")
            f.write(f"    {matchup.red_agent} wins: {matchup.red_wins} ({matchup.red_win_rate:.1%})\n")
            f.write(f"    {matchup.black_agent} wins: {matchup.black_wins} ({matchup.black_win_rate:.1%})\n")
            f.write(f"    Ties: {matchup.ties} ({matchup.tie_rate:.1%})\n")
        f.write("\n")
        
        # Tournament summary
        f.write("TOURNAMENT RANKINGS:\n")
        f.write("-"*70 + "\n")
        rankings = tournament_results['rankings']
        for agent, info in sorted(rankings.items(), key=lambda x: x[1]['rank']):
            f.write(f"  {info['rank']}. {agent:20s} - {info['total_wins']} total wins\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ Summary report saved to: {report_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = ArgumentParser(
        description="Master training and evaluation pipeline for Connect Four RL agents"
    )
    
    # =================================================================
    # WHAT TO RUN
    # =================================================================
    parser.add_argument(
        '--agents',
        nargs='+',
        choices=['q-learning', 'sarsa', 'expected-sarsa', 'monte-carlo', 'dqn', 'all'],
        default=['all'],
        help='Which agents to train (default: all)'
    )
    
    parser.add_argument(
        '--training-mode',
        choices=['self-play', 'vs-random', 'curriculum'],
        default='curriculum',
        help='Training mode (default: curriculum)'
    )
    
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
    
    # =================================================================
    # ENVIRONMENT CONFIGURATION
    # =================================================================
    parser.add_argument(
        '-r', '--rows',
        type=int,
        default=6,
        help='Number of rows in board (default: 6)'
    )
    
    parser.add_argument(
        '-c', '--columns',
        type=int,
        default=7,
        help='Number of columns in board (default: 7)'
    )
    
    parser.add_argument(
        '-n', '--connect-n',
        type=int,
        default=4,
        help='Number in a row to win (default: 4)'
    )
    
    parser.add_argument(
        '--reward',
        type=float,
        default=1.0,
        help='Win reward (default: 1.0)'
    )
    
    parser.add_argument(
        '--penalty',
        type=float,
        default=0.0,
        help='Loss/draw penalty (default: 0.0)'
    )
    
    parser.add_argument(
        '--move-cost',
        type=float,
        default=0.0,
        help='Cost per move (default: 0.0)'
    )
    
    # =================================================================
    # TRAINING HYPERPARAMETERS
    # =================================================================
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9,
        help='Discount factor (default: 0.9)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.1,
        help='Exploration rate (default: 0.1)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=5000,
        help='Training episodes (default: 5000)'
    )
    
    parser.add_argument(
        '--initial-val',
        type=float,
        default=0.0,
        help='Initial Q-value (default: 0.0)'
    )
    
    parser.add_argument(
        '--num-agents',
        type=int,
        default=5,
        help='Ensemble size (default: 5)'
    )
    
    parser.add_argument(
        '--curriculum-iterations',
        type=int,
        default=3,
        help='Number of additional curriculum iterations after base phases (default: 3)'
    )
    
    # =================================================================
    # EVALUATION PARAMETERS
    # =================================================================
    parser.add_argument(
        '--eval-games',
        type=int,
        default=100,
        help='Games per evaluation matchup (default: 100)'
    )
    
    # =================================================================
    # OUTPUT CONFIGURATION
    # =================================================================
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='results',
        help='Base output directory (default: results)'
    )
    
    opts = parser.parse_args()
    
    # Expand 'all' agents
    if 'all' in opts.agents:
        opts.agents = ['q-learning', 'sarsa', 'expected-sarsa', 'monte-carlo']
    
    # Create output directories
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    training_dir = output_dir / "training"
    training_dir.mkdir(exist_ok=True)
    eval_dir = output_dir / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    
    # Create environment
    env = ConnectFourEnv(
        rows=opts.rows,
        cols=opts.columns,
        connect_n=opts.connect_n,
        reward=opts.reward,
        penalty=opts.penalty,
        move_cost=opts.move_cost
    )
    
    print("="*70)
    print("CONNECT FOUR RL MASTER PIPELINE")
    print("="*70)
    print(f"Environment: {opts.rows}x{opts.columns} Connect-{opts.connect_n}")
    print(f"Agents: {', '.join(opts.agents)}")
    print(f"Training mode: {opts.training_mode}")
    print(f"Episodes: {opts.episodes}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # =================================================================
    # TRAINING PHASE
    # =================================================================
    agent_paths = {}
    training_times = {}  # Track training time per agent
    
    if not opts.skip_training:
        print(f"\n{'='*70}")
        print("PHASE 1: TRAINING")
        print(f"{'='*70}")
        
        for agent in opts.agents:
            try:
                print(f"\nStarting training: {agent}")
                with TrainingTimer() as timer:
                    workspace = train_agent(
                        algorithm=agent,
                        training_mode=opts.training_mode,
                        env=env,
                        opts=opts,
                        output_dir=str(training_dir)
                    )
                agent_paths[agent] = workspace
                training_times[agent] = timer.elapsed
                print(f"✓ {agent} training completed in {timer.elapsed:.2f} seconds ({timer.elapsed/60:.1f} minutes)")
            except Exception as e:
                print(f"\nERROR training {agent}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not agent_paths:
            print("\nERROR: No agents were successfully trained")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        for agent_name, path in agent_paths.items():
            train_time = training_times.get(agent_name, 0)
            print(f"  {agent_name}: {path}")
            print(f"    Training time: {train_time:.2f}s ({train_time/60:.1f} min)")
    
    else:
        # Use pre-trained agents
        if not opts.agent_dirs:
            print("ERROR: --agent-dirs required when using --skip-training")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("USING PRE-TRAINED AGENTS")
        print(f"{'='*70}")
        
        for agent_dir in opts.agent_dirs:
            agent_name = Path(agent_dir).name
            # Try to extract algorithm from directory name
            for algo in ['q-learning', 'sarsa', 'expected-sarsa', 'monte-carlo']:
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
            
            try:
                # Pairwise evaluations
                pairwise_results = run_pairwise_evaluations(
                    agent_paths=agent_paths,
                    env=env,
                    opts=opts,
                    output_dir=str(eval_dir)
                )
                
                # Tournament
                tournament_results = run_tournament(
                    agent_paths=agent_paths,
                    env=env,
                    opts=opts,
                    output_dir=str(eval_dir)
                )
                
                # Generate summary
                generate_summary_report(
                    training_results=agent_paths,
                    pairwise_results=pairwise_results,
                    tournament_results=tournament_results,
                    output_dir=str(output_dir)
                )
                
                print(f"\n{'='*70}")
                print("EVALUATION COMPLETE")
                print(f"{'='*70}")
            
            except Exception as e:
                print(f"\nERROR during evaluation: {e}")
                import traceback
                traceback.print_exc()
    
    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults directory: {output_dir}")
    print(f"  Training outputs: {training_dir}")
    print(f"  Evaluation outputs: {eval_dir}")
    print(f"  Summary report: {output_dir / 'SUMMARY_REPORT.txt'}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
