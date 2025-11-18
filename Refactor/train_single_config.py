"""
Single-Configuration Training Script for RL Agents
==================================================

Trains ONE RL agent with specified configuration.
Designed to be called repeatedly by external loop/wrapper.

Usage:
    python train_single_config.py \
        --rows 3 --cols 4 --connect_n 3 \
        --algorithm q_learning \
        --alpha 0.1 --gamma 0.9 --epsilon 0.1 \
        --episodes 50000 \
        --output_dir results \
        --run_name "3x4_q_learning_alpha0.1"

Output Structure:
    {output_dir}/{run_name}/
        config.json              # Full configuration
        metrics.csv              # Episode-by-episode metrics
        checkpoints/             # Periodic Q-table saves
            checkpoint_10000.pkl
            checkpoint_20000.pkl
            ...
        final_model.pkl          # Final trained Q-table
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import RL components
from connect_four_env import ConnectFourEnv
from q_learning import QLearning
from monte_carlo import MonteCarlo
from sarsa import SARSA
from argparse import Namespace


class SingleConfigTrainer:
    """Manages training for a single configuration."""
    
    def __init__(self, args):
        """
        Initialize trainer with parsed arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        
        # Create output directory structure
        self.output_dir = os.path.join(args.output_dir, args.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        # Initialize environment
        self.env = ConnectFourEnv(
            rows=args.rows,
            cols=args.cols,
            connect_n=args.connect_n
        )
        
        # Save configuration
        self.save_config()
        
        # Initialize agent
        self.agent = self.create_agent()
        
        # Metrics storage
        self.metrics = []
        
        # Baseline agent for evaluation (if requested)
        self.baseline_agent = None
        if args.baseline_episodes > 0:
            print(f"Training baseline agent ({args.baseline_episodes} episodes)...")
            self.train_baseline()
    
    def save_config(self):
        """Save full configuration to JSON."""
        config = {
            'run_name': self.args.run_name,
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'rows': self.args.rows,
                'cols': self.args.cols,
                'connect_n': self.args.connect_n
            },
            'algorithm': self.args.algorithm,
            'hyperparameters': {
                'alpha': self.args.alpha,
                'gamma': self.args.gamma,
                'epsilon_start': self.args.epsilon_start,
                'epsilon_end': self.args.epsilon_end,
                'epsilon_decay': self.args.epsilon_decay,
                'initial_val': self.args.initial_val
            },
            'training': {
                'episodes': self.args.episodes,
                'eval_interval': self.args.eval_interval,
                'checkpoint_interval': self.args.checkpoint_interval,
                'baseline_episodes': self.args.baseline_episodes,
                'eval_games': self.args.eval_games
            }
        }
        
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
    
    def create_agent(self):
        """Create RL agent based on algorithm name."""
        # Agent options
        opts = Namespace(
            gamma=self.args.gamma,
            alpha=self.args.alpha,
            epsilon_start=self.args.epsilon_start,
            epsilon_end=self.args.epsilon_end,
            epsilon_decay=self.args.epsilon_decay,
            initial_val=self.args.initial_val,
            episodes=self.args.episodes,
            workspace=None,  # We handle saving manually
            overwrite=False
        )
        
        # Select algorithm
        algorithm_map = {
            'q_learning': QLearning,
            'monte_carlo': MonteCarlo,
            'sarsa': SARSA
        }
        
        if self.args.algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {self.args.algorithm}. "
                           f"Choose from: {list(algorithm_map.keys())}")
        
        agent_class = algorithm_map[self.args.algorithm]
        agent = agent_class(self.env, opts)
        
        print(f"Created agent: {self.args.algorithm}")
        return agent
    
    def train_baseline(self):
        """Train a baseline Q-Learning agent for comparison."""
        opts = Namespace(
            gamma=0.9,
            alpha=0.1,
            epsilon_start=0.1,
            epsilon_end=0.1,
            epsilon_decay=1.0,
            initial_val=0.0,
            episodes=self.args.baseline_episodes,
            workspace=None,
            overwrite=False
        )
        
        self.baseline_agent = QLearning(self.env, opts)
        
        for ep in tqdm(range(self.args.baseline_episodes), desc="Baseline training"):
            self.baseline_agent.train_step(ep)
        
        print("✓ Baseline agent trained")
    
    def play_games(self, agent, opponent_agent=None, num_games: int = 100,
                   agent_plays_first: bool = True) -> Dict[str, int]:
        """
        Play evaluation games.
        
        Args:
            agent: RL agent to evaluate
            opponent_agent: Opponent (None = random)
            num_games: Number of games
            agent_plays_first: Whether agent goes first
        
        Returns:
            Dict with wins/losses/draws
        """
        results = {"wins": 0, "losses": 0, "draws": 0}
        
        # Disable exploration
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        if opponent_agent:
            original_epsilon_opp = opponent_agent.epsilon
            opponent_agent.epsilon = 0.0
        
        for _ in range(num_games):
            self.env.reset()
            done = False
            
            while not done:
                if agent_plays_first:
                    if self.env.current_player == 1:
                        action = agent.eval_step(self.env)
                    else:
                        if opponent_agent:
                            action = opponent_agent.eval_step(self.env)
                        else:
                            valid = self.env.get_valid_actions()
                            action = np.random.choice(valid) if valid else 0
                else:
                    if self.env.current_player == -1:
                        action = agent.eval_step(self.env)
                    else:
                        if opponent_agent:
                            action = opponent_agent.eval_step(self.env)
                        else:
                            valid = self.env.get_valid_actions()
                            action = np.random.choice(valid) if valid else 0
                
                _, _, done, info = self.env.make_move(action)
            
            winner = info.get("winner")
            agent_player = 1 if agent_plays_first else -1
            
            if winner == agent_player:
                results["wins"] += 1
            elif winner == -agent_player:
                results["losses"] += 1
            else:
                results["draws"] += 1
        
        # Restore epsilon
        agent.epsilon = original_epsilon
        if opponent_agent:
            opponent_agent.epsilon = original_epsilon_opp
        
        return results
    
    def evaluate_agent(self, episode: int) -> Dict:
        """
        Comprehensive evaluation at current training stage.
        
        Args:
            episode: Current episode number
        
        Returns:
            Dict with all metrics
        """
        metrics = {'episode': episode}
        
        # Win rate vs random (first and second)
        results_first = self.play_games(
            self.agent, opponent_agent=None,
            num_games=self.args.eval_games, agent_plays_first=True
        )
        results_second = self.play_games(
            self.agent, opponent_agent=None,
            num_games=self.args.eval_games, agent_plays_first=False
        )
        
        metrics['win_rate_vs_random_first'] = results_first['wins'] / self.args.eval_games
        metrics['win_rate_vs_random_second'] = results_second['wins'] / self.args.eval_games
        metrics['loss_rate_vs_random_first'] = results_first['losses'] / self.args.eval_games
        metrics['loss_rate_vs_random_second'] = results_second['losses'] / self.args.eval_games
        metrics['draw_rate_vs_random_first'] = results_first['draws'] / self.args.eval_games
        metrics['draw_rate_vs_random_second'] = results_second['draws'] / self.args.eval_games
        
        # Win rate vs baseline (if available)
        if self.baseline_agent:
            results_baseline = self.play_games(
                self.agent, opponent_agent=self.baseline_agent,
                num_games=self.args.eval_games, agent_plays_first=True
            )
            metrics['win_rate_vs_baseline'] = results_baseline['wins'] / self.args.eval_games
            metrics['loss_rate_vs_baseline'] = results_baseline['losses'] / self.args.eval_games
            metrics['draw_rate_vs_baseline'] = results_baseline['draws'] / self.args.eval_games
        else:
            metrics['win_rate_vs_baseline'] = 0.0
            metrics['loss_rate_vs_baseline'] = 0.0
            metrics['draw_rate_vs_baseline'] = 0.0
        
        # Q-value at initial state (empty board, center column)
        self.env.reset()
        initial_state = self.agent.get_state_key()
        center_col = self.env.cols // 2
        metrics['q_value_initial_state'] = self.agent.get_q(initial_state, center_col)
        
        # State coverage
        metrics['states_visited'] = len(self.agent.Q)
        
        # Current epsilon value
        metrics['epsilon'] = self.agent.epsilon
        
        return metrics
    
    def save_checkpoint(self, episode: int):
        """Save Q-table checkpoint."""
        checkpoint_path = os.path.join(
            self.output_dir, "checkpoints", f"checkpoint_{episode}.pkl"
        )
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(dict(self.agent.Q), f)
    
    def save_final_model(self):
        """Save final trained model."""
        model_path = os.path.join(self.output_dir, "final_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(dict(self.agent.Q), f)
        print(f"✓ Final model saved: {model_path}")
    
    def save_metrics(self):
        """Save all metrics to CSV."""
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            csv_path = os.path.join(self.output_dir, "metrics.csv")
            df.to_csv(csv_path, index=False)
            print(f"✓ Metrics saved: {csv_path}")
    
    def train(self):
        """Execute full training run."""
        print("\n" + "="*60)
        print(f"TRAINING: {self.args.run_name}")
        print("="*60)
        print(f"Algorithm: {self.args.algorithm}")
        print(f"Board: {self.args.rows}×{self.args.cols}, connect-{self.args.connect_n}")
        print(f"Episodes: {self.args.episodes}")
        print(f"Hyperparameters: α={self.args.alpha}, γ={self.args.gamma}, ε={self.args.epsilon_start}→{self.args.epsilon_end} (decay={self.args.epsilon_decay})")
        print("="*60 + "\n")
        
        # Initial evaluation (episode 0)
        if self.args.eval_at_start:
            print("Evaluating at episode 0...")
            metrics = self.evaluate_agent(episode=0)
            self.metrics.append(metrics)
            print(f"  Initial win rate (vs random, first): {metrics['win_rate_vs_random_first']:.1%}\n")
        
        # Training loop with progress bar
        for episode in tqdm(range(1, self.args.episodes + 1), desc="Training"):
            # Train one episode
            self.agent.train_step(episode)
            
            # Evaluate at intervals
            if episode % self.args.eval_interval == 0:
                tqdm.write(f"\nEvaluating at episode {episode}...")
                metrics = self.evaluate_agent(episode)
                self.metrics.append(metrics)
                
                tqdm.write(f"  Win rate (vs random, first): {metrics['win_rate_vs_random_first']:.1%}")
                tqdm.write(f"  Win rate (vs random, second): {metrics['win_rate_vs_random_second']:.1%}")
                if self.baseline_agent:
                    tqdm.write(f"  Win rate (vs baseline): {metrics['win_rate_vs_baseline']:.1%}")
                tqdm.write(f"  Q-value (initial state): {metrics['q_value_initial_state']:.4f}")
                tqdm.write(f"  States visited: {metrics['states_visited']}\n")
            
            # Save checkpoints
            if episode % self.args.checkpoint_interval == 0:
                tqdm.write(f"  → Saving checkpoint at episode {episode}")
                self.save_checkpoint(episode)
        
        # Save final artifacts
        print("\n" + "="*60)
        print("TRAINING COMPLETE - Saving results...")
        print("="*60)
        self.save_final_model()
        self.save_metrics()
        
        # Final summary
        if self.metrics:
            final = self.metrics[-1]
            print("\nFinal Performance:")
            print(f"  Win rate (vs random, first): {final['win_rate_vs_random_first']:.1%}")
            print(f"  Win rate (vs random, second): {final['win_rate_vs_random_second']:.1%}")
            if self.baseline_agent:
                print(f"  Win rate (vs baseline): {final['win_rate_vs_baseline']:.1%}")
            print(f"  States explored: {final['states_visited']}")
        
        print("\n" + "="*60)
        print(f"All results saved to: {self.output_dir}")
        print("="*60 + "\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train single RL agent with specified configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment configuration
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--rows', type=int, required=True,
                          help='Number of rows in board')
    env_group.add_argument('--cols', type=int, required=True,
                          help='Number of columns in board')
    env_group.add_argument('--connect_n', type=int, required=True,
                          help='Number in a row to win')
    
    # Algorithm selection
    algo_group = parser.add_argument_group('Algorithm')
    algo_group.add_argument('--algorithm', type=str, required=True,
                           choices=['q_learning', 'monte_carlo', 'sarsa'],
                           help='RL algorithm to use')
    
    # Hyperparameters
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--alpha', type=float, default=0.1,
                            help='Learning rate')
    hyper_group.add_argument('--gamma', type=float, default=0.9,
                            help='Discount factor')
    hyper_group.add_argument('--epsilon_start', type=float, default=1.0,
                            help='Initial exploration rate')
    hyper_group.add_argument('--epsilon_end', type=float, default=0.01,
                            help='Final exploration rate')
    hyper_group.add_argument('--epsilon_decay', type=float, default=0.9999,
                            help='Epsilon decay rate per episode')
    hyper_group.add_argument('--initial_val', type=float, default=0.0,
                            help='Initial Q-value')
    
    # Training configuration
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--episodes', type=int, default=50000,
                            help='Total training episodes')
    train_group.add_argument('--eval_interval', type=int, default=1000,
                            help='Episodes between evaluations')
    train_group.add_argument('--checkpoint_interval', type=int, default=10000,
                            help='Episodes between checkpoints')
    train_group.add_argument('--baseline_episodes', type=int, default=0,
                            help='Episodes to train baseline agent (0 = no baseline)')
    train_group.add_argument('--eval_games', type=int, default=100,
                            help='Number of games per evaluation')
    train_group.add_argument('--eval_at_start', action='store_true',
                            help='Evaluate before training starts')
    
    # Output configuration
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_dir', type=str, default='results',
                             help='Base output directory')
    output_group.add_argument('--run_name', type=str, required=True,
                             help='Unique name for this run (creates subdir)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Create trainer and run
        trainer = SingleConfigTrainer(args)
        trainer.train()
        
        # Success
        print("✓ Training completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
