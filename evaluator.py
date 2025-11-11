"""
Comprehensive Evaluation Framework for Connect Four RL Agents

This module provides the main Evaluator class that orchestrates all evaluation
activities including head-to-head matches, tournaments, statistical analysis,
and visualization generation.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from connect_four_env import ConnectFourEnv, PLAYERS
from rl_agent import RLModel
from checkpoints import load_params
from data_structures import GameResult, MatchupResult
from game_analyzer import GameAnalyzer
from advanced_metrics import (
    AdvancedMetricsAnalyzer,
    generate_outcome_report,
    generate_quality_score_report
)


class Evaluator:
    """
    Main evaluation orchestrator for Connect Four RL agents.
    
    Supports:
    - Head-to-head matchups
    - Round-robin tournaments
    - Checkpoint progression analysis
    - Statistical analysis and visualization
    """
    
    def __init__(self, env: ConnectFourEnv, output_dir: str = "evaluation_results"):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for evaluation results
        self.matchup_results: List[MatchupResult] = []
        self.agents: Dict[str, RLModel] = {}
        
    def load_agent(self, workspace: str, agent_name: Optional[str] = None) -> str:
        """
        Load an agent from a workspace directory.
        
        Args:
            workspace: Path to agent workspace
            agent_name: Optional custom name for agent (otherwise uses workspace)
            
        Returns:
            agent_name: The name assigned to this agent
        """
        from evaluate import load_model
        
        if agent_name is None:
            agent_name = Path(workspace).name
        
        agent, _ = load_model(workspace, self.env)
        self.agents[agent_name] = agent
        
        print(f"Loaded agent '{agent_name}' from {workspace}")
        return agent_name
    
    def run_single_game(
        self,
        red_agent: RLModel,
        black_agent: RLModel,
        game_id: int = 0,
        max_moves: int = 10000,
        verbose: bool = False
    ) -> GameResult:
        """
        Run a single game between two agents and collect detailed results.
        
        Args:
            red_agent: Agent playing as red (player 1)
            black_agent: Agent playing as black (player -1)
            game_id: Unique identifier for this game
            max_moves: Maximum moves before declaring tie
            verbose: Print game progress
            
        Returns:
            GameResult with detailed game information
        """
        self.env.reset()
        move_count = 0
        
        while move_count < max_moves:
            # Check if game is already over before asking for next move
            if self.env.game_over:
                break
            
            # Check if board is full (no valid actions)
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                # Board is full - it's a tie
                return GameResult(
                    game_id=game_id,
                    winner='tie',
                    num_moves=move_count,
                    move_history=self.env.move_history.copy(),
                    red_agent=red_agent.name() if hasattr(red_agent, 'name') else 'red',
                    black_agent=black_agent.name() if hasattr(black_agent, 'name') else 'black',
                    final_reward=self.env.penalty
                )
            
            current_player = self.env.current_player
            
            # Select action based on current player
            if PLAYERS[current_player] == 'red':
                action = red_agent.eval_step(self.env)
            else:
                action = black_agent.eval_step(self.env)
            
            # Make move
            _, reward, done, info = self.env.make_move(action)
            
            # Only count valid moves (invalid moves don't increment counter)
            if "error" not in info:
                move_count += 1
            
            if verbose:
                self.env.render()
                print(f"Move {move_count}: {PLAYERS[current_player]} played column {action}")
            
            if done:
                # Determine winner from environment
                if reward == self.env.reward:
                    # Someone won - use the winner from environment
                    winner = PLAYERS[self.env.winner]
                elif reward == self.env.penalty:
                    winner = 'tie'
                else:
                    winner = 'tie'
                
                return GameResult(
                    game_id=game_id,
                    winner=winner,
                    num_moves=move_count,
                    move_history=self.env.move_history.copy(),
                    red_agent=red_agent.name() if hasattr(red_agent, 'name') else 'red',
                    black_agent=black_agent.name() if hasattr(black_agent, 'name') else 'black',
                    final_reward=reward
                )
        
        # Max moves reached - game failed to complete
        print(f"WARNING: Game {game_id} exceeded {max_moves} iterations without completing")
        return GameResult(
            game_id=game_id,
            winner='failed',
            num_moves=move_count,
            move_history=self.env.move_history.copy(),
            red_agent=red_agent.name() if hasattr(red_agent, 'name') else 'red',
            black_agent=black_agent.name() if hasattr(black_agent, 'name') else 'black',
            final_reward=0.0
        )
    
    def evaluate_matchup(
        self,
        red_agent_name: str,
        black_agent_name: str,
        num_games: int = 100,
        verbose: bool = False,
        show_progress: bool = True
    ) -> MatchupResult:
        """
        Evaluate two agents against each other over multiple games.
        
        Args:
            red_agent_name: Name of agent playing red
            black_agent_name: Name of agent playing black
            num_games: Number of games to play
            verbose: Print detailed game information
            show_progress: Show progress bar
            
        Returns:
            MatchupResult with aggregated statistics
        """
        if red_agent_name not in self.agents:
            raise ValueError(f"Agent '{red_agent_name}' not loaded")
        if black_agent_name not in self.agents:
            raise ValueError(f"Agent '{black_agent_name}' not loaded")
        
        red_agent = self.agents[red_agent_name]
        black_agent = self.agents[black_agent_name]
        
        games = []
        red_wins = 0
        black_wins = 0
        ties = 0
        failed = 0
        
        for i in range(num_games):
            if show_progress and not verbose:
                print(f"\rGame {i+1}/{num_games}", end="")
            
            game_result = self.run_single_game(
                red_agent=red_agent,
                black_agent=black_agent,
                game_id=i,
                verbose=False
            )
            
            games.append(game_result)
            
            if game_result.winner == 'red':
                red_wins += 1
            elif game_result.winner == 'black':
                black_wins += 1
            elif game_result.winner == 'tie':
                ties += 1
            elif game_result.winner == 'failed':
                failed += 1
        
        if show_progress and not verbose:
            print()  # New line after progress
        
        result = MatchupResult(
            red_agent=red_agent_name,
            black_agent=black_agent_name,
            num_games=num_games,
            red_wins=red_wins,
            black_wins=black_wins,
            ties=ties,
            games=games
        )
        
        self.matchup_results.append(result)
        
        # Basic results
        print(f"\nMatchup Results: {red_agent_name} (red) vs {black_agent_name} (black)")
        print(f"  Red wins:   {red_wins:4d} ({result.red_win_rate:.1%})")
        print(f"  Black wins: {black_wins:4d} ({result.black_win_rate:.1%})")
        print(f"  Ties:       {ties:4d} ({result.tie_rate:.1%})")
        if failed > 0:
            failed_rate = failed / num_games if num_games > 0 else 0.0
            print(f"  Failed:     {failed:4d} ({failed_rate:.1%}) ⚠️")
        
        # Automatically analyze games and generate advanced reports
        if num_games >= 10:  # Only for reasonably-sized evaluations
            print("\nAnalyzing game quality...")
            game_analyzer = GameAnalyzer(self.env)
            game_analyses = game_analyzer.analyze_multiple_games(games, progress=False)
            
            # Outcome-specific metrics
            print(generate_outcome_report(result, game_analyses))
            
            # Move quality scores
            print(generate_quality_score_report(game_analyses, red_agent_name, black_agent_name))
        
        return result
    
    def run_tournament(
        self,
        agent_names: List[str],
        games_per_matchup: int = 100,
        bidirectional: bool = True
    ) -> Dict[str, Any]:
        """
        Run a round-robin tournament between multiple agents.
        
        Args:
            agent_names: List of agent names to compete
            games_per_matchup: Number of games for each matchup
            bidirectional: If True, play both (A vs B) and (B vs A)
            
        Returns:
            Dictionary with tournament results and rankings
        """
        print(f"\n{'='*60}")
        print(f"Starting Tournament with {len(agent_names)} agents")
        print(f"{'='*60}\n")
        
        tournament_results = {
            'agents': agent_names,
            'matchups': [],
            'rankings': {}
        }
        
        # Run all pairwise matchups
        for i, agent1 in enumerate(agent_names):
            for j, agent2 in enumerate(agent_names):
                if i >= j:  # Skip self-play and duplicates in one direction
                    continue
                
                # Agent1 as red, Agent2 as black
                print(f"\nMatchup {len(tournament_results['matchups'])+1}: {agent1} (red) vs {agent2} (black)")
                result1 = self.evaluate_matchup(
                    red_agent_name=agent1,
                    black_agent_name=agent2,
                    num_games=games_per_matchup,
                    show_progress=True
                )
                tournament_results['matchups'].append(result1.to_dict())
                
                # If bidirectional, also play with colors swapped
                if bidirectional:
                    print(f"\nMatchup {len(tournament_results['matchups'])+1}: {agent2} (red) vs {agent1} (black)")
                    result2 = self.evaluate_matchup(
                        red_agent_name=agent2,
                        black_agent_name=agent1,
                        num_games=games_per_matchup,
                        show_progress=True
                    )
                    tournament_results['matchups'].append(result2.to_dict())
        
        # Calculate rankings (total wins)
        win_counts = {agent: 0 for agent in agent_names}
        for matchup in tournament_results['matchups']:
            win_counts[matchup['red_agent']] += matchup['red_wins']
            win_counts[matchup['black_agent']] += matchup['black_wins']
        
        # Sort by wins
        ranked_agents = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
        tournament_results['rankings'] = {
            agent: {'rank': i+1, 'total_wins': wins}
            for i, (agent, wins) in enumerate(ranked_agents)
        }
        
        print(f"\n{'='*60}")
        print("Tournament Rankings:")
        print(f"{'='*60}")
        for agent, info in tournament_results['rankings'].items():
            print(f"  {info['rank']}. {agent:30s} - {info['total_wins']} wins")
        
        return tournament_results
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save all evaluation results to JSON file.
        
        Args:
            filename: Optional custom filename (default: auto-generated with timestamp)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'env_config': self.env.get_parameters(),
            'num_matchups': len(self.matchup_results),
            'matchups': [m.to_dict() for m in self.matchup_results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def load_results(self, filepath: str):
        """Load previously saved evaluation results."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct matchup results
        self.matchup_results = []
        for matchup_data in data['matchups']:
            games = [
                GameResult(**game_data)
                for game_data in matchup_data['games']
            ]
            matchup = MatchupResult(
                red_agent=matchup_data['red_agent'],
                black_agent=matchup_data['black_agent'],
                num_games=matchup_data['num_games'],
                red_wins=matchup_data['red_wins'],
                black_wins=matchup_data['black_wins'],
                ties=matchup_data['ties'],
                games=games,
                timestamp=matchup_data['timestamp']
            )
            self.matchup_results.append(matchup)
        
        print(f"Loaded {len(self.matchup_results)} matchup results from {filepath}")
