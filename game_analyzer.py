"""
Game Analyzer Module

Analyzes individual games and move sequences to evaluate:
- Move quality (winning moves, blocking, threats)
- Strategic patterns and preferences
- Opening move analysis
- Endgame performance
- Tactical awareness
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass

from connect_four_env import ConnectFourEnv, PLAYERS
from data_structures import GameResult


@dataclass
class MoveAnalysis:
    """Analysis of a single move."""
    move_number: int
    player: str
    column: int
    row: int
    was_winning_move: bool
    missed_win: bool
    blocked_opponent_win: bool
    missed_block: bool
    created_threat: bool
    board_state_before: np.ndarray
    board_state_after: np.ndarray


@dataclass
class GameAnalysis:
    """Comprehensive analysis of a single game."""
    game_result: GameResult
    move_analyses: List[MoveAnalysis]
    red_winning_moves_taken: int
    red_winning_moves_missed: int
    red_blocks_made: int
    red_blocks_missed: int
    black_winning_moves_taken: int
    black_winning_moves_missed: int
    black_blocks_made: int
    black_blocks_missed: int
    opening_column: int
    decisive_move_number: Optional[int]  # Move that determined outcome


class GameAnalyzer:
    """
    Analyzes games to evaluate move quality and strategic patterns.
    """
    
    def __init__(self, env: ConnectFourEnv):
        self.env = env
    
    def check_winning_move(self, board: np.ndarray, player: int) -> Set[int]:
        """
        Find all columns where player could win immediately.
        
        Args:
            board: Current board state
            player: Player to check (1 or -1)
            
        Returns:
            Set of column indices that would result in immediate win
        """
        winning_cols = set()
        
        # Create temporary environment to test moves
        temp_env = ConnectFourEnv(
            rows=self.env.rows,
            cols=self.env.cols,
            connect_n=self.env.connect_n
        )
        
        for col in range(self.env.cols):
            # Check if move is valid
            if board[0, col] != 0:
                continue
            
            # Simulate the move
            temp_env.board = board.copy()
            temp_env.current_player = player
            temp_env.game_over = False
            
            # Find row where piece would land
            row = -1
            for r in range(self.env.rows - 1, -1, -1):
                if board[r, col] == 0:
                    row = r
                    break
            
            if row >= 0:
                # Check if this move wins
                temp_env.board[row, col] = player
                if temp_env._check_win(row, col):
                    winning_cols.add(col)
        
        return winning_cols
    
    def analyze_move(
        self,
        move_number: int,
        player: int,
        column: int,
        board_before: np.ndarray,
        board_after: np.ndarray
    ) -> MoveAnalysis:
        """
        Analyze the quality of a single move.
        
        Args:
            move_number: Move number in game
            player: Player who made move (1 or -1)
            column: Column played
            board_before: Board state before move
            board_after: Board state after move
            
        Returns:
            MoveAnalysis object
        """
        # Find row where piece landed
        row = -1
        for r in range(self.env.rows - 1, -1, -1):
            if board_before[r, column] == 0:
                row = r
                break
        
        # Check if this was a winning move
        was_winning = False
        temp_env = ConnectFourEnv(
            rows=self.env.rows,
            cols=self.env.cols,
            connect_n=self.env.connect_n
        )
        temp_env.board = board_after.copy()
        if row >= 0:
            was_winning = temp_env._check_win(row, column)
        
        # Check for winning moves available
        winning_moves = self.check_winning_move(board_before, player)
        missed_win = len(winning_moves) > 0 and column not in winning_moves
        
        # Check if this blocked opponent win
        opponent = -player
        opponent_winning_moves = self.check_winning_move(board_before, opponent)
        blocked_opponent_win = column in opponent_winning_moves
        missed_block = len(opponent_winning_moves) > 0 and column not in opponent_winning_moves
        
        # Check if move created a threat (would win next turn if not blocked)
        # This is a simplification - real threat detection is more complex
        temp_env.board = board_after.copy()
        future_winning_moves = self.check_winning_move(board_after, player)
        created_threat = len(future_winning_moves) > 0
        
        return MoveAnalysis(
            move_number=move_number,
            player=PLAYERS[player],
            column=column,
            row=row,
            was_winning_move=was_winning,
            missed_win=missed_win,
            blocked_opponent_win=blocked_opponent_win,
            missed_block=missed_block,
            created_threat=created_threat,
            board_state_before=board_before.copy(),
            board_state_after=board_after.copy()
        )
    
    def analyze_game(self, game_result: GameResult) -> GameAnalysis:
        """
        Perform comprehensive analysis of a complete game.
        
        Args:
            game_result: GameResult to analyze
            
        Returns:
            GameAnalysis with detailed move-by-move breakdown
        """
        # Replay game and analyze each move
        self.env.reset()
        move_analyses = []
        
        red_winning_taken = 0
        red_winning_missed = 0
        red_blocks_made = 0
        red_blocks_missed = 0
        black_winning_taken = 0
        black_winning_missed = 0
        black_blocks_made = 0
        black_blocks_missed = 0
        
        decisive_move = None
        
        for i, (row, col, player) in enumerate(game_result.move_history):
            board_before = self.env.board.copy()
            
            # Make the move
            self.env.board[row, col] = player
            board_after = self.env.board.copy()
            
            # Analyze this move
            analysis = self.analyze_move(
                move_number=i + 1,
                player=player,
                column=col,
                board_before=board_before,
                board_after=board_after
            )
            
            move_analyses.append(analysis)
            
            # Track statistics
            if player == 1:  # Red
                if analysis.was_winning_move:
                    red_winning_taken += 1
                    decisive_move = i + 1
                if analysis.missed_win:
                    red_winning_missed += 1
                if analysis.blocked_opponent_win:
                    red_blocks_made += 1
                if analysis.missed_block:
                    red_blocks_missed += 1
            else:  # Black
                if analysis.was_winning_move:
                    black_winning_taken += 1
                    decisive_move = i + 1
                if analysis.missed_win:
                    black_winning_missed += 1
                if analysis.blocked_opponent_win:
                    black_blocks_made += 1
                if analysis.missed_block:
                    black_blocks_missed += 1
        
        # Opening move (first move of game)
        opening_column = game_result.move_history[0][1] if game_result.move_history else -1
        
        return GameAnalysis(
            game_result=game_result,
            move_analyses=move_analyses,
            red_winning_moves_taken=red_winning_taken,
            red_winning_moves_missed=red_winning_missed,
            red_blocks_made=red_blocks_made,
            red_blocks_missed=red_blocks_missed,
            black_winning_moves_taken=black_winning_taken,
            black_winning_moves_missed=black_winning_missed,
            black_blocks_made=black_blocks_made,
            black_blocks_missed=black_blocks_missed,
            opening_column=opening_column,
            decisive_move_number=decisive_move
        )
    
    def analyze_multiple_games(
        self,
        game_results: List[GameResult],
        progress: bool = True
    ) -> List[GameAnalysis]:
        """
        Analyze multiple games.
        
        Args:
            game_results: List of GameResult objects
            progress: Show progress
            
        Returns:
            List of GameAnalysis objects
        """
        analyses = []
        
        for i, game in enumerate(game_results):
            if progress:
                print(f"\rAnalyzing game {i+1}/{len(game_results)}", end="")
            
            analysis = self.analyze_game(game)
            analyses.append(analysis)
        
        if progress:
            print()
        
        return analyses


class StrategicPatternAnalyzer:
    """
    Analyzes strategic patterns across multiple games.
    """
    
    @staticmethod
    def opening_preferences(game_analyses: List[GameAnalysis]) -> Dict[int, int]:
        """
        Analyze opening move preferences.
        
        Args:
            game_analyses: List of GameAnalysis objects
            
        Returns:
            Dictionary mapping column -> frequency
        """
        opening_counts = Counter()
        
        for analysis in game_analyses:
            if analysis.opening_column >= 0:
                opening_counts[analysis.opening_column] += 1
        
        return dict(opening_counts)
    
    @staticmethod
    def tactical_awareness(
        game_analyses: List[GameAnalysis],
        player: str = 'red'
    ) -> Dict[str, float]:
        """
        Compute tactical awareness metrics for a player.
        
        Args:
            game_analyses: List of GameAnalysis objects
            player: 'red' or 'black'
            
        Returns:
            Dictionary with tactical metrics
        """
        total_winning_opportunities = 0
        total_winning_taken = 0
        total_block_opportunities = 0
        total_blocks_made = 0
        
        for analysis in game_analyses:
            if player == 'red':
                total_winning_taken += analysis.red_winning_moves_taken
                total_winning_opportunities += (
                    analysis.red_winning_moves_taken + 
                    analysis.red_winning_moves_missed
                )
                total_blocks_made += analysis.red_blocks_made
                total_block_opportunities += (
                    analysis.red_blocks_made + 
                    analysis.red_blocks_missed
                )
            else:
                total_winning_taken += analysis.black_winning_moves_taken
                total_winning_opportunities += (
                    analysis.black_winning_moves_taken + 
                    analysis.black_winning_moves_missed
                )
                total_blocks_made += analysis.black_blocks_made
                total_block_opportunities += (
                    analysis.black_blocks_made + 
                    analysis.black_blocks_missed
                )
        
        winning_accuracy = (
            total_winning_taken / total_winning_opportunities 
            if total_winning_opportunities > 0 else 0
        )
        
        blocking_accuracy = (
            total_blocks_made / total_block_opportunities 
            if total_block_opportunities > 0 else 0
        )
        
        return {
            'winning_opportunities': total_winning_opportunities,
            'winning_moves_taken': total_winning_taken,
            'winning_accuracy': winning_accuracy,
            'block_opportunities': total_block_opportunities,
            'blocks_made': total_blocks_made,
            'blocking_accuracy': blocking_accuracy
        }
    
    @staticmethod
    def column_usage_distribution(
        game_analyses: List[GameAnalysis],
        player: str = 'red'
    ) -> Dict[int, int]:
        """
        Analyze which columns a player uses most frequently.
        
        Args:
            game_analyses: List of GameAnalysis objects
            player: 'red' or 'black'
            
        Returns:
            Dictionary mapping column -> count
        """
        column_counts = Counter()
        
        for analysis in game_analyses:
            for move_analysis in analysis.move_analyses:
                if move_analysis.player == player:
                    column_counts[move_analysis.column] += 1
        
        return dict(column_counts)
    
    @staticmethod
    def average_decisive_move(game_analyses: List[GameAnalysis]) -> float:
        """
        Calculate average move number where game was decided.
        
        Args:
            game_analyses: List of GameAnalysis objects
            
        Returns:
            Average decisive move number
        """
        decisive_moves = [
            a.decisive_move_number 
            for a in game_analyses 
            if a.decisive_move_number is not None
        ]
        
        if not decisive_moves:
            return 0.0
        
        return np.mean(decisive_moves)


def generate_game_quality_report(
    game_analyses: List[GameAnalysis],
    red_agent: str,
    black_agent: str
) -> str:
    """
    Generate report on game quality and move accuracy.
    
    Args:
        game_analyses: List of GameAnalysis objects
        red_agent: Name of red agent
        black_agent: Name of black agent
        
    Returns:
        Formatted report string
    """
    analyzer = StrategicPatternAnalyzer()
    
    # Tactical awareness
    red_tactics = analyzer.tactical_awareness(game_analyses, 'red')
    black_tactics = analyzer.tactical_awareness(game_analyses, 'black')
    
    # Opening preferences
    openings = analyzer.opening_preferences(game_analyses)
    
    # Average decisive move
    avg_decisive = analyzer.average_decisive_move(game_analyses)
    
    report = f"""
{'='*70}
GAME QUALITY REPORT
{'='*70}

Games Analyzed: {len(game_analyses)}

Tactical Awareness:

{red_agent} (Red):
  Winning Move Accuracy: {red_tactics['winning_accuracy']:.1%} 
    ({red_tactics['winning_moves_taken']}/{red_tactics['winning_opportunities']} opportunities)
  Blocking Accuracy:     {red_tactics['blocking_accuracy']:.1%}
    ({red_tactics['blocks_made']}/{red_tactics['block_opportunities']} opportunities)

{black_agent} (Black):
  Winning Move Accuracy: {black_tactics['winning_accuracy']:.1%}
    ({black_tactics['winning_moves_taken']}/{black_tactics['winning_opportunities']} opportunities)
  Blocking Accuracy:     {black_tactics['blocking_accuracy']:.1%}
    ({black_tactics['blocks_made']}/{black_tactics['block_opportunities']} opportunities)

Opening Move Distribution:
"""
    
    # Sort openings by column
    for col in sorted(openings.keys()):
        freq = openings[col]
        pct = 100 * freq / len(game_analyses)
        report += f"  Column {col}: {freq:3d} games ({pct:5.1f}%)\n"
    
    report += f"\nAverage Decisive Move: {avg_decisive:.1f}\n"
    report += "="*70 + "\n"
    
    return report
