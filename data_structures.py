"""
Shared data structures for evaluation system.

This module contains dataclasses used across multiple modules to avoid circular imports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Dict


@dataclass
class GameResult:
    """Detailed results from a single game."""
    game_id: int
    winner: str  # 'red', 'black', or 'tie'
    num_moves: int
    move_history: List[Tuple[int, int, int]]  # (row, col, player)
    red_agent: str
    black_agent: str
    final_reward: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        # Convert numpy types to Python types for JSON serialization
        return {
            'game_id': int(self.game_id),
            'winner': str(self.winner),
            'num_moves': int(self.num_moves),
            'move_history': [(int(r), int(c), int(p)) for r, c, p in self.move_history],
            'red_agent': str(self.red_agent),
            'black_agent': str(self.black_agent),
            'final_reward': float(self.final_reward),
            'timestamp': str(self.timestamp)
        }


@dataclass
class MatchupResult:
    """Results from multiple games between two agents."""
    red_agent: str
    black_agent: str
    num_games: int
    red_wins: int
    black_wins: int
    ties: int
    games: List[GameResult]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def red_win_rate(self) -> float:
        return self.red_wins / self.num_games if self.num_games > 0 else 0.0
    
    @property
    def black_win_rate(self) -> float:
        return self.black_wins / self.num_games if self.num_games > 0 else 0.0
    
    @property
    def tie_rate(self) -> float:
        return self.ties / self.num_games if self.num_games > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            'red_agent': str(self.red_agent),
            'black_agent': str(self.black_agent),
            'num_games': int(self.num_games),
            'red_wins': int(self.red_wins),
            'black_wins': int(self.black_wins),
            'ties': int(self.ties),
            'red_win_rate': float(self.red_win_rate),
            'black_win_rate': float(self.black_win_rate),
            'tie_rate': float(self.tie_rate),
            'games': [g.to_dict() for g in self.games],
            'timestamp': str(self.timestamp)
        }
