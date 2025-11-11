"""
Advanced Metrics Module

Extends the base metrics.py with training-specific metrics:
- Average moves to win (by outcome type)
- Q-value statistics (mean, variance, sparsity)
- Training time tracking
- Memory usage analysis
- Move quality scoring
- Blunder rate calculation

These metrics complement the statistical metrics in metrics.py
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from data_structures import GameResult, MatchupResult
from game_analyzer import GameAnalysis


@dataclass
class OutcomeMetrics:
    """Metrics broken down by game outcome."""
    wins_avg_moves: float
    wins_std_moves: float
    losses_avg_moves: float
    losses_std_moves: float
    ties_avg_moves: float
    ties_std_moves: float
    total_games: int


@dataclass
class QValueStatistics:
    """Statistics about Q-table contents."""
    mean_q: float
    std_q: float
    min_q: float
    max_q: float
    sparsity: float  # Fraction of (state, action) pairs that are zero
    num_states: int
    num_nonzero_states: int
    total_state_action_pairs: int
    nonzero_state_action_pairs: int


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    algorithm_name: str
    total_time_seconds: float
    episodes: int
    time_per_episode: float
    final_q_stats: QValueStatistics
    curriculum_phase: Optional[str] = None


@dataclass
class MoveQualityScore:
    """Aggregated move quality metrics."""
    overall_score: float  # 0-100 scale
    winning_move_score: float  # 0-100
    blocking_score: float  # 0-100
    blunder_rate: float  # Fraction of moves that were blunders
    explanation: str


class AdvancedMetricsAnalyzer:
    """
    Computes advanced metrics for training and gameplay analysis.
    """
    
    @staticmethod
    def compute_outcome_metrics(
        games: List[GameResult],
        perspective_agent: str
    ) -> OutcomeMetrics:
        """
        Calculate average moves to win/loss/tie from one agent's perspective.
        
        Args:
            games: List of GameResult objects
            perspective_agent: Name of agent to analyze ('red' or 'black')
            
        Returns:
            OutcomeMetrics with averages and standard deviations
        """
        wins = []
        losses = []
        ties = []
        
        for game in games:
            # Determine if this was win/loss/tie from perspective agent's view
            if game.winner == 'tie':
                ties.append(game.num_moves)
            elif (perspective_agent == 'red' and game.winner == 'red') or \
                 (perspective_agent == 'black' and game.winner == 'black'):
                wins.append(game.num_moves)
            else:
                losses.append(game.num_moves)
        
        return OutcomeMetrics(
            wins_avg_moves=float(np.mean(wins)) if wins else 0.0,
            wins_std_moves=float(np.std(wins)) if wins else 0.0,
            losses_avg_moves=float(np.mean(losses)) if losses else 0.0,
            losses_std_moves=float(np.std(losses)) if losses else 0.0,
            ties_avg_moves=float(np.mean(ties)) if ties else 0.0,
            ties_std_moves=float(np.std(ties)) if ties else 0.0,
            total_games=len(games)
        )
    
    @staticmethod
    def analyze_q_table(q_table: Dict, num_actions: int) -> QValueStatistics:
        """
        Compute statistics about Q-table contents.
        
        Args:
            q_table: Dictionary mapping states to Q-value arrays
            num_actions: Number of possible actions
            
        Returns:
            QValueStatistics object
        """
        all_q_values = []
        num_states = len(q_table)
        num_nonzero_states = 0
        total_pairs = 0
        nonzero_pairs = 0
        
        for state, q_values in q_table.items():
            state_has_nonzero = False
            for q_val in q_values:
                all_q_values.append(q_val)
                total_pairs += 1
                if abs(q_val) > 1e-10:
                    nonzero_pairs += 1
                    state_has_nonzero = True
            
            if state_has_nonzero:
                num_nonzero_states += 1
        
        q_array = np.array(all_q_values)
        
        return QValueStatistics(
            mean_q=float(np.mean(q_array)),
            std_q=float(np.std(q_array)),
            min_q=float(np.min(q_array)),
            max_q=float(np.max(q_array)),
            sparsity=1.0 - (nonzero_pairs / total_pairs if total_pairs > 0 else 0),
            num_states=num_states,
            num_nonzero_states=num_nonzero_states,
            total_state_action_pairs=total_pairs,
            nonzero_state_action_pairs=nonzero_pairs
        )
    
    @staticmethod
    def compute_move_quality_score(game_analyses: List[GameAnalysis], player: str) -> MoveQualityScore:
        """
        Compute overall move quality score (0-100) for a player.
        
        Scoring breakdown:
        - 50 points: Winning move accuracy (taking wins when available)
        - 50 points: Blocking accuracy (blocking opponent wins)
        
        Args:
            game_analyses: List of GameAnalysis objects
            player: 'red' or 'black'
            
        Returns:
            MoveQualityScore object
        """
        total_winning_opportunities = 0
        total_winning_taken = 0
        total_block_opportunities = 0
        total_blocks_made = 0
        total_blunders = 0  # Missed wins or missed blocks
        total_moves = 0
        
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
                total_blunders += (
                    analysis.red_winning_moves_missed + 
                    analysis.red_blocks_missed
                )
                # Count red's moves
                for move_analysis in analysis.move_analyses:
                    if move_analysis.player == 'red':
                        total_moves += 1
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
                total_blunders += (
                    analysis.black_winning_moves_missed + 
                    analysis.black_blocks_missed
                )
                # Count black's moves
                for move_analysis in analysis.move_analyses:
                    if move_analysis.player == 'black':
                        total_moves += 1
        
        # Calculate component scores
        winning_accuracy = (
            total_winning_taken / total_winning_opportunities 
            if total_winning_opportunities > 0 else 1.0
        )
        
        blocking_accuracy = (
            total_blocks_made / total_block_opportunities 
            if total_block_opportunities > 0 else 1.0
        )
        
        blunder_rate = total_blunders / total_moves if total_moves > 0 else 0.0
        
        # Overall score (weighted)
        winning_score = winning_accuracy * 50
        blocking_score = blocking_accuracy * 50
        overall_score = winning_score + blocking_score
        
        # Generate explanation
        explanation = f"Won {total_winning_taken}/{total_winning_opportunities} opportunities, "
        explanation += f"Blocked {total_blocks_made}/{total_block_opportunities} threats, "
        explanation += f"{total_blunders} blunders in {total_moves} moves"
        
        return MoveQualityScore(
            overall_score=overall_score,
            winning_move_score=winning_score,
            blocking_score=blocking_score,
            blunder_rate=blunder_rate,
            explanation=explanation
        )
    
    @staticmethod
    def estimate_memory_usage(q_table: Dict, num_actions: int) -> Dict[str, float]:
        """
        Estimate memory usage of Q-table.
        
        Args:
            q_table: Q-table dictionary
            num_actions: Number of actions
            
        Returns:
            Dictionary with memory estimates in MB
        """
        import sys
        
        # Estimate sizes
        num_states = len(q_table)
        
        # Each state key is a string (estimate ~50 bytes)
        # Each Q-value array is num_actions * 8 bytes (float64)
        # Dictionary overhead ~200 bytes per entry
        
        state_key_bytes = num_states * 50
        q_value_bytes = num_states * num_actions * 8
        dict_overhead = num_states * 200
        
        total_bytes = state_key_bytes + q_value_bytes + dict_overhead
        total_mb = total_bytes / (1024 * 1024)
        
        return {
            'state_keys_mb': state_key_bytes / (1024 * 1024),
            'q_values_mb': q_value_bytes / (1024 * 1024),
            'overhead_mb': dict_overhead / (1024 * 1024),
            'total_mb': total_mb,
            'num_states': num_states
        }


class TrainingTimer:
    """
    Context manager for timing training phases.
    
    Usage:
        with TrainingTimer() as timer:
            # training code
            pass
        print(f"Training took {timer.elapsed} seconds")
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return False


def generate_outcome_report(
    matchup: MatchupResult,
    game_analyses: List[GameAnalysis]
) -> str:
    """
    Generate report showing average moves by outcome type.
    
    Args:
        matchup: MatchupResult object
        game_analyses: List of GameAnalysis objects
        
    Returns:
        Formatted report string
    """
    analyzer = AdvancedMetricsAnalyzer()
    
    red_metrics = analyzer.compute_outcome_metrics(matchup.games, 'red')
    black_metrics = analyzer.compute_outcome_metrics(matchup.games, 'black')
    
    report = f"""
{'='*70}
OUTCOME-SPECIFIC METRICS
{'='*70}

{matchup.red_agent} (Red) - Average Moves:
  Wins:   {red_metrics.wins_avg_moves:5.1f} ± {red_metrics.wins_std_moves:4.1f} moves
  Losses: {red_metrics.losses_avg_moves:5.1f} ± {red_metrics.losses_std_moves:4.1f} moves
  Ties:   {red_metrics.ties_avg_moves:5.1f} ± {red_metrics.ties_std_moves:4.1f} moves

{matchup.black_agent} (Black) - Average Moves:
  Wins:   {black_metrics.wins_avg_moves:5.1f} ± {black_metrics.wins_std_moves:4.1f} moves
  Losses: {black_metrics.losses_avg_moves:5.1f} ± {black_metrics.losses_std_moves:4.1f} moves
  Ties:   {black_metrics.ties_avg_moves:5.1f} ± {black_metrics.ties_std_moves:4.1f} moves

Interpretation:
  - Lower "wins" = More efficient victories
  - Higher "losses" = Prolonged losing games (opponent took longer to win)
  - Consistent tie length suggests deterministic play patterns
{'='*70}
"""
    return report


def generate_quality_score_report(
    game_analyses: List[GameAnalysis],
    red_agent: str,
    black_agent: str
) -> str:
    """
    Generate report with move quality scores.
    
    Args:
        game_analyses: List of GameAnalysis objects
        red_agent: Name of red agent
        black_agent: Name of black agent
        
    Returns:
        Formatted report string
    """
    analyzer = AdvancedMetricsAnalyzer()
    
    red_quality = analyzer.compute_move_quality_score(game_analyses, 'red')
    black_quality = analyzer.compute_move_quality_score(game_analyses, 'black')
    
    report = f"""
{'='*70}
MOVE QUALITY SCORES (0-100 scale)
{'='*70}

{red_agent} (Red):
  Overall Score:       {red_quality.overall_score:5.1f} / 100
  Winning Move Score:  {red_quality.winning_move_score:5.1f} / 50
  Blocking Score:      {red_quality.blocking_score:5.1f} / 50
  Blunder Rate:        {red_quality.blunder_rate:5.1%}
  Details: {red_quality.explanation}

{black_agent} (Black):
  Overall Score:       {black_quality.overall_score:5.1f} / 100
  Winning Move Score:  {black_quality.winning_move_score:5.1f} / 50
  Blocking Score:      {black_quality.blocking_score:5.1f} / 50
  Blunder Rate:        {black_quality.blunder_rate:5.1%}
  Details: {black_quality.explanation}

Score Interpretation:
  90-100: Excellent tactical play
  75-89:  Good tactical awareness
  60-74:  Moderate tactical skill
  <60:    Needs improvement
{'='*70}
"""
    return report


def generate_q_table_report(
    q_tables: Dict[str, Dict],
    num_actions: int,
    agent_name: str
) -> str:
    """
    Generate report about Q-table statistics.
    
    Args:
        q_tables: Dict with 'red' and 'black' Q-tables
        num_actions: Number of actions
        agent_name: Name of agent
        
    Returns:
        Formatted report string
    """
    analyzer = AdvancedMetricsAnalyzer()
    
    red_stats = analyzer.analyze_q_table(q_tables['red'], num_actions)
    black_stats = analyzer.analyze_q_table(q_tables['black'], num_actions)
    
    red_mem = analyzer.estimate_memory_usage(q_tables['red'], num_actions)
    black_mem = analyzer.estimate_memory_usage(q_tables['black'], num_actions)
    
    report = f"""
{'='*70}
Q-TABLE STATISTICS: {agent_name}
{'='*70}

Red Player Q-Table:
  States Visited:      {red_stats.num_states:,}
  States with Updates: {red_stats.num_nonzero_states:,} ({100*red_stats.num_nonzero_states/red_stats.num_states:.1f}%)
  Mean Q-value:        {red_stats.mean_q:8.4f}
  Std Q-value:         {red_stats.std_q:8.4f}
  Q-value Range:       [{red_stats.min_q:7.4f}, {red_stats.max_q:7.4f}]
  Sparsity:            {red_stats.sparsity:.1%} (fraction of zero entries)
  Memory Usage:        {red_mem['total_mb']:.2f} MB

Black Player Q-Table:
  States Visited:      {black_stats.num_states:,}
  States with Updates: {black_stats.num_nonzero_states:,} ({100*black_stats.num_nonzero_states/black_stats.num_states:.1f}%)
  Mean Q-value:        {black_stats.mean_q:8.4f}
  Std Q-value:         {black_stats.std_q:8.4f}
  Q-value Range:       [{black_stats.min_q:7.4f}, {black_stats.max_q:7.4f}]
  Sparsity:            {black_stats.sparsity:.1%} (fraction of zero entries)
  Memory Usage:        {black_mem['total_mb']:.2f} MB

Total Memory:          {red_mem['total_mb'] + black_mem['total_mb']:.2f} MB
{'='*70}
"""
    return report
