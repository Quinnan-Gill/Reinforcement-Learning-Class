"""
Statistical Metrics Module

Provides rigorous statistical analysis for evaluating RL agent performance:
- Confidence intervals
- Hypothesis testing
- Effect size calculations
- Performance variance analysis
- Statistical significance testing
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from evaluator import GameResult, MatchupResult


@dataclass
class ConfidenceInterval:
    """Confidence interval for a statistic."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    
    def __repr__(self) -> str:
        return f"{self.point_estimate:.3f} [{self.lower_bound:.3f}, {self.upper_bound:.3f}] ({self.confidence_level:.0%} CI)"


@dataclass
class HypothesisTest:
    """Results from a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    null_hypothesis: str
    alternative_hypothesis: str
    
    def __repr__(self) -> str:
        sig_str = "SIGNIFICANT" if self.significant else "NOT SIGNIFICANT"
        return f"{self.test_name}: p={self.p_value:.4f} ({sig_str} at α={self.alpha})"


class StatisticalAnalyzer:
    """
    Performs statistical analysis on evaluation results.
    """
    
    @staticmethod
    def binomial_ci(
        successes: int,
        trials: int,
        confidence: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for a binomial proportion.
        Uses Wilson score interval (better for small samples and extreme proportions).
        
        Args:
            successes: Number of successful outcomes
            trials: Total number of trials
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            ConfidenceInterval object
        """
        if trials == 0:
            return ConfidenceInterval(0, 0, 0, confidence)
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / trials
        
        # Wilson score interval
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) / trials + z**2 / (4 * trials**2))) / denominator
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return ConfidenceInterval(
            point_estimate=p,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence
        )
    
    @staticmethod
    def win_rate_ci(
        matchup: MatchupResult,
        player: str = 'red',
        confidence: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for win rate.
        
        Args:
            matchup: MatchupResult to analyze
            player: 'red' or 'black'
            confidence: Confidence level
            
        Returns:
            ConfidenceInterval for win rate
        """
        if player == 'red':
            wins = matchup.red_wins
        elif player == 'black':
            wins = matchup.black_wins
        else:
            raise ValueError(f"Invalid player: {player}")
        
        return StatisticalAnalyzer.binomial_ci(wins, matchup.num_games, confidence)
    
    @staticmethod
    def compare_win_rates(
        matchup1: MatchupResult,
        matchup2: MatchupResult,
        alpha: float = 0.05
    ) -> HypothesisTest:
        """
        Test if two agents have significantly different win rates.
        Uses two-proportion z-test.
        
        Args:
            matchup1: First matchup (red agent is agent of interest)
            matchup2: Second matchup (red agent is agent of interest)
            alpha: Significance level
            
        Returns:
            HypothesisTest result
        """
        # Extract data
        n1 = matchup1.num_games
        n2 = matchup2.num_games
        x1 = matchup1.red_wins
        x2 = matchup2.red_wins
        
        p1 = x1 / n1 if n1 > 0 else 0
        p2 = x2 / n2 if n2 > 0 else 0
        
        # Pooled proportion
        p_pool = (x1 + x2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        # Z-statistic
        if se == 0:
            z_stat = 0
            p_value = 1.0
        else:
            z_stat = (p1 - p2) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed
        
        return HypothesisTest(
            test_name="Two-Proportion Z-Test",
            statistic=z_stat,
            p_value=p_value,
            significant=(p_value < alpha),
            alpha=alpha,
            null_hypothesis=f"Win rates are equal (p1={p1:.3f} vs p2={p2:.3f})",
            alternative_hypothesis="Win rates differ"
        )
    
    @staticmethod
    def cohens_h(p1: float, p2: float) -> float:
        """
        Calculate Cohen's h effect size for difference in proportions.
        
        Interpretation:
            |h| < 0.2: Small effect
            0.2 ≤ |h| < 0.5: Medium effect
            |h| ≥ 0.5: Large effect
        
        Args:
            p1: Proportion 1
            p2: Proportion 2
            
        Returns:
            Cohen's h effect size
        """
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = 2 * np.arcsin(np.sqrt(p2))
        return phi1 - phi2
    
    @staticmethod
    def effect_size_interpretation(h: float) -> str:
        """Interpret Cohen's h effect size."""
        abs_h = abs(h)
        if abs_h < 0.2:
            return "Small"
        elif abs_h < 0.5:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def game_length_statistics(games: List[GameResult]) -> Dict[str, float]:
        """
        Calculate statistics for game lengths.
        
        Args:
            games: List of GameResult objects
            
        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not games:
            return {
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'q25': 0,
                'q75': 0
            }
        
        lengths = np.array([g.num_moves for g in games])
        
        return {
            'mean': float(np.mean(lengths)),
            'median': float(np.median(lengths)),
            'std': float(np.std(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'q25': float(np.percentile(lengths, 25)),
            'q75': float(np.percentile(lengths, 75))
        }
    
    @staticmethod
    def variance_ratio_test(
        games1: List[GameResult],
        games2: List[GameResult],
        alpha: float = 0.05
    ) -> HypothesisTest:
        """
        Test if two agents have different variance in game length.
        Uses F-test (Levene's test would be more robust for non-normal data).
        
        Args:
            games1: Games for agent 1
            games2: Games for agent 2
            alpha: Significance level
            
        Returns:
            HypothesisTest result
        """
        lengths1 = np.array([g.num_moves for g in games1])
        lengths2 = np.array([g.num_moves for g in games2])
        
        # Levene's test (more robust than F-test)
        statistic, p_value = stats.levene(lengths1, lengths2)
        
        return HypothesisTest(
            test_name="Levene's Test for Variance Equality",
            statistic=statistic,
            p_value=p_value,
            significant=(p_value < alpha),
            alpha=alpha,
            null_hypothesis="Variances are equal",
            alternative_hypothesis="Variances differ"
        )
    
    @staticmethod
    def mann_whitney_test(
        games1: List[GameResult],
        games2: List[GameResult],
        alpha: float = 0.05
    ) -> HypothesisTest:
        """
        Non-parametric test for difference in game length distributions.
        Mann-Whitney U test (doesn't assume normality).
        
        Args:
            games1: Games for agent 1
            games2: Games for agent 2
            alpha: Significance level
            
        Returns:
            HypothesisTest result
        """
        lengths1 = np.array([g.num_moves for g in games1])
        lengths2 = np.array([g.num_moves for g in games2])
        
        statistic, p_value = stats.mannwhitneyu(
            lengths1,
            lengths2,
            alternative='two-sided'
        )
        
        return HypothesisTest(
            test_name="Mann-Whitney U Test",
            statistic=statistic,
            p_value=p_value,
            significant=(p_value < alpha),
            alpha=alpha,
            null_hypothesis="Game length distributions are equal",
            alternative_hypothesis="Game length distributions differ"
        )
    
    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic_func: callable = np.mean,
        confidence: float = 0.95,
        n_bootstrap: int = 10000
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval using bootstrap resampling.
        
        Args:
            data: Data to bootstrap
            statistic_func: Function to compute statistic (default: mean)
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            ConfidenceInterval object
        """
        if len(data) == 0:
            return ConfidenceInterval(0, 0, 0, confidence)
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        # Calculate percentiles
        alpha = 1 - confidence
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        point_estimate = statistic_func(data)
        lower = np.percentile(bootstrap_stats, lower_percentile)
        upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ConfidenceInterval(
            point_estimate=point_estimate,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence
        )
    
    @staticmethod
    def skill_rating_comparison(
        matchup_results: List[MatchupResult],
        agent_name: str
    ) -> Dict[str, float]:
        """
        Compute aggregate skill metrics for an agent across multiple matchups.
        
        Args:
            matchup_results: List of matchups involving this agent
            agent_name: Name of agent to analyze
            
        Returns:
            Dictionary with aggregate statistics
        """
        total_games = 0
        total_wins = 0
        total_losses = 0
        total_ties = 0
        
        for matchup in matchup_results:
            if matchup.red_agent == agent_name:
                total_games += matchup.num_games
                total_wins += matchup.red_wins
                total_losses += matchup.black_wins
                total_ties += matchup.ties
            elif matchup.black_agent == agent_name:
                total_games += matchup.num_games
                total_wins += matchup.black_wins
                total_losses += matchup.red_wins
                total_ties += matchup.ties
        
        if total_games == 0:
            return {
                'total_games': 0,
                'win_rate': 0,
                'loss_rate': 0,
                'tie_rate': 0,
                'win_loss_ratio': 0
            }
        
        win_rate = total_wins / total_games
        loss_rate = total_losses / total_games
        tie_rate = total_ties / total_games
        win_loss_ratio = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_games': total_games,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'total_ties': total_ties,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'tie_rate': tie_rate,
            'win_loss_ratio': win_loss_ratio
        }


def generate_statistical_report(matchup: MatchupResult, confidence: float = 0.95) -> str:
    """
    Generate a comprehensive statistical report for a matchup.
    
    Args:
        matchup: MatchupResult to analyze
        confidence: Confidence level for intervals
        
    Returns:
        Formatted string report
    """
    analyzer = StatisticalAnalyzer()
    
    # Win rate CIs
    red_ci = analyzer.win_rate_ci(matchup, 'red', confidence)
    black_ci = analyzer.win_rate_ci(matchup, 'black', confidence)
    
    # Effect size
    effect_size = analyzer.cohens_h(matchup.red_win_rate, matchup.black_win_rate)
    effect_interpretation = analyzer.effect_size_interpretation(effect_size)
    
    # Game length stats
    game_stats = analyzer.game_length_statistics(matchup.games)
    
    report = f"""
{'='*70}
STATISTICAL REPORT: {matchup.red_agent} vs {matchup.black_agent}
{'='*70}

Sample Size: {matchup.num_games} games

Win Rates ({confidence:.0%} Confidence Intervals):
  {matchup.red_agent:30s}: {red_ci}
  {matchup.black_agent:30s}: {black_ci}

Effect Size (Cohen's h): {effect_size:.3f} ({effect_interpretation})

Game Length Statistics:
  Mean:   {game_stats['mean']:.1f} moves
  Median: {game_stats['median']:.1f} moves
  Std:    {game_stats['std']:.1f} moves
  Range:  [{game_stats['min']}, {game_stats['max']}]
  IQR:    [{game_stats['q25']:.1f}, {game_stats['q75']:.1f}]

Outcome Distribution:
  {matchup.red_agent} wins: {matchup.red_wins:4d} ({matchup.red_win_rate:.1%})
  {matchup.black_agent} wins: {matchup.black_wins:4d} ({matchup.black_win_rate:.1%})
  Ties:       {matchup.ties:4d} ({matchup.tie_rate:.1%})
{'='*70}
"""
    return report
