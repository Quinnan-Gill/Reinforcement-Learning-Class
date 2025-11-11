"""
Visualization Module

Comprehensive visualization suite for RL agent evaluation:
- Win rate comparisons with confidence intervals
- Game length distributions
- Learning curves across checkpoints
- Move heatmaps
- Tournament result matrices
- Statistical comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from evaluator import GameResult, MatchupResult
from metrics import StatisticalAnalyzer, ConfidenceInterval
from game_analyzer import GameAnalysis, StrategicPatternAnalyzer

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Creates visualizations for evaluation results.
    """
    
    def __init__(self, output_dir: str = "evaluation_results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_win_rates_with_ci(
        self,
        matchup: MatchupResult,
        confidence: float = 0.95,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot win rates with confidence intervals.
        
        Args:
            matchup: MatchupResult to visualize
            confidence: Confidence level for intervals
            save: Whether to save figure
            filename: Custom filename
            
        Returns:
            matplotlib Figure
        """
        analyzer = StatisticalAnalyzer()
        
        # Calculate CIs
        red_ci = analyzer.win_rate_ci(matchup, 'red', confidence)
        black_ci = analyzer.win_rate_ci(matchup, 'black', confidence)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        agents = [matchup.red_agent, matchup.black_agent]
        win_rates = [red_ci.point_estimate, black_ci.point_estimate]
        errors = [
            [red_ci.point_estimate - red_ci.lower_bound, black_ci.point_estimate - black_ci.lower_bound],
            [red_ci.upper_bound - red_ci.point_estimate, black_ci.upper_bound - black_ci.point_estimate]
        ]
        
        colors = ['#e74c3c', '#34495e']
        bars = ax.bar(agents, win_rates, color=colors, alpha=0.7, edgecolor='black')
        ax.errorbar(
            agents,
            win_rates,
            yerr=errors,
            fmt='none',
            ecolor='black',
            capsize=10,
            capthick=2,
            elinewidth=2
        )
        
        # Labels and formatting
        ax.set_ylabel('Win Rate', fontsize=14, fontweight='bold')
        ax.set_title(
            f'Win Rates with {confidence:.0%} Confidence Intervals\n({matchup.num_games} games)',
            fontsize=16,
            fontweight='bold'
        )
        ax.set_ylim([0, 1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Add value labels on bars
        for bar, rate, ci in zip(bars, win_rates, [red_ci, black_ci]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.05,
                f'{rate:.1%}\n[{ci.lower_bound:.1%}, {ci.upper_bound:.1%}]',
                ha='center',
                va='bottom',
                fontsize=11
            )
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"win_rates_{matchup.red_agent}_vs_{matchup.black_agent}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_game_length_distribution(
        self,
        matchup: MatchupResult,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of game lengths.
        
        Args:
            matchup: MatchupResult to visualize
            save: Whether to save figure
            filename: Custom filename
            
        Returns:
            matplotlib Figure
        """
        game_lengths = [g.num_moves for g in matchup.games]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(game_lengths, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(np.mean(game_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(game_lengths):.1f}')
        ax1.axvline(np.median(game_lengths), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(game_lengths):.1f}')
        ax1.set_xlabel('Number of Moves', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Game Length Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(game_lengths, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Number of Moves', fontsize=12, fontweight='bold')
        ax2.set_title('Game Length Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"n={len(game_lengths)}\nμ={np.mean(game_lengths):.1f}\nσ={np.std(game_lengths):.1f}"
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(
            f'{matchup.red_agent} vs {matchup.black_agent}',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"game_lengths_{matchup.red_agent}_vs_{matchup.black_agent}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_win_rate_by_game_number(
        self,
        matchup: MatchupResult,
        window: int = 10,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot rolling win rate over time to see if performance changes.
        
        Args:
            matchup: MatchupResult to visualize
            window: Rolling window size
            save: Whether to save figure
            filename: Custom filename
            
        Returns:
            matplotlib Figure
        """
        # Calculate rolling win rate for red
        red_outcomes = [1 if g.winner == 'red' else 0 for g in matchup.games]
        
        rolling_wr = []
        for i in range(len(red_outcomes)):
            start = max(0, i - window + 1)
            rolling_wr.append(np.mean(red_outcomes[start:i+1]))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        game_numbers = list(range(1, len(red_outcomes) + 1))
        ax.plot(game_numbers, rolling_wr, linewidth=2, color='steelblue', label=f'Rolling Win Rate (window={window})')
        ax.axhline(matchup.red_win_rate, color='red', linestyle='--', linewidth=2, label=f'Overall: {matchup.red_win_rate:.1%}')
        
        ax.set_xlabel('Game Number', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{matchup.red_agent} Win Rate', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Win Rate Over Time: {matchup.red_agent} vs {matchup.black_agent}',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"win_rate_over_time_{matchup.red_agent}_vs_{matchup.black_agent}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_opening_moves(
        self,
        game_analyses: List[GameAnalysis],
        env_cols: int,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of opening moves.
        
        Args:
            game_analyses: List of GameAnalysis objects
            env_cols: Number of columns in board
            save: Whether to save figure
            filename: Custom filename
            
        Returns:
            matplotlib Figure
        """
        analyzer = StrategicPatternAnalyzer()
        openings = analyzer.opening_preferences(game_analyses)
        
        # Ensure all columns represented
        all_openings = {col: openings.get(col, 0) for col in range(env_cols)}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        columns = list(all_openings.keys())
        counts = list(all_openings.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))
        bars = ax.bar(columns, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                pct = 100 * count / total
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center',
                    va='bottom'
                )
        
        ax.set_xlabel('Starting Column', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Opening Move Distribution ({len(game_analyses)} games)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xticks(columns)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = "opening_moves.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_tactical_accuracy(
        self,
        game_analyses: List[GameAnalysis],
        red_agent: str,
        black_agent: str,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot tactical accuracy (winning moves and blocks) for both agents.
        
        Args:
            game_analyses: List of GameAnalysis objects
            red_agent: Name of red agent
            black_agent: Name of black agent
            save: Whether to save figure
            filename: Custom filename
            
        Returns:
            matplotlib Figure
        """
        analyzer = StrategicPatternAnalyzer()
        
        red_tactics = analyzer.tactical_awareness(game_analyses, 'red')
        black_tactics = analyzer.tactical_awareness(game_analyses, 'black')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Winning accuracy
        agents = [red_agent, black_agent]
        winning_acc = [red_tactics['winning_accuracy'], black_tactics['winning_accuracy']]
        colors = ['#e74c3c', '#34495e']
        
        bars1 = ax1.bar(agents, winning_acc, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Winning Move Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, acc, tactics in zip(bars1, winning_acc, [red_tactics, black_tactics]):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.05,
                f'{acc:.1%}\n({tactics["winning_moves_taken"]}/{tactics["winning_opportunities"]})',
                ha='center',
                va='bottom'
            )
        
        # Blocking accuracy
        blocking_acc = [red_tactics['blocking_accuracy'], black_tactics['blocking_accuracy']]
        
        bars2 = ax2.bar(agents, blocking_acc, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Blocking Accuracy', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, acc, tactics in zip(bars2, blocking_acc, [red_tactics, black_tactics]):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.05,
                f'{acc:.1%}\n({tactics["blocks_made"]}/{tactics["block_opportunities"]})',
                ha='center',
                va='bottom'
            )
        
        plt.suptitle(
            f'Tactical Awareness: {red_agent} vs {black_agent}',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"tactical_accuracy_{red_agent}_vs_{black_agent}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_tournament_matrix(
        self,
        tournament_results: Dict,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap matrix showing all pairwise win rates.
        
        Args:
            tournament_results: Dictionary from run_tournament
            save: Whether to save figure
            filename: Custom filename
            
        Returns:
            matplotlib Figure
        """
        agents = tournament_results['agents']
        n_agents = len(agents)
        
        # Create win rate matrix
        win_matrix = np.zeros((n_agents, n_agents))
        
        for matchup in tournament_results['matchups']:
            red_idx = agents.index(matchup['red_agent'])
            black_idx = agents.index(matchup['black_agent'])
            win_matrix[red_idx, black_idx] = matchup['red_win_rate']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(win_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Win Rate (as row agent vs column agent)', fontsize=12, fontweight='bold')
        
        # Set ticks
        ax.set_xticks(np.arange(n_agents))
        ax.set_yticks(np.arange(n_agents))
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.set_yticklabels(agents)
        
        # Add text annotations
        for i in range(n_agents):
            for j in range(n_agents):
                if win_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{win_matrix[i, j]:.1%}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Tournament Win Rate Matrix\n(Row agent vs Column agent)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = "tournament_matrix.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_column_usage(
        self,
        game_analyses: List[GameAnalysis],
        red_agent: str,
        black_agent: str,
        env_cols: int,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare column usage patterns between two agents.
        
        Args:
            game_analyses: List of GameAnalysis objects
            red_agent: Name of red agent
            black_agent: Name of black agent
            env_cols: Number of columns in board
            save: Whether to save figure
            filename: Custom filename
            
        Returns:
            matplotlib Figure
        """
        analyzer = StrategicPatternAnalyzer()
        
        red_usage = analyzer.column_usage_distribution(game_analyses, 'red')
        black_usage = analyzer.column_usage_distribution(game_analyses, 'black')
        
        # Normalize to percentages
        red_total = sum(red_usage.values())
        black_total = sum(black_usage.values())
        
        red_pct = {col: 100 * red_usage.get(col, 0) / red_total for col in range(env_cols)}
        black_pct = {col: 100 * black_usage.get(col, 0) / black_total for col in range(env_cols)}
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(env_cols)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [red_pct[col] for col in range(env_cols)],
                      width, label=red_agent, color='#e74c3c', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, [black_pct[col] for col in range(env_cols)],
                      width, label=black_agent, color='#34495e', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Column', fontsize=12, fontweight='bold')
        ax.set_ylabel('Usage Percentage', fontsize=12, fontweight='bold')
        ax.set_title('Column Usage Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(range(env_cols))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"column_usage_{red_agent}_vs_{black_agent}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def create_comprehensive_report(
        self,
        matchup: MatchupResult,
        game_analyses: List[GameAnalysis],
        env_cols: int
    ):
        """
        Generate all visualizations for a matchup.
        
        Args:
            matchup: MatchupResult to visualize
            game_analyses: List of GameAnalysis objects
            env_cols: Number of columns in board
        """
        print("\nGenerating comprehensive visualization report...")
        
        self.plot_win_rates_with_ci(matchup)
        self.plot_game_length_distribution(matchup)
        self.plot_win_rate_by_game_number(matchup)
        self.plot_opening_moves(game_analyses, env_cols)
        self.plot_tactical_accuracy(game_analyses, matchup.red_agent, matchup.black_agent)
        self.plot_column_usage(game_analyses, matchup.red_agent, matchup.black_agent, env_cols)
        
        print(f"\nAll visualizations saved to: {self.output_dir}")
