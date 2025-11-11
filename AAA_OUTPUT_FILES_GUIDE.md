# Complete Guide to Output Files

**What This Document Covers:** Every single file that gets saved when you run `run_project.py`

---

## 📂 High-Level Directory Structure

```
results/
├── training/                    (Agent checkpoints from training)
├── evaluations/                 (Evaluation results and analysis)
└── SUMMARY_REPORT.txt          (Text summary of everything)
```

---

## 🎯 TRAINING OUTPUT - Q-Tables and Parameters

### Location: `results/training/`

When you train agents, each one gets its own timestamped directory:

```
results/training/
├── q-learning_curriculum_20251111_123456/
├── sarsa_curriculum_20251111_123456/
└── monte-carlo_curriculum_20251111_123456/
```

### Inside Each Agent Directory:

#### For Q-Learning and SARSA:
```
q-learning_curriculum_20251111_123456/
├── parameters.json          ← Hyperparameters used
├── red.npz                  ← Q-table for red player
└── black.npz                ← Q-table for black player
```

#### For Monte Carlo:
```
monte-carlo_curriculum_20251111_123456/
├── parameters.json          ← Hyperparameters used
└── final_checkpoint.save    ← Q-tables for both players (single file)
```

---

### FILE FORMAT DETAILS:

#### 1. `parameters.json`
**What it is:** JSON file with all training configuration  
**Format:**
```json
{
  "env": {
    "rows": 3,
    "columns": 4,
    "connect_n": 3,
    "reward": 1.0,
    "penalty": 0.0,
    "move_cost": 0.0
  },
  "model": {
    "agent-type": "q-learning",
    "gamma": 0.9,
    "alpha": 0.1,
    "epsilon": 0.1,
    "episodes": 5000
  }
}
```

**Why it exists:** So you can reload the agent later and know exactly what settings were used

**How to read it:** Any JSON viewer, or:
```python
import json
with open('parameters.json', 'r') as f:
    params = json.load(f)
print(params)
```

---

#### 2. `red.npz` and `black.npz`
**What it is:** NumPy compressed array files containing Q-tables  
**Format:** NPZ (NumPy's compressed format for dictionaries of arrays)

**Contents:**
- Keys: State strings (e.g., "000000000000_1" = empty board, player 1's turn)
- Values: Arrays of Q-values (one per action/column)

**File size:** Varies by state space explored
- Small board (3×4): ~50 KB - 500 KB per file
- Large board (6×7): Could be several MB

**How to read it:**
```python
import numpy as np

# Load Q-table for red player
data = np.load('red.npz', allow_pickle=True)

# Access Q-values for a specific state
state = "000000000000_1"  # Example state
if state in data.files:
    q_values = data[state]
    print(f"Q-values for state {state}: {q_values}")

# See all states explored
print(f"Total states visited: {len(data.files)}")
```

**What's inside the Q-values array:**
```python
# Example for 3×4 board (4 columns = 4 actions)
q_values = [0.234, 0.456, 0.123, 0.789]
#           col 0  col 1  col 2  col 3
```

---

#### 3. `final_checkpoint.save` (Monte Carlo only)
**What it is:** Single NPY file containing both red and black Q-tables  
**Format:** NumPy pickle format

**Why it's different:** Monte Carlo implementation uses a slightly different save format for historical reasons

**How to read it:**
```python
import numpy as np
from collections import defaultdict

# Load checkpoint
data = np.load('final_checkpoint.save', allow_pickle=True).item()

# Access Q-tables
q_red = data['red']
q_black = data['black']

print(f"Red states explored: {len(q_red)}")
print(f"Black states explored: {len(q_black)}")
```

---

## 📊 EVALUATION OUTPUT - Results and Analysis

### Location: `results/evaluations/`

Two types of evaluations get saved:

```
results/evaluations/
├── pairwise_evaluations_20251111_123456/
└── tournament_20251111_123456/
```

---

### PAIRWISE EVALUATIONS

```
pairwise_evaluations_20251111_123456/
├── all_pairwise_results.json
└── plots/
    ├── win_rates_q-learning_vs_sarsa.png
    ├── win_rates_q-learning_vs_monte-carlo.png
    ├── win_rates_sarsa_vs_monte-carlo.png
    ├── game_lengths_q-learning_vs_sarsa.png
    ├── game_lengths_q-learning_vs_monte-carlo.png
    ├── game_lengths_sarsa_vs_monte-carlo.png
    ├── win_rate_over_time_q-learning_vs_sarsa.png
    ├── win_rate_over_time_q-learning_vs_monte-carlo.png
    ├── win_rate_over_time_sarsa_vs_monte-carlo.png
    ├── opening_moves.png
    ├── tactical_accuracy_q-learning_vs_sarsa.png
    ├── tactical_accuracy_q-learning_vs_monte-carlo.png
    ├── tactical_accuracy_sarsa_vs_monte-carlo.png
    ├── column_usage_q-learning_vs_sarsa.png
    ├── column_usage_q-learning_vs_monte-carlo.png
    └── column_usage_sarsa_vs_monte-carlo.png
```

---

#### FILE: `all_pairwise_results.json`

**What it is:** Complete game-by-game results for all matchups  
**Size:** Can be large (several MB for 100+ games per matchup)

**Structure:**
```json
{
  "timestamp": "2025-11-11T12:34:56.789",
  "env_config": {
    "rows": 3,
    "columns": 4,
    "connect_n": 3,
    "reward": 1.0,
    "penalty": 0.0,
    "move_cost": 0.0
  },
  "num_matchups": 6,
  "matchups": [
    {
      "red_agent": "q-learning",
      "black_agent": "sarsa",
      "num_games": 100,
      "red_wins": 67,
      "black_wins": 27,
      "ties": 6,
      "red_win_rate": 0.67,
      "black_win_rate": 0.27,
      "tie_rate": 0.06,
      "games": [
        {
          "game_id": 0,
          "winner": "red",
          "num_moves": 8,
          "move_history": [
            [2, 0, 1],   // [row, col, player]
            [2, 1, -1],
            [1, 0, 1],
            // ... all moves in this game
          ],
          "red_agent": "q-learning",
          "black_agent": "sarsa",
          "final_reward": 1.0,
          "timestamp": "2025-11-11T12:35:01.234"
        },
        // ... 99 more games
      ],
      "timestamp": "2025-11-11T12:35:00.123"
    },
    // ... more matchups
  ]
}
```

**Key fields explained:**

- **`num_moves`**: Total moves in the game (should be ≤ rows × cols)
- **`move_history`**: List of [row, col, player] for every move
  - `player = 1` is red
  - `player = -1` is black
  - Example: `[2, 0, 1]` = red placed piece in row 2, column 0
- **`winner`**: "red", "black", or "tie"
- **`final_reward`**: The reward at game end (typically 1.0 for win, 0.0 for tie)

**How to analyze this file:**
```python
import json

with open('all_pairwise_results.json', 'r') as f:
    data = json.load(f)

# Get specific matchup
for matchup in data['matchups']:
    if matchup['red_agent'] == 'q-learning' and matchup['black_agent'] == 'sarsa':
        print(f"Q-Learning vs SARSA:")
        print(f"  Red wins: {matchup['red_wins']}")
        print(f"  Games: {len(matchup['games'])}")
        
        # Analyze first game
        first_game = matchup['games'][0]
        print(f"\n  First game:")
        print(f"    Winner: {first_game['winner']}")
        print(f"    Moves: {first_game['num_moves']}")
        print(f"    Move history: {first_game['move_history']}")
```

---

### VISUALIZATION FILES (PNG)

All plots are saved at **300 DPI** for publication quality.

#### 1. `win_rates_AGENT1_vs_AGENT2.png`
**What it shows:**
- Bar chart comparing win/loss/tie rates
- Error bars showing 95% confidence intervals
- Color-coded: Red for player 1, Black for player 2, Gray for ties

**Dimensions:** ~1200×800 pixels  
**Use case:** Quick comparison of head-to-head performance

---

#### 2. `game_lengths_AGENT1_vs_AGENT2.png`
**What it shows:**
- Histogram of game lengths (number of moves)
- Separate distributions for red wins, black wins, and ties
- Overlaid to show patterns

**Interpretation:**
- **Short games** (few moves) = Quick decisive victories
- **Long games** (many moves) = Back-and-forth tactical play
- **Games at max length** (rows × cols) = Ties (board full)

**Dimensions:** ~1200×800 pixels

---

#### 3. `win_rate_over_time_AGENT1_vs_AGENT2.png`
**What it shows:**
- Line plot of win rate across games
- Rolling average (smoothed)
- Shows if agent performance is stable or drifting

**Interpretation:**
- **Flat line** = Stable, consistent performance
- **Upward/downward trend** = One agent adapting or degrading
- **High variance** = Inconsistent play

**Dimensions:** ~1200×800 pixels

---

#### 4. `opening_moves.png`
**What it shows:**
- Heatmap of first move preferences for each agent
- One column per agent, showing which columns they prefer to open with

**Interpretation:**
- **Center preference** = Strategic play (center columns often stronger)
- **Edge preference** = Might indicate suboptimal learning
- **Uniform distribution** = Random-like play

**Dimensions:** ~1000×600 pixels

---

#### 5. `tactical_accuracy_AGENT1_vs_AGENT2.png`
**What it shows:**
- Bar chart comparing:
  - Winning moves taken (%)
  - Blocking moves made (%)
  - Blunders committed

**Interpretation:**
- **High winning move %** = Agent recognizes win opportunities
- **High blocking %** = Agent defends well
- **Low blunders** = Consistent tactical play

**Dimensions:** ~1200×800 pixels

---

#### 6. `column_usage_AGENT1_vs_AGENT2.png`
**What it shows:**
- Heatmap of how often each agent plays each column
- Darker = more frequently used

**Interpretation:**
- **Uniform usage** = Balanced strategy
- **Center-heavy** = Good positional play
- **Edge-heavy** = Potentially weak strategy

**Dimensions:** ~1000×600 pixels

---

### TOURNAMENT OUTPUT

```
tournament_20251111_123456/
├── tournament_results.json
└── plots/
    └── tournament_matrix.png
```

---

#### FILE: `tournament_results.json`

**What it is:** Round-robin tournament results with rankings

**Structure:**
```json
{
  "agents": ["q-learning", "sarsa", "monte-carlo"],
  "matchups": [
    // All pairwise matchup results (same format as pairwise JSON)
  ],
  "rankings": {
    "q-learning": {
      "rank": 1,
      "total_wins": 134
    },
    "sarsa": {
      "rank": 2,
      "total_wins": 89
    },
    "monte-carlo": {
      "rank": 3,
      "total_wins": 77
    }
  }
}
```

**Key fields:**
- **`agents`**: List of all participants
- **`matchups`**: Full details of every head-to-head (same as pairwise JSON)
- **`rankings`**: Final standings by total wins

---

#### PLOT: `tournament_matrix.png`

**What it shows:**
- Heatmap matrix showing win rates
- Rows = Red player, Columns = Black player
- Each cell = win rate for row agent when playing as red against column agent

**Interpretation:**
- **Diagonal** = N/A (agent doesn't play itself)
- **Darker cells** = Higher win rate
- **Asymmetric** = Position advantage (red vs black matters)

**Dimensions:** ~1000×1000 pixels

---

## 📄 SUMMARY REPORT

### FILE: `results/SUMMARY_REPORT.txt`

**What it is:** Plain text summary of entire run  
**Format:** Human-readable text

**Contents:**
```
========================================
TRAINING SUMMARY
========================================
Trained 3 agents using curriculum learning

q-learning:
  Workspace: results/training/q-learning_curriculum_20251111_123456
  Training time: 5.2 seconds
  Episodes: 5000

sarsa:
  Workspace: results/training/sarsa_curriculum_20251111_123456
  Training time: 5.4 seconds
  Episodes: 5000

monte-carlo:
  Workspace: results/training/monte-carlo_curriculum_20251111_123456
  Training time: 6.1 seconds
  Episodes: 5000

========================================
EVALUATION SUMMARY
========================================
Pairwise Evaluations:
  Location: results/evaluations/pairwise_evaluations_20251111_123456
  
  q-learning vs sarsa:
    Red (q-learning) wins: 67 (67.0%)
    Black (sarsa) wins: 27 (27.0%)
    Ties: 6 (6.0%)
  
  [... more matchups ...]

Tournament Results:
  Location: results/evaluations/tournament_20251111_123456
  
  Rankings:
    1. q-learning (134 wins)
    2. sarsa (89 wins)
    3. monte-carlo (77 wins)
```

**Use case:** Quick reference without opening JSON files

---

## 🔍 WHAT IS NOT SAVED TO FILES

### Metrics Computed But Only Printed to Console:

1. **Outcome-Specific Metrics** (from `advanced_metrics.py`)
   - Average moves to win by agent
   - Average moves to lose by agent
   - Average moves to tie
   - Standard deviations for all above

2. **Move Quality Scores** (from `advanced_metrics.py`)
   - 0-100 tactical quality score
   - Winning move score (0-50)
   - Blocking score (0-50)
   - Blunder rate percentage
   - Detailed breakdown (opportunities vs taken)

3. **Q-Table Statistics** (from `advanced_metrics.py`)
   - Mean Q-value
   - Q-value variance
   - Sparsity (% of Q-table that's zero)
   - Number of states visited
   - Memory usage estimates

### Why These Aren't Saved:

- They're computed on-the-fly from the JSON data
- JSON already has the raw data (game results, move histories)
- You can re-compute them anytime by re-analyzing the JSON
- Saves disk space

### How to Capture Console Output:

```bash
# Save all output to a log file
python run_project.py --agents all --episodes 5000 2>&1 | tee run_log.txt

# Now run_log.txt contains everything printed to console, including:
# - Training progress
# - Outcome metrics
# - Quality scores
# - Q-table statistics
```

---

## 📊 FILE SIZE ESTIMATES

**For a typical run (3 agents, 5000 episodes, 100 evaluation games):**

```
results/
├── training/                           ~1-5 MB total
│   ├── q-learning_.../                ~300-500 KB
│   │   ├── parameters.json            ~1 KB
│   │   ├── red.npz                    ~100-200 KB
│   │   └── black.npz                  ~100-200 KB
│   ├── sarsa_.../                     ~300-500 KB
│   └── monte-carlo_.../               ~300-500 KB
│
├── evaluations/                        ~5-15 MB total
│   ├── pairwise_.../
│   │   ├── all_pairwise_results.json  ~2-5 MB (has ALL game data)
│   │   └── plots/                     ~3-8 MB (18 PNG files × ~200-400 KB each)
│   └── tournament_.../
│       ├── tournament_results.json    ~3-6 MB
│       └── plots/tournament_matrix.png ~200-400 KB
│
└── SUMMARY_REPORT.txt                  ~5-10 KB

TOTAL: ~10-25 MB for complete run
```

**Scaling factors:**
- **More episodes** → Larger Q-tables (more states explored)
- **Larger board** → Exponentially larger Q-tables
- **More evaluation games** → Linearly larger JSON files
- **More agents** → More matchup combinations

---

## 🎯 QUICK REFERENCE TABLE

| File Type | Location | Contains | Size | Can Re-analyze? |
|-----------|----------|----------|------|-----------------|
| `parameters.json` | training/ | Hyperparameters | 1 KB | N/A |
| `red.npz` | training/ | Red Q-table | 100-500 KB | No |
| `black.npz` | training/ | Black Q-table | 100-500 KB | No |
| `final_checkpoint.save` | training/ | Both Q-tables | 200-800 KB | No |
| `all_pairwise_results.json` | evaluations/ | Every game's details | 2-5 MB | **Yes** |
| `tournament_results.json` | evaluations/ | Tournament + games | 3-6 MB | **Yes** |
| `*.png` | evaluations/plots/ | Visualizations | 200-400 KB each | **Yes** (from JSON) |
| `SUMMARY_REPORT.txt` | results/ | Text summary | 5-10 KB | No |

**Key insight:** The JSON files have ALL the raw data. You can always re-run analysis without re-running games.

---

## 💡 COMMON TASKS

### Task 1: Load a Trained Agent

```python
from evaluate import load_model
from connect_four_env import ConnectFourEnv

# Create environment
env = ConnectFourEnv(rows=3, cols=4, connect_n=3)

# Load agent
workspace = "results/training/q-learning_curriculum_20251111_123456"
agent, _ = load_model(workspace, env)

# Now you can use agent.eval_step(env) to get moves
```

---

### Task 2: Analyze Game Results Without Re-Running

```python
import json

# Load results
with open('results/evaluations/pairwise_evaluations_TIMESTAMP/all_pairwise_results.json', 'r') as f:
    data = json.load(f)

# Analyze specific matchup
for matchup in data['matchups']:
    if matchup['red_agent'] == 'q-learning':
        # Calculate average game length
        avg_length = sum(g['num_moves'] for g in matchup['games']) / len(matchup['games'])
        print(f"Average game length: {avg_length:.1f} moves")
        
        # Find longest game
        longest = max(matchup['games'], key=lambda g: g['num_moves'])
        print(f"Longest game: {longest['num_moves']} moves")
```

---

### Task 3: Re-create Visualizations from JSON

```python
from visualizations import Visualizer
from data_structures import MatchupResult, GameResult
import json

# Load results
with open('all_pairwise_results.json', 'r') as f:
    data = json.load(f)

# Reconstruct MatchupResult object
matchup_data = data['matchups'][0]  # First matchup
games = [GameResult(**g) for g in matchup_data['games']]
matchup = MatchupResult(
    red_agent=matchup_data['red_agent'],
    black_agent=matchup_data['black_agent'],
    num_games=matchup_data['num_games'],
    red_wins=matchup_data['red_wins'],
    black_wins=matchup_data['black_wins'],
    ties=matchup_data['ties'],
    games=games
)

# Re-create visualizations
visualizer = Visualizer(output_dir="new_plots")
visualizer.plot_win_rates_with_ci(matchup)
visualizer.plot_game_length_distribution(matchup)
# etc.
```

---

## ❓ FAQ

**Q: Can I delete the checkpoint files after evaluation?**  
A: Yes, but you won't be able to load those agents again. Keep them if you want to:
- Continue training
- Use them as opponents for new agents
- Analyze their Q-tables

**Q: Why are JSON files so large?**  
A: They contain complete move-by-move history for every game. Each game is ~20-50 moves, each move is 3 integers, so 100 games = ~6,000-15,000 integers.

**Q: Can I convert Q-tables to a more portable format?**  
A: Yes! Example:
```python
import numpy as np
import json

# Load
data = np.load('red.npz', allow_pickle=True)

# Convert to regular dict
q_table = {state: data[state].tolist() for state in data.files}

# Save as JSON
with open('red_q_table.json', 'w') as f:
    json.dump(q_table, f)
```

**Q: How do I know which workspace is the "best" agent?**  
A: Check `SUMMARY_REPORT.txt` for tournament rankings, or the final section of console output.

---

**Bottom line:** JSON files have everything. Q-tables are for loading agents. PNGs are for visualization. Console output has metrics not saved to files.
