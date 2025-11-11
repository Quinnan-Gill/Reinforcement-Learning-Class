# Connect Four RL Project - Complete Architecture Map

---

## 🎯 THE BIG PICTURE (What Happens When You Run)

```
YOU TYPE:
python run_project.py --agents all --episodes 5000

WHAT HAPPENS:
1. Creates environment (connect_four_env.py)
2. Trains 3 agents (q_learning.py, sarsa.py, monte_carlo.py)
   - Each inherits from rl_agent.py base class
   - Trains against random_agent.py then frozen_agent.py checkpoints
3. Evaluates all agents (evaluator.py)
   - Uses game_analyzer.py for tactical analysis
   - Uses advanced_metrics.py for quality scores
   - Uses metrics.py for statistical tests
   - Uses visualizations.py for plots
4. Saves everything (checkpoints.py, data_structures.py)
5. Prints comprehensive report
```

---

## 📊 FILE HIERARCHY (By Layer)

### **LAYER 1: Core Game Engine**
Files that define the game itself

```
connect_four_env.py
├── Defines the game board, rules, win detection
├── Handles move validation
├── Tracks game state
└── Used by: EVERYONE (all other files need this)
```

---

### **LAYER 2: Base Agent Framework**
Abstract base class and utilities

```
rl_agent.py (BASE CLASS)
├── Defines Q-tables structure (red, black)
├── Implements epsilon-greedy selection
├── Handles save/load of models
├── Provides get_q() and set_q() methods
└── Extended by: q_learning.py, sarsa.py, monte_carlo.py
```

---

### **LAYER 3: RL Algorithm Implementations**
Concrete agents that inherit from RLModel

```
q_learning.py
├── Inherits: RLModel
├── Algorithm: Off-policy TD learning
├── Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
└── Used by: run_project.py

sarsa.py
├── Inherits: RLModel
├── Algorithm: On-policy TD learning
├── Update: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
└── Used by: run_project.py

monte_carlo.py
├── Inherits: RLModel
├── Algorithm: Episode-based learning
├── Update: Q(s,a) ← Q(s,a) + α[G - Q(s,a)] where G = sum of future rewards
└── Used by: run_project.py
```

---

### **LAYER 4: Opponent Agents**
Non-learning agents used during training

```
random_agent.py
├── Selects random valid moves
├── Used for: Phase 1 training (vs Random)
└── Called by: run_project.py training loops

frozen_agent.py
├── Loads saved Q-tables from checkpoint
├── Plays greedily (no exploration)
├── Used for: Curriculum learning phases (vs past self)
└── Called by: run_project.py training loops
```

---

### **LAYER 5: Training Orchestration**
The master controller

```
run_project.py ⭐ (THIS IS WHAT YOU RUN)
│
├── Imports:
│   ├── connect_four_env.py (creates game)
│   ├── q_learning.py, sarsa.py, monte_carlo.py (agents to train)
│   ├── random_agent.py (Phase 1 opponent)
│   ├── frozen_agent.py (Phase 3+ opponent)
│   ├── evaluator.py (runs tournaments)
│   ├── advanced_metrics.py (timing)
│   └── visualizations.py (plots)
│
├── Training Flow (Curriculum Learning):
│   1. Phase 1: Train vs Random (2× episodes)
│   2. Phase 2: Self-play (2× episodes)
│   3. Phase 3: vs Phase 1 checkpoint
│   4. Iteration phases: vs previous best (N times)
│   5. Save final checkpoint
│
└── Evaluation Flow:
    1. Load all trained agents
    2. Run pairwise matchups (A vs B, B vs A)
    3. Run tournament (round-robin)
    4. Generate visualizations
    5. Save results
```

---

### **LAYER 6: Evaluation System**
Comprehensive analysis after training

```
evaluator.py (EVALUATION ORCHESTRATOR)
├── Class: Evaluator
├── Methods:
│   ├── load_agent() → uses evaluate.py's load_model()
│   ├── run_single_game() → plays one game
│   ├── evaluate_matchup() → plays N games, prints reports
│   └── run_tournament() → round-robin competition
│
├── Uses:
│   ├── data_structures.py (GameResult, MatchupResult)
│   ├── game_analyzer.py (tactical analysis)
│   ├── advanced_metrics.py (outcome metrics, quality scores)
│   ├── metrics.py (statistical tests)
│   └── visualizations.py (plots)
│
└── Called by: run_project.py

evaluate.py (UTILITY FUNCTIONS)
├── Function: load_model(workspace) → loads saved agent
├── Function: evaluate_game() → simple game playing
├── Standalone script: Can be run directly for quick tests
└── Used by: evaluator.py (imports load_model)
    └── NOTE: evaluator.py DEPENDS on evaluate.py for load_model()
```

---

### **LAYER 7: Analysis Modules**
Detailed game and agent analysis

```
game_analyzer.py (TACTICAL ANALYSIS)
├── Class: GameAnalyzer
├── Analyzes individual games for:
│   ├── Winning moves taken/missed
│   ├── Blocking moves made/missed
│   ├── Blunders (missed opportunities)
│   └── Move-by-move quality
├── Uses: data_structures.py (GameResult)
└── Called by: evaluator.py, advanced_metrics.py

advanced_metrics.py (ADVANCED METRICS)
├── Classes:
│   ├── OutcomeMetrics (avg moves by win/loss/tie)
│   ├── QValueStatistics (Q-table analysis)
│   ├── MoveQualityScore (0-100 tactical score)
│   └── AdvancedMetricsAnalyzer (computes all metrics)
├── Functions:
│   ├── generate_outcome_report() → formatted text
│   ├── generate_quality_score_report() → formatted text
│   └── generate_q_table_report() → Q-table stats
├── Uses: game_analyzer.py, data_structures.py
└── Called by: evaluator.py (automatically during evaluation)

metrics.py (STATISTICAL ANALYSIS)
├── Class: StatisticalAnalyzer
├── Methods:
│   ├── binomial_ci() → confidence intervals
│   ├── compare_win_rates() → hypothesis testing
│   ├── cohens_h() → effect size
│   └── bootstrap_ci() → resampling estimates
├── Uses: data_structures.py
└── Called by: evaluator.py (can be used for analysis)

visualizations.py (PLOTTING)
├── Class: Visualizer
├── Creates 6+ plots per matchup:
│   ├── Win rate comparisons
│   ├── Game length distributions
│   ├── Temporal stability
│   ├── Opening move preferences
│   ├── Tactical accuracy
│   └── Column usage patterns
├── Uses: data_structures.py
└── Called by: run_project.py, evaluator.py
```

---

### **LAYER 8: Data Structures**
Shared types to avoid circular imports

```
data_structures.py (SHARED TYPES)
├── Class: GameResult (single game data)
│   ├── winner, num_moves, move_history
│   ├── agent names, rewards, timestamp
│   └── to_dict() for JSON serialization
│
├── Class: MatchupResult (aggregated results)
│   ├── red/black wins/ties
│   ├── list of GameResult objects
│   └── computed properties (win_rate, tie_rate)
│
└── Used by:
    ├── evaluator.py (creates these objects)
    ├── game_analyzer.py (analyzes GameResult)
    ├── advanced_metrics.py (computes from MatchupResult)
    ├── metrics.py (statistical analysis)
    └── visualizations.py (plots from data)

WHY THIS EXISTS: Breaks circular import between evaluator.py and game_analyzer.py
```

---

### **LAYER 9: Persistence**
Save and load functionality

```
checkpoints.py (SAVE/LOAD SYSTEM)
├── Functions:
│   ├── save_checkpoint() → saves Q-tables
│   ├── load_checkpoint() → loads Q-tables
│   ├── save_params() → saves hyperparameters
│   └── load_params() → loads hyperparameters
│
├── Formats supported:
│   ├── Directory format: workspace/red.npz, workspace/black.npz
│   └── Single file format: checkpoint.save
│
└── Used by:
    ├── rl_agent.py (save_model, load_workspace)
    ├── evaluate.py (load_model)
    ├── frozen_agent.py (loads checkpoints)
    └── All algorithm implementations
```

---

### **LAYER 10: Utilities**
Diagnostic and standalone tools

```
diagnostics.py (TESTING TOOL)
├── Class: ConnectFourDiagnostics
├── Functions:
│   ├── count_reachable_states() → state space size
│   ├── analyze_terminal_states() → win/loss/draw counts
│   ├── test_basic_functionality() → unit tests
│   └── sample_random_game() → play random game
└── Usage: Standalone tool for testing environment
    └── python diagnostics.py

plot_learning_curve.py (VISUALIZATION TOOL)
├── Function: plot_learning_curve(workspaces)
├── Plots training progress over episodes
└── Usage: Standalone tool for analyzing training
    └── python plot_learning_curve.py -i workspace1 -i workspace2

run_evaluation.py (EVALUATION TOOL)
├── Comprehensive evaluation script
├── Modes:
│   ├── matchup: Head-to-head evaluation
│   ├── tournament: Multi-agent competition
│   └── checkpoints: Compare training checkpoints
└── Usage: Alternative to run_project.py for evaluation only
    └── python run_evaluation.py --mode matchup --red-agent w1 --black-agent w2
```

---

## 🔄 DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│  YOU RUN: python run_project.py --agents all --episodes 5000   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: CREATE ENVIRONMENT                                     │
│  run_project.py → connect_four_env.py                          │
│  Creates: ConnectFourEnv(rows=3, cols=4, connect_n=3)         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: TRAIN AGENTS (for each algorithm)                     │
│                                                                  │
│  run_project.py creates:                                        │
│  ├── agent = QLearning(env, opts)   ← inherits from rl_agent  │
│  ├── opponent = RandomAgent(env)                               │
│  └── frozen = FrozenAgent(env, checkpoint_path)                │
│                                                                  │
│  Training phases:                                               │
│  1. agent.train() vs random_agent                              │
│  2. agent.train() (self-play)                                  │
│  3. agent.train() vs frozen_agent (phase 1 checkpoint)         │
│  4. agent.train() vs frozen_agent (best checkpoint) [repeat]   │
│                                                                  │
│  Each phase:                                                    │
│  ├── Updates Q-tables (red, black)                             │
│  └── Saves checkpoint → checkpoints.py                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: EVALUATION                                             │
│                                                                  │
│  run_project.py → evaluator.py                                 │
│                                                                  │
│  evaluator.load_agent(workspace)                               │
│    ↓                                                            │
│  evaluate.py: load_model(workspace)                            │
│    ↓                                                            │
│  checkpoints.py: load_checkpoint()                             │
│    ↓                                                            │
│  Returns: Loaded agent with Q-tables                           │
│                                                                  │
│  evaluator.evaluate_matchup(agent1, agent2, num_games=100)    │
│    ↓                                                            │
│  Plays 100 games → creates GameResult objects                  │
│    ↓                                                            │
│  Stores in MatchupResult                                       │
│    ↓                                                            │
│  AUTOMATIC ANALYSIS:                                           │
│  ├── game_analyzer.py → analyzes each game                    │
│  ├── advanced_metrics.py → computes quality scores            │
│  └── Prints comprehensive reports                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: VISUALIZATIONS                                         │
│                                                                  │
│  run_project.py → visualizations.py                            │
│                                                                  │
│  For each matchup:                                              │
│  ├── Win rate bar charts                                       │
│  ├── Game length histograms                                    │
│  ├── Temporal stability plots                                  │
│  ├── Opening move heatmaps                                     │
│  ├── Tactical accuracy comparisons                             │
│  └── Column usage patterns                                     │
│                                                                  │
│  Saves to: results/evaluations/.../plots/*.png                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: SAVE RESULTS                                           │
│                                                                  │
│  Results saved:                                                 │
│  ├── results/training/agent_name/                              │
│  │   └── Q-tables (red.npz, black.npz, parameters.json)        │
│  ├── results/evaluations/pairwise_evaluations_TIMESTAMP/       │
│  │   ├── all_pairwise_results.json                             │
│  │   └── plots/                                                 │
│  └── results/SUMMARY_REPORT.txt                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎓 CLASS INHERITANCE DIAGRAM

```
┌─────────────────────┐
│   RLModel           │  (rl_agent.py)
│   (Base Class)      │
├─────────────────────┤
│ • q (Q-tables)      │
│ • gamma, alpha, ε   │
│ • get_q(), set_q()  │
│ • select_action()   │
│ • eval_step()       │
│ • save_model()      │
│ • load_workspace()  │
└──────┬──────────────┘
       │
       ├─────────────────────────────────────────┐
       │                 │                       │
       ▼                 ▼                       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
│ QLearning    │  │   Sarsa      │  │   MonteCarlo         │
├──────────────┤  ├──────────────┤  ├──────────────────────┤
│ • train_step()│  │ • train_step()│  │ • generate_episode() │
│   Off-policy │  │   On-policy  │  │ • calculate_returns()│
│   TD update  │  │   TD update  │  │ • update_q_values()  │
│              │  │              │  │   Episode-based      │
└──────────────┘  └──────────────┘  └──────────────────────┘
```

---

## 🔍 KEY RELATIONSHIPS

### Who Calls Who?

```
run_project.py (MASTER)
├── Creates: ConnectFourEnv
├── Creates: QLearning, Sarsa, MonteCarlo (inherit from RLModel)
├── Creates: RandomAgent, FrozenAgent
├── Calls: agent.train() methods
├── Calls: Evaluator.load_agent()
│   └── Which calls: evaluate.load_model()
│       └── Which calls: checkpoints.load_checkpoint()
├── Calls: Evaluator.evaluate_matchup()
│   ├── Which calls: GameAnalyzer.analyze_multiple_games()
│   ├── Which calls: AdvancedMetricsAnalyzer methods
│   └── Which calls: Visualizer.plot_*()
└── Saves results via checkpoints.py

evaluator.py
├── Uses: evaluate.py (imports load_model)
├── Uses: data_structures.py (GameResult, MatchupResult)
├── Uses: game_analyzer.py (analyzes games)
├── Uses: advanced_metrics.py (computes metrics)
└── Creates results that metrics.py and visualizations.py can analyze

evaluate.py (DEPENDENCY of evaluator.py)
├── Provides: load_model() function
└── Used by: evaluator.py (line 58)

game_analyzer.py
├── Uses: data_structures.py (GameResult)
├── Uses: connect_four_env.py (replays games)
└── Returns: GameAnalysis objects

advanced_metrics.py
├── Uses: data_structures.py (GameResult, MatchupResult)
├── Uses: game_analyzer.py (GameAnalysis)
└── Returns: Formatted reports (printed to console)

data_structures.py
├── Used by: evaluator.py, game_analyzer.py, advanced_metrics.py
├── Used by: metrics.py, visualizations.py
└── Purpose: Breaks circular imports
```

---

## 🎯 WHICH FILES DO YOU NEED?

### **CRITICAL (Cannot run without these):**
```
✅ connect_four_env.py       - The game
✅ rl_agent.py                - Base class for algorithms
✅ q_learning.py              - Algorithm
✅ sarsa.py                   - Algorithm  
✅ monte_carlo.py             - Algorithm
✅ random_agent.py            - Training opponent
✅ frozen_agent.py            - Training opponent
✅ checkpoints.py             - Save/load system
✅ run_project.py             - Master orchestrator
✅ evaluator.py               - Evaluation framework
✅ evaluate.py                - Provides load_model() to evaluator
✅ data_structures.py         - Shared types
✅ game_analyzer.py           - Tactical analysis
✅ advanced_metrics.py        - Quality metrics
✅ metrics.py                 - Statistical analysis
✅ visualizations.py          - Plotting
```

### **OPTIONAL (Utility tools):**
```
⚪ diagnostics.py             - Testing tool (run independently)
⚪ plot_learning_curve.py     - Visualization tool (run independently)
⚪ run_evaluation.py          - Alternative evaluation script
```

---

## 📋 SUMMARY TABLE

| File | Layer | Purpose | Used By | Depends On |
|------|-------|---------|---------|------------|
| `connect_four_env.py` | 1 - Core | Game engine | Everyone | - |
| `rl_agent.py` | 2 - Base | Base agent class | Algorithm files | connect_four_env |
| `q_learning.py` | 3 - Algorithms | Q-Learning agent | run_project | rl_agent |
| `sarsa.py` | 3 - Algorithms | SARSA agent | run_project | rl_agent |
| `monte_carlo.py` | 3 - Algorithms | Monte Carlo agent | run_project | rl_agent |
| `random_agent.py` | 4 - Opponents | Random opponent | run_project | connect_four_env |
| `frozen_agent.py` | 4 - Opponents | Checkpoint opponent | run_project | connect_four_env, checkpoints |
| `run_project.py` | 5 - Orchestration | **Main script** | **You** | All of above + evaluator |
| `evaluator.py` | 6 - Evaluation | Evaluation framework | run_project | evaluate, data_structures, game_analyzer, advanced_metrics |
| `evaluate.py` | 6 - Evaluation | Load utility | evaluator | checkpoints, rl_agent |
| `game_analyzer.py` | 7 - Analysis | Tactical analysis | evaluator, advanced_metrics | data_structures, connect_four_env |
| `advanced_metrics.py` | 7 - Analysis | Quality metrics | evaluator | data_structures, game_analyzer |
| `metrics.py` | 7 - Analysis | Statistical tests | evaluator (optional) | data_structures |
| `visualizations.py` | 7 - Analysis | Plotting | run_project, evaluator | data_structures |
| `data_structures.py` | 8 - Data | Shared types | All analysis modules | - |
| `checkpoints.py` | 9 - Persistence | Save/load | rl_agent, evaluate, frozen_agent | - |
| `diagnostics.py` | 10 - Utils | Testing tool | Standalone | connect_four_env |
| `plot_learning_curve.py` | 10 - Utils | Plot training | Standalone | - |
| `run_evaluation.py` | 10 - Utils | Evaluation tool | Standalone | evaluator |

---

## 💡 KEY INSIGHTS

1. **evaluate.py vs evaluator.py confusion:**
   - `evaluate.py` = Utility functions (especially `load_model`)
   - `evaluator.py` = Comprehensive framework (Evaluator class)
   - `evaluator.py` DEPENDS on `evaluate.py` (line 58 import)
   - **Both are needed**

2. **data_structures.py exists to break circular imports:**
   - Without it: evaluator.py imports from game_analyzer.py AND game_analyzer.py imports from evaluator.py = ERROR
   - With it: Both import from data_structures.py = works

3. **The 3 layers of evaluation:**
   - Layer 1: `evaluator.py` orchestrates
   - Layer 2: `game_analyzer.py` analyzes tactics
   - Layer 3: `advanced_metrics.py` computes scores
   - All use `data_structures.py` to communicate

4. **Why so many files:**
   - Separation of concerns
   - Each file has ONE clear responsibility
   - Makes testing easier
   - Makes future changes easier

---

## 🚀 WHAT HAPPENS WHEN YOU RUN

**Command:**
```bash
python run_project.py --agents all --episodes 5000
```

**Step-by-step execution:**

1. **Parse arguments** (run_project.py lines 1-50)
2. **Create environment** (connect_four_env.py)
3. **For each algorithm** (Q-Learning, SARSA, Monte Carlo):
   - Create agent instance (inherits from rl_agent.py)
   - Run curriculum training phases:
     - Phase 1: vs random_agent.py
     - Phase 2: self-play
     - Phase 3: vs frozen_agent.py (past checkpoint)
     - Iterations: vs frozen_agent.py (best checkpoint)
   - Save final checkpoint (checkpoints.py)
4. **Load all trained agents:**
   - Uses evaluator.py → evaluate.py → checkpoints.py
5. **Run pairwise evaluations:**
   - Uses evaluator.py
   - Each game creates GameResult (data_structures.py)
   - Aggregates into MatchupResult (data_structures.py)
   - Automatically analyzes:
     - Tactical quality (game_analyzer.py)
     - Outcome metrics (advanced_metrics.py)
     - Statistical tests (metrics.py - optional)
6. **Generate visualizations:**
   - Uses visualizations.py
   - Creates 6+ plots per matchup
7. **Run tournament:**
   - Uses evaluator.py
   - Round-robin all agents
8. **Save everything:**
   - JSON results
   - Plots
   - Summary report


