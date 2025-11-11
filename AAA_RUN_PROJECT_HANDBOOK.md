# run_project.py - Command Line Reference

**Your complete handbook for running the Connect Four RL training and evaluation pipeline**

---

## 🚀 Quick Start

### Simplest Possible Run
```bash
python run_project.py
```
This uses ALL defaults:
- Trains all 3 algorithms (Q-Learning, SARSA, Monte Carlo)
- Uses 6×7 board with Connect-4
- 5000 episodes per training phase
- Curriculum learning mode
- 100 games per evaluation
- Saves to `results/`

---

### Common Use Cases

#### 1. Quick Test (Fast, for debugging)
```bash
python run_project.py --agents q-learning sarsa --episodes 100 --rows 3 --columns 4 --connect-n 3
```
**Time:** ~10 seconds  
**Use:** Testing changes, quick experiments

---

#### 2. Standard Research Run
```bash
python run_project.py --agents all --episodes 5000 --rows 3 --columns 4 --connect-n 3
```
**Time:** ~30-60 seconds  
**Use:** Most experiments

---

#### 3. High-Quality Run (More games)
```bash
python run_project.py --agents all --episodes 10000 --eval-games 500
```
**Time:** ~5-10 minutes  
**Use:** Publication-quality results

---

#### 4. Hyperparameter Experiment
```bash
python run_project.py --gamma 0.95 --alpha 0.3 --epsilon 0.2 --episodes 5000
```
**Time:** ~30-60 seconds  
**Use:** Testing different learning parameters

---

#### 5. Evaluation Only (Use Pre-Trained Agents)
```bash
python run_project.py --skip-training --agent-dirs \
    results/training/q-learning_curriculum_20251111_123456 \
    results/training/sarsa_curriculum_20251111_123456 \
    results/training/monte-carlo_curriculum_20251111_123456
```
**Time:** ~10-30 seconds  
**Use:** Re-evaluate without re-training

---

#### 6. Training Only (No Evaluation)
```bash
python run_project.py --skip-evaluation --episodes 10000
```
**Time:** ~2-5 minutes  
**Use:** Just generate trained agents for later analysis

---

## 📋 Complete Argument Reference

### WHAT TO RUN

#### `--agents [AGENT_LIST]`
**Description:** Which algorithms to train  
**Type:** List of strings  
**Choices:** `q-learning`, `sarsa`, `monte-carlo`, `all`  
**Default:** `all`

**Examples:**
```bash
# Train all algorithms
python run_project.py --agents all

# Train just Q-Learning
python run_project.py --agents q-learning

# Train Q-Learning and SARSA only
python run_project.py --agents q-learning sarsa

# Train all three (explicit)
python run_project.py --agents q-learning sarsa monte-carlo
```

**Notes:**
- `all` is equivalent to `q-learning sarsa monte-carlo`
- Order doesn't matter
- Each agent gets its own timestamped directory

---

#### `--training-mode [MODE]`
**Description:** How agents should train  
**Type:** String  
**Choices:** `curriculum`, `self-play`, `vs-random`  
**Default:** `curriculum`

**Options:**

1. **`curriculum`** (Recommended)
   - Phase 1: vs Random opponent (2× episodes)
   - Phase 2: Self-play (2× episodes)
   - Phase 3: vs Phase 1 checkpoint
   - Phase 4-N: vs previous best checkpoint
   - Most sophisticated, best results

2. **`self-play`**
   - Agent plays against itself for all episodes
   - Both red and black players learn
   - Simpler but can lead to local optima

3. **`vs-random`**
   - Agent always plays against random opponent
   - Only red (agent) learns, black is random
   - Good baseline but limited learning

**Examples:**
```bash
# Use curriculum learning (default)
python run_project.py --training-mode curriculum

# Simple self-play
python run_project.py --training-mode self-play

# Train against random opponent only
python run_project.py --training-mode vs-random
```

---

#### `--skip-training`
**Description:** Skip training phase, use pre-trained agents  
**Type:** Flag (no value needed)  
**Default:** False (training happens)

**Must also provide:** `--agent-dirs` with paths to trained agents

**Example:**
```bash
python run_project.py --skip-training --agent-dirs \
    results/training/q-learning_curriculum_20251111_123456 \
    results/training/sarsa_curriculum_20251111_123456
```

**Use cases:**
- Re-run evaluation with different settings (e.g., more games)
- Compare agents trained on different runs
- Generate new visualizations from existing agents

---

#### `--skip-evaluation`
**Description:** Skip evaluation phase, only train  
**Type:** Flag (no value needed)  
**Default:** False (evaluation happens)

**Example:**
```bash
python run_project.py --skip-evaluation --episodes 50000
```

**Use cases:**
- Long training runs where you'll evaluate later
- When you just want to generate trained agents
- Batch training multiple configurations

---

#### `--agent-dirs [PATH_LIST]`
**Description:** Paths to pre-trained agent directories  
**Type:** List of paths  
**Required with:** `--skip-training`  
**Default:** None

**Example:**
```bash
python run_project.py --skip-training --agent-dirs \
    results/training/q-learning_curriculum_20251111_123456 \
    results/training/sarsa_curriculum_20251111_123456 \
    results/training/monte-carlo_curriculum_20251111_123456
```

**Notes:**
- Each path should be a directory containing `parameters.json` and Q-table files
- Order doesn't matter
- Can mix agents from different training runs

---

## 🎮 ENVIRONMENT CONFIGURATION

These control the game board and reward structure.

#### `-r, --rows [INT]`
**Description:** Number of rows in the board  
**Type:** Integer  
**Default:** 6  
**Range:** 3-10 (practical)

**Examples:**
```bash
# Standard Connect Four (6 rows)
python run_project.py --rows 6

# Small board for fast testing (3 rows)
python run_project.py --rows 3

# Tall board (8 rows)
python run_project.py --rows 8
```

**Trade-offs:**
- **Smaller boards (3-4):** Fast training, limited strategy, good for testing
- **Standard (6-7):** Balanced complexity, standard Connect Four
- **Larger (8-10):** Deeper strategy, much longer training, huge state space

---

#### `-c, --columns [INT]`
**Description:** Number of columns in the board  
**Type:** Integer  
**Default:** 7  
**Range:** 3-10 (practical)

**Examples:**
```bash
# Standard Connect Four (7 columns)
python run_project.py --columns 7

# Narrow board (4 columns)
python run_project.py --columns 4

# Wide board (9 columns)
python run_project.py --columns 9
```

**Trade-offs:**
- **Fewer columns (3-5):** Fewer choices, faster learning
- **Standard (7):** Classic Connect Four
- **More columns (8-10):** More strategic options, longer training

---

#### `-n, --connect-n [INT]`
**Description:** How many pieces in a row needed to win  
**Type:** Integer  
**Default:** 4  
**Range:** 3 to min(rows, columns)

**Examples:**
```bash
# Standard Connect Four (4 in a row)
python run_project.py --connect-n 4

# Connect Three (easier to win)
python run_project.py --connect-n 3 --rows 3 --columns 4

# Connect Five (harder to win, needs bigger board)
python run_project.py --connect-n 5 --rows 6 --columns 7
```

**Trade-offs:**
- **Lower (3):** Easier to win, shorter games, simpler tactics
- **Standard (4):** Classic Connect Four balance
- **Higher (5+):** Harder to win, longer games, deeper strategy

---

#### `--reward [FLOAT]`
**Description:** Reward for winning a game  
**Type:** Float  
**Default:** 1.0  
**Range:** Any positive number (typically 0.1 to 10.0)

**Examples:**
```bash
# Standard reward
python run_project.py --reward 1.0

# Large reward (emphasizes winning)
python run_project.py --reward 10.0

# Small reward (subtle learning signal)
python run_project.py --reward 0.1
```

**When to change:**
- Larger rewards → Faster learning but potentially less stable
- Smaller rewards → More stable but slower convergence

---

#### `--penalty [FLOAT]`
**Description:** Penalty for losing or tying  
**Type:** Float  
**Default:** 0.0  
**Range:** Any negative number (typically -1.0 to 0.0)

**Examples:**
```bash
# No penalty (default)
python run_project.py --penalty 0.0

# Penalize losses
python run_project.py --penalty -1.0

# Small penalty
python run_project.py --penalty -0.1
```

**When to use:**
- `0.0` → Only positive reinforcement for wins
- `-1.0` → Symmetric rewards (win = +1, lose = -1)
- Negative penalties encourage agents to avoid losing

---

#### `--move-cost [FLOAT]`
**Description:** Cost incurred for each move  
**Type:** Float  
**Default:** 0.0  
**Range:** Any negative number (typically -0.1 to 0.0)

**Examples:**
```bash
# No move cost (default)
python run_project.py --move-cost 0.0

# Small cost per move (encourages faster wins)
python run_project.py --move-cost -0.01

# Larger cost per move
python run_project.py --move-cost -0.1
```

**When to use:**
- `0.0` → No preference for game length
- Negative → Encourages agents to win quickly
- Useful to prevent agents from "stalling"

---

## 🧠 TRAINING HYPERPARAMETERS

These control how the agents learn.

#### `--gamma [FLOAT]`
**Description:** Discount factor for future rewards  
**Type:** Float  
**Default:** 0.9  
**Range:** 0.0 to 1.0

**What it means:**
- **0.0:** Only care about immediate rewards (myopic)
- **0.5:** Balance short and long-term
- **0.9:** Strong preference for long-term rewards (default)
- **0.99:** Very long-term thinking
- **1.0:** Equal weight to all future rewards (no discounting)

**Examples:**
```bash
# Short-term thinking
python run_project.py --gamma 0.5

# Standard (default)
python run_project.py --gamma 0.9

# Very long-term thinking
python run_project.py --gamma 0.99
```

**Trade-offs:**
- **Lower (0.5-0.7):** Faster learning, more reactive play
- **Standard (0.9):** Good balance for most games
- **Higher (0.95-0.99):** Better strategic planning, slower learning

---

#### `--alpha [FLOAT]`
**Description:** Learning rate  
**Type:** Float  
**Default:** 0.1  
**Range:** 0.0 to 1.0

**What it means:**
- **0.0:** No learning (Q-values never change)
- **0.1:** Slow, stable learning (default)
- **0.5:** Moderate learning speed
- **1.0:** Immediate replacement (each experience completely overwrites old Q-value)

**Examples:**
```bash
# Very slow learning
python run_project.py --alpha 0.01

# Standard (default)
python run_project.py --alpha 0.1

# Fast learning
python run_project.py --alpha 0.5
```

**Trade-offs:**
- **Lower (0.01-0.1):** More stable, requires more episodes
- **Standard (0.1-0.3):** Good balance
- **Higher (0.5-1.0):** Fast adaptation, but can be unstable

**Common issue:** If alpha is too low AND you don't train enough episodes, agents won't learn enough.

---

#### `--epsilon [FLOAT]`
**Description:** Exploration rate (epsilon-greedy)  
**Type:** Float  
**Default:** 0.1  
**Range:** 0.0 to 1.0

**What it means:**
- **0.0:** Pure exploitation (always choose best known action)
- **0.1:** 10% random exploration, 90% exploitation (default)
- **0.5:** 50% random, 50% greedy
- **1.0:** Pure random (no learning used)

**Examples:**
```bash
# No exploration (pure exploitation)
python run_project.py --epsilon 0.0

# Standard (default)
python run_project.py --epsilon 0.1

# High exploration
python run_project.py --epsilon 0.3
```

**Trade-offs:**
- **Lower (0.0-0.05):** Exploits learned knowledge, might miss better strategies
- **Standard (0.1-0.2):** Good balance for most cases
- **Higher (0.3-0.5):** More exploration, slower convergence

**⚠️ Known Issue:** Current implementation has CONSTANT epsilon (no decay). Agents keep exploring randomly even after learning. **Phase 3 work will add epsilon decay.**

---

#### `--initial-val [FLOAT]`
**Description:** Initial value for all Q-values  
**Type:** Float  
**Default:** 0.0  
**Range:** Any float (typically -1.0 to 1.0)

**What it means:**
- **Pessimistic init (negative):** Assumes everything is bad until proven otherwise
- **Neutral init (0.0):** No bias (default)
- **Optimistic init (positive):** Encourages exploration (everything seems good initially)

**Examples:**
```bash
# Neutral (default)
python run_project.py --initial-val 0.0

# Optimistic initialization (encourages exploration)
python run_project.py --initial-val 1.0

# Pessimistic initialization
python run_project.py --initial-val -0.5
```

**When to use:**
- **0.0:** Standard, no bias
- **Positive (0.5-1.0):** Encourages exploration (agent tries everything)
- **Negative:** Discourages actions until proven good

---

#### `--episodes [INT]`
**Description:** Number of training episodes PER PHASE  
**Type:** Integer  
**Default:** 5000  
**Range:** 100 to 100,000+ (practical)

**What happens:**
- **Curriculum mode:** This is episodes per curriculum phase
  - Phase 1 (vs random): 2× this many episodes
  - Phase 2 (self-play): 2× this many episodes
  - Phase 3+ (vs checkpoints): 1× this many episodes each
- **Self-play/vs-random modes:** Total episodes = this value

**Examples:**
```bash
# Quick test
python run_project.py --episodes 100

# Standard
python run_project.py --episodes 5000

# Thorough training
python run_project.py --episodes 50000
```

**Time estimates (3×4 board):**
- 100 episodes: ~2-3 seconds per agent
- 1,000 episodes: ~5-10 seconds per agent
- 5,000 episodes: ~30-60 seconds per agent
- 10,000 episodes: ~1-2 minutes per agent
- 50,000 episodes: ~5-10 minutes per agent

**Trade-offs:**
- **Fewer (100-1000):** Fast, good for testing, might not converge
- **Standard (5000):** Usually sufficient for small boards
- **More (10000-50000):** Better convergence, but diminishing returns
- **⚠️ Current finding:** 5K → 50K shows minimal improvement (hyperparameters are the bottleneck, not training time)

---

#### `--curriculum-iterations [INT]`
**Description:** Number of additional training iterations against best checkpoint  
**Type:** Integer  
**Default:** 3  
**Range:** 0 to 10+ (practical)

**Only used in:** `curriculum` training mode

**What happens:**
After base curriculum phases (vs random, self-play, vs phase 1 checkpoint), agent trains N more times against its previous best checkpoint.

**Examples:**
```bash
# No extra iterations (just base 3 phases)
python run_project.py --curriculum-iterations 0

# Standard (default)
python run_project.py --curriculum-iterations 3

# Many iterations
python run_project.py --curriculum-iterations 10
```

**Total curriculum phases:**
- 0 iterations → 3 phases total
- 3 iterations → 6 phases total (default)
- 10 iterations → 13 phases total

**Trade-offs:**
- **Fewer (0-1):** Faster training, might not fully converge
- **Standard (3-5):** Good balance
- **More (10+):** Better refinement, but each iteration may show diminishing returns

---

## 📊 EVALUATION PARAMETERS

#### `--eval-games [INT]`
**Description:** Number of games to play per matchup during evaluation  
**Type:** Integer  
**Default:** 100  
**Range:** 10 to 1000+ (practical)

**What it affects:**
- Pairwise evaluations: Each A vs B matchup plays this many games
- Tournament: Each pairing plays this many games (can be hundreds of games total)

**Examples:**
```bash
# Quick evaluation
python run_project.py --eval-games 10

# Standard (default)
python run_project.py --eval-games 100

# High confidence results
python run_project.py --eval-games 500

# Publication quality
python run_project.py --eval-games 1000
```

**Time estimates:**
- 10 games: ~1 second per matchup
- 100 games: ~5-10 seconds per matchup
- 500 games: ~30-60 seconds per matchup
- 1000 games: ~1-2 minutes per matchup

**Total matchups for 3 agents:**
- Pairwise (bidirectional): 6 matchups
- Time = games_per_matchup × 6 × (seconds per game)

**Statistical significance:**
- **10-50 games:** Directional insights only
- **100 games:** Reasonable confidence
- **500+ games:** High confidence, tight error bars

---

## 📁 OUTPUT CONFIGURATION

#### `-o, --output-dir [PATH]`
**Description:** Base directory for all output files  
**Type:** String (path)  
**Default:** `results`

**Creates:**
```
[output-dir]/
├── training/          (Agent checkpoints)
├── evaluations/       (Results and plots)
└── SUMMARY_REPORT.txt (Text summary)
```

**Examples:**
```bash
# Default
python run_project.py --output-dir results

# Custom directory
python run_project.py --output-dir my_experiment_1

# Organize by date
python run_project.py --output-dir results/2025-11-11

# Organize by experiment
python run_project.py --output-dir experiments/epsilon_sweep/run_1
```

**Notes:**
- Directory will be created if it doesn't exist
- Timestamped subdirectories prevent overwriting
- Use meaningful names for experiments

---

## 🎯 COMPLETE EXAMPLES BY USE CASE

### Research & Development

#### Baseline Run (Standard Settings)
```bash
python run_project.py
```

#### Small Board for Fast Iteration
```bash
python run_project.py \
    --rows 3 --columns 4 --connect-n 3 \
    --episodes 1000 \
    --eval-games 50
```

#### Hyperparameter Grid Search (run multiple)
```bash
# Vary gamma
for gamma in 0.7 0.8 0.9 0.95 0.99; do
    python run_project.py --gamma $gamma --output-dir results/gamma_$gamma
done

# Vary epsilon
for eps in 0.05 0.1 0.2 0.3; do
    python run_project.py --epsilon $eps --output-dir results/epsilon_$eps
done

# Vary alpha
for alpha in 0.05 0.1 0.3 0.5; do
    python run_project.py --alpha $alpha --output-dir results/alpha_$alpha
done
```

#### Long Training Run (Overnight)
```bash
python run_project.py \
    --episodes 50000 \
    --curriculum-iterations 10 \
    --eval-games 500 \
    --output-dir results/long_run_$(date +%Y%m%d) \
    2>&1 | tee logs/long_run_$(date +%Y%m%d).log
```

---

### Testing & Debugging

#### Smoke Test (Is everything working?)
```bash
python run_project.py \
    --agents q-learning \
    --episodes 10 \
    --rows 3 --columns 3 --connect-n 3 \
    --eval-games 5
```

#### Test Single Algorithm
```bash
python run_project.py \
    --agents sarsa \
    --episodes 1000 \
    --skip-evaluation
```

#### Test Evaluation Only
```bash
# First, do a quick training
python run_project.py --episodes 100 --skip-evaluation

# Then test evaluation separately
python run_project.py \
    --skip-training \
    --agent-dirs results/training/* \
    --eval-games 50
```

---

### Publication-Quality Results

#### High-Confidence Standard Board
```bash
python run_project.py \
    --rows 6 --columns 7 --connect-n 4 \
    --episodes 20000 \
    --curriculum-iterations 5 \
    --eval-games 1000 \
    --output-dir publication_runs/standard_board \
    2>&1 | tee publication_runs/standard_board.log
```

#### Comprehensive Comparison Across Board Sizes
```bash
# 3×4
python run_project.py \
    --rows 3 --columns 4 --connect-n 3 \
    --episodes 10000 --eval-games 500 \
    --output-dir publication_runs/3x4 \
    2>&1 | tee publication_runs/3x4.log

# 4×5
python run_project.py \
    --rows 4 --columns 5 --connect-n 4 \
    --episodes 15000 --eval-games 500 \
    --output-dir publication_runs/4x5 \
    2>&1 | tee publication_runs/4x5.log

# 6×7 (standard)
python run_project.py \
    --rows 6 --columns 7 --connect-n 4 \
    --episodes 20000 --eval-games 500 \
    --output-dir publication_runs/6x7 \
    2>&1 | tee publication_runs/6x7.log
```

---

## ⚠️ Common Mistakes & Gotchas

### 1. Forgetting to Capture Console Output
**Problem:** Quality scores and outcome metrics only print to console, not saved.

**Solution:**
```bash
python run_project.py [...] 2>&1 | tee output.log
```

---

### 2. Using --skip-training Without --agent-dirs
**Problem:** Error message about missing agent directories

**Solution:**
```bash
python run_project.py --skip-training --agent-dirs path/to/agent1 path/to/agent2
```

---

### 3. Board Too Small for connect-n
**Problem:** `--connect-n 4` with `--rows 3` (impossible to get 4 in a row)

**Solution:** Ensure `connect-n ≤ min(rows, columns)`
```bash
python run_project.py --rows 3 --columns 4 --connect-n 3  # ✓ Correct
```

---

### 4. Epsilon = 0.1 Forever (No Decay)
**Problem:** Agents keep making 10% random moves even after learning

**Current Limitation:** Epsilon decay not yet implemented

**Workaround:** Lower epsilon manually
```bash
python run_project.py --epsilon 0.05  # Less exploration
```

**Future:** Phase 3 will add epsilon decay schedules

---

### 5. Too Few Episodes for Large Boards
**Problem:** 5000 episodes might not be enough for 6×7 board

**Solution:** Scale episodes with board size
- 3×4: 1000-5000 episodes
- 4×5: 5000-10000 episodes
- 6×7: 10000-20000 episodes

---

## 🔧 Advanced Tips

### Parallel Experiments
```bash
# Run multiple configurations in parallel (if you have multiple cores)
python run_project.py --gamma 0.7 --output-dir results/exp1 &
python run_project.py --gamma 0.9 --output-dir results/exp2 &
python run_project.py --gamma 0.99 --output-dir results/exp3 &
wait  # Wait for all to complete
```

### Timestamped Runs
```bash
timestamp=$(date +%Y%m%d_%H%M%S)
python run_project.py --output-dir results/run_$timestamp
```

### Environment Variable Configuration
```bash
# Set defaults via environment
export CONNECT4_EPISODES=10000
export CONNECT4_EVAL_GAMES=500

# Then just run
python run_project.py --episodes $CONNECT4_EPISODES --eval-games $CONNECT4_EVAL_GAMES
```

---

## 📊 Expected Output

Running this command:
```bash
python run_project.py --agents all --episodes 1000 --rows 3 --columns 4
```

Creates this output structure:
```
results/
├── training/
│   ├── q-learning_curriculum_20251111_123456/
│   │   ├── parameters.json
│   │   ├── red.npz
│   │   └── black.npz
│   ├── sarsa_curriculum_20251111_123456/
│   └── monte-carlo_curriculum_20251111_123456/
├── evaluations/
│   ├── pairwise_evaluations_20251111_123500/
│   │   ├── all_pairwise_results.json
│   │   └── plots/ (18 PNG files)
│   └── tournament_20251111_123530/
│       ├── tournament_results.json
│       └── plots/tournament_matrix.png
└── SUMMARY_REPORT.txt
```

Plus console output with:
- Training progress
- Outcome metrics
- Quality scores (0-100)
- Tournament rankings

---

## 🆘 Help & Troubleshooting

### Get Help
```bash
python run_project.py --help
```

### Verify Installation
```bash
python -c "import numpy, matplotlib, scipy; print('All dependencies installed')"
```

### Check Available Agents
```bash
ls results/training/
```

### Test Environment
```bash
python diagnostics.py
```

---

## 📚 Related Documentation

- **OUTPUT_FILES_GUIDE.md** - What files get created and how to read them
- **COMPLETE_ARCHITECTURE.md** - How all the Python files work together
- **QUICK_REFERENCE.md** - Meeting prep and one-liners
- **TECHNICAL_SUMMARY.md** - Technical deep dive

---

**Last Updated:** November 11, 2025  
**Compatible with:** run_project.py v1.0
