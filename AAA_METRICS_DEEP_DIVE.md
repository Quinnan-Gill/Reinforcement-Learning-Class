# Understanding the Analysis - Deep Dive into Metrics

**What This Guide Covers:** Every metric, score, and analysis output explained in plain English with examples

---

## 📊 Table of Contents

1. [Basic Win/Loss Statistics](#basic-winloss-statistics)
2. [Outcome-Specific Metrics](#outcome-specific-metrics)
3. [Move Quality Scores (0-100)](#move-quality-scores-0-100)
4. [Tactical Analysis](#tactical-analysis)
5. [Statistical Metrics](#statistical-metrics)
6. [Visualizations Explained](#visualizations-explained)
7. [Q-Table Statistics](#q-table-statistics)
8. [Interpreting Results](#interpreting-results)

---

## 1. BASIC WIN/LOSS STATISTICS

These are the simplest metrics - just counting outcomes.

### What You See:
```
Matchup Results: q-learning (red) vs sarsa (black)
  Red wins:   67 (67.0%)
  Black wins: 27 (27.0%)
  Ties:       6 (6.0%)
```

### What It Means:

**Win Rate:** Percentage of games won by each player
- **67% red wins** = Q-Learning won 67 out of 100 games when playing as red
- **27% black wins** = SARSA won 27 out of 100 games when playing as black
- **6% ties** = 6 games filled the entire board without a winner

### What's Good:
- **>60% win rate** = Strong dominance
- **50-60%** = Moderate advantage
- **45-55%** = Close matchup (roughly even)
- **<45%** = Clear disadvantage

### Important Note:
Always check BOTH directions (A as red vs B, and B as red vs A). Position matters in Connect Four! Playing first (red) often has an advantage.

---

## 2. OUTCOME-SPECIFIC METRICS

These tell you HOW agents win, lose, or tie.

### What You See:
```
OUTCOME-SPECIFIC METRICS

q-learning (Red) - Average Moves:
  Wins:   7.2 ± 1.8 moves
  Losses: 9.5 ± 2.1 moves
  Ties:   12.0 ± 0.0 moves

sarsa (Black) - Average Moves:
  Wins:   9.3 ± 2.0 moves
  Losses: 7.4 ± 1.9 moves
  Ties:   12.0 ± 0.0 moves
```

### Breaking It Down:

#### **Average Moves to Win**
**Example:** Q-Learning wins in 7.2 moves on average

**What this means:**
- When Q-Learning wins, it takes about 7-8 moves to victory
- Lower is better (faster, more efficient wins)
- Shows tactical efficiency

**Interpretation:**
- **6-8 moves** = Very efficient, finding quick wins
- **9-11 moves** = Moderate efficiency
- **12 moves** = Maximum possible (board filled)

---

#### **Average Moves to Lose**
**Example:** Q-Learning loses after 9.5 moves on average

**What this means:**
- When Q-Learning loses, it survives about 9-10 moves before opponent wins
- Higher is better (you made opponent work harder)
- Shows defensive capability

**Interpretation:**
- **High number (9-11)** = Resilient play, hard to beat
- **Low number (6-7)** = Opponent dominates quickly, weak defense
- If losses take LONGER than wins, it means: "I win fast, but when I lose, I fight hard"

---

#### **Average Moves to Tie**
**Example:** Both agents tie after 12.0 moves

**What this means:**
- Ties always happen at exactly 12 moves on a 3×4 board (rows × columns)
- Board is completely full with no winner
- Standard deviation of 0.0 means EVERY tie is exactly 12 moves

**Why this matters:**
- Validates that game logic is working correctly
- Should always = rows × columns

---

#### **Standard Deviation (± value)**
**Example:** Wins: 7.2 ± 1.8 moves

**What this means:**
- The "± 1.8" shows how much variation there is
- Most wins happen between 5.4 and 9.0 moves (7.2 - 1.8 to 7.2 + 1.8)
- Lower standard deviation = More consistent play

**Interpretation:**
- **Low std dev (0.5-1.0)** = Very consistent strategy
- **Medium std dev (1.5-2.5)** = Some variation in tactics
- **High std dev (3.0+)** = Inconsistent, chaotic play

---

### Real-World Example:

```
Agent A - Average Moves:
  Wins:   6.5 ± 1.2 moves
  Losses: 10.8 ± 1.5 moves

Agent B - Average Moves:
  Wins:   10.2 ± 1.8 moves
  Losses: 6.8 ± 1.3 moves
```

**What this tells us:**
- **Agent A wins quickly (6.5 moves)** = Aggressive, finds openings fast
- **Agent A loses slowly (10.8 moves)** = Defends well when losing
- **Agent B wins slowly (10.2 moves)** = Cautious, methodical
- **Agent B loses quickly (6.8 moves)** = Weak defense, gets dominated
- **Conclusion:** Agent A is the stronger player (fast wins, slow losses)

---

## 3. MOVE QUALITY SCORES (0-100)

This is a tactical "report card" for each agent.

### What You See:
```
MOVE QUALITY SCORES (0-100 scale)

q-learning (Red):
  Overall Score:       87.5 / 100
  Winning Move Score:  45.0 / 50
  Blocking Score:      42.5 / 50
  Blunder Rate:        4.2%
  Details: Won 18/20 opportunities, Blocked 17/20 threats, 3 blunders in 72 moves
```

### The 0-100 Scale:

**Overall Score = Winning Move Score + Blocking Score**

#### Grade Scale:
- **90-100:** Excellent (A) - Elite tactical play
- **75-89:** Good (B) - Solid tactical awareness
- **60-74:** Moderate (C) - Competent but makes mistakes
- **40-59:** Poor (D) - Misses many opportunities
- **<40:** Very Poor (F) - Needs significant improvement

---

### Component 1: Winning Move Score (0-50 points)

**What it measures:** Does the agent take winning moves when available?

**Example:** 45.0 / 50 = 90% accuracy

**How it's calculated:**
```
Winning Move Score = (Wins Taken / Win Opportunities) × 50

Example:
- Agent had 20 chances to win immediately
- Agent took 18 of those chances
- Score = (18/20) × 50 = 45.0 points
```

**What "taking a winning move" means:**
- There's a column where playing gives you 4-in-a-row (instant win)
- Agent plays that column ✓

**Missing a winning move (a blunder):**
- There's a winning move available
- Agent plays a different column that doesn't win ✗

**Interpretation:**
- **48-50 points** = Nearly perfect (96-100% accuracy)
- **45-47 points** = Excellent (90-94% accuracy)
- **40-44 points** = Good (80-88% accuracy)
- **30-39 points** = Moderate (60-78% accuracy)
- **<30 points** = Poor (<60% accuracy)

---

### Component 2: Blocking Score (0-50 points)

**What it measures:** Does the agent block opponent's winning moves?

**Example:** 42.5 / 50 = 85% accuracy

**How it's calculated:**
```
Blocking Score = (Blocks Made / Block Opportunities) × 50

Example:
- Opponent had 20 chances to win on their next move
- Agent blocked 17 of those threats
- Score = (17/20) × 50 = 42.5 points
```

**What "blocking" means:**
- Opponent has a column where playing would give them 4-in-a-row
- Agent plays that column FIRST to prevent opponent's win ✓

**Missing a block (a blunder):**
- Opponent threatens to win next move
- Agent plays elsewhere
- Opponent wins on next turn ✗

**Interpretation:**
- **48-50 points** = Nearly perfect defense
- **45-47 points** = Excellent defense
- **40-44 points** = Good defense
- **30-39 points** = Moderate defense (misses some threats)
- **<30 points** = Poor defense (frequently lets opponent win)

---

### Blunder Rate

**What it measures:** Percentage of moves that are mistakes

**Example:** 4.2% = 3 blunders out of 72 total moves

**What counts as a blunder:**
1. **Missed winning move** - You could have won but played elsewhere
2. **Missed block** - Opponent threatens to win, you don't block

**How it's calculated:**
```
Blunder Rate = (Missed Wins + Missed Blocks) / Total Moves

Example:
- 2 missed winning moves
- 1 missed block
- 72 total moves made
- Blunder Rate = 3/72 = 4.2%
```

**Interpretation:**
- **0-5%** = Excellent (very few mistakes)
- **5-10%** = Good (occasional mistakes)
- **10-20%** = Moderate (frequent mistakes)
- **20-30%** = Poor (makes many mistakes)
- **>30%** = Very poor (mistake-prone)

---

### The "Details" Line Explained:

```
Details: Won 18/20 opportunities, Blocked 17/20 threats, 3 blunders in 72 moves
```

**Won 18/20 opportunities:**
- Agent faced 20 situations where a winning move existed
- Agent correctly took the winning move 18 times
- Missed 2 winning moves (blunders)

**Blocked 17/20 threats:**
- Opponent threatened to win 20 times
- Agent blocked 17 of those threats
- Missed 3 blocks (blunders, but only 1 counted in total since some overlapped)

**3 blunders in 72 moves:**
- Agent made 72 moves total across all games
- 3 of those moves were tactical mistakes
- Blunder rate = 3/72 = 4.2%

---

### Real-World Example:

**Strong Agent:**
```
Overall Score:       92.5 / 100
Winning Move Score:  47.5 / 50  (19/20 opportunities = 95%)
Blocking Score:      45.0 / 50  (18/20 threats = 90%)
Blunder Rate:        2.8%
```
**Interpretation:** Elite tactical play. Rarely misses opportunities or threats.

**Weak Agent:**
```
Overall Score:       52.5 / 100
Winning Move Score:  25.0 / 50  (10/20 opportunities = 50%)
Blocking Score:      27.5 / 50  (11/20 threats = 55%)
Blunder Rate:        28.3%
```
**Interpretation:** Poor tactical play. Misses half of winning moves and blocks. Needs more training.

---

## 4. TACTICAL ANALYSIS

This is the detailed move-by-move breakdown used to compute quality scores.

### What Game Analyzer Does:

For every game, it replays the entire game and checks EACH move:

#### For Each Move, It Asks:

1. **Was there a winning move available?**
   - Simulate each column
   - Check if any column results in 4-in-a-row
   - If yes, mark as "winning move available"

2. **Did the agent take the winning move?**
   - If winning move existed AND agent played that column → ✓ Winning move taken
   - If winning move existed AND agent played elsewhere → ✗ Missed winning move (BLUNDER)

3. **Did opponent threaten to win next turn?**
   - Simulate opponent's next turn in each column
   - Check if opponent could win
   - If yes, mark as "blocking opportunity"

4. **Did the agent block the threat?**
   - If threat existed AND agent blocked it → ✓ Block made
   - If threat existed AND agent didn't block → ✗ Missed block (BLUNDER)

---

### Example Game Walkthrough:

**3×4 Connect-3 Game (simplified):**

```
Move 1: Red plays column 1
  - No winning moves available yet
  - No threats to block yet
  - Result: Normal move

Move 2: Black plays column 2
  - No winning moves available yet
  - No threats to block yet
  - Result: Normal move

Move 3: Red plays column 1
  - No winning moves available yet
  - No threats to block yet
  - Result: Normal move

Move 4: Black plays column 2
  - No winning moves available yet
  - Red threatens column 1 (3 in a row vertically)
  - Black should block column 1
  - Black played column 2 instead
  - Result: MISSED BLOCK (blunder)

Move 5: Red plays column 1
  - Red has winning move in column 1 (3 vertical)
  - Red played column 1
  - Result: WINNING MOVE TAKEN ✓
  - Game Over: Red wins
```

**Scoring for this game:**
- **Red:** 1 winning move taken, 0 missed
- **Black:** 0 blocks made, 1 missed block
- **Red gets points, Black loses points for the blunder**

---

### Move-by-Move Quality

**What this means:** How good was each individual move?

**Categories:**
1. **Perfect moves:** Took winning move OR made crucial block
2. **Good moves:** Safe strategic play (no immediate threats/opportunities)
3. **Neutral moves:** Not harmful but not helpful
4. **Bad moves:** Missed winning move or missed block (BLUNDERS)

---

### Blunders Explained in Detail:

#### Type 1: Missed Winning Move
```
Board state:
  0 1 2 3
 ┌─────────┐
 │ . . . . │ 2
 │ X X . . │ 1  
 │ X O O O │ 0
 └─────────┘

Red's turn:
- Column 2 gives 3 X's vertically = WIN
- Red plays column 3 instead
- Result: MISSED WINNING MOVE (blunder)
```

**Why this is bad:**
- Had guaranteed win
- Gave opponent another chance
- Opponent might win instead

---

#### Type 2: Missed Block
```
Board state:
  0 1 2 3
 ┌─────────┐
 │ . . . . │ 2
 │ O O . . │ 1  
 │ X O X X │ 0
 └─────────┘

Red's turn:
- Black threatens column 2 (3 O's horizontally)
- Red should play column 2 to block
- Red plays column 3 instead
- Black wins next turn by playing column 2
- Result: MISSED BLOCK (blunder)
```

**Why this is bad:**
- Didn't defend against immediate threat
- Opponent wins on next turn
- Game lost due to missed block

---

### Real Game Example:

A game between two agents might have:
- **20 total moves** (10 per player)
- **2 winning move opportunities** (1 per player)
- **Both agents took their winning moves** ✓
- **3 blocking opportunities** 
- **2 blocks made** ✓
- **1 block missed** ✗ (blunder)
- **Final blunder count:** 1
- **Blunder rate:** 1/20 = 5%

---

## 5. STATISTICAL METRICS

These help you know if differences are real or just random luck.

### Confidence Intervals

**What You See:**
```
Red wins: 67 [61.2, 72.8] (95% CI)
```

**What it means:**
- We're 95% confident the TRUE win rate is between 61.2% and 72.8%
- The wider the interval, the less certain we are
- More games = narrower intervals (more confidence)

**How to interpret:**
```
Agent A wins: 65% [60%, 70%]
Agent B wins: 35% [30%, 40%]
```
- Intervals don't overlap
- Clear difference: Agent A is genuinely better

```
Agent A wins: 52% [47%, 57%]
Agent B wins: 48% [43%, 53%]
```
- Intervals overlap significantly
- Difference might be random luck
- Need more games to know for sure

---

### Effect Size (Cohen's h)

**What it measures:** HOW MUCH better one agent is

**Scale:**
- **< 0.2:** Small effect (barely noticeable)
- **0.2 - 0.5:** Medium effect (clear difference)
- **> 0.5:** Large effect (huge difference)

**Example:**
```
Agent A vs Agent B:
  Win rates: 70% vs 30%
  Cohen's h: 0.85 (Large effect)
  
Interpretation: Agent A is MUCH better (not just lucky)
```

---

### Statistical Significance (p-value)

**What it measures:** Is this difference real or random chance?

**p-value < 0.05 = Statistically significant**
- Less than 5% chance this difference is random luck
- We can confidently say one agent is better

**Example:**
```
Two-Proportion Z-Test: p=0.0023

Interpretation: 
- Only 0.23% chance the difference is random
- Highly significant
- One agent is genuinely better
```

---

## 6. VISUALIZATIONS EXPLAINED

### 1. Win Rates Bar Chart

**What it shows:**
Three bars showing red wins, black wins, and ties with error bars.

**How to read it:**
- **Taller bar** = Higher win rate
- **Error bars** = Confidence intervals
- **Bars that don't overlap** = Clear winner
- **Overlapping bars** = Close matchup

**What good looks like:**
- One agent's bar clearly taller with minimal overlap

**What bad looks like:**
- All bars roughly same height (no agent is better)

---

### 2. Game Lengths Distribution

**What it shows:**
Histogram of how many moves games took.

**How to read it:**
- **X-axis:** Number of moves (6, 7, 8, ... 12)
- **Y-axis:** How many games had that length
- **Three colors:** Red wins, Black wins, Ties

**Patterns to look for:**

**Short games cluster (6-8 moves):**
- Quick decisive wins
- Strong tactical play
- One agent dominates

**Long games cluster (10-12 moves):**
- Back-and-forth tactical play
- Both agents defensive
- Close matchups

**Spike at max length (12 moves):**
- Many ties
- Board fills without winner
- Both agents play conservatively

---

### 3. Win Rate Over Time

**What it shows:**
Line graph of win rate across games in sequence.

**How to read it:**
- **X-axis:** Game number (1, 2, 3, ... 100)
- **Y-axis:** Win rate (0% to 100%)
- **Line:** Rolling average

**Patterns:**

**Flat line:**
```
Win rate: ───────────────  (stays at 65%)
```
- Stable, consistent performance
- Good sign (no drift)

**Upward trend:**
```
Win rate: ─────────┌─────  (starts 50%, ends 75%)
```
- One agent improving during evaluation
- Might indicate online learning (shouldn't happen)
- Or random early games

**Downward trend:**
```
Win rate: ─────┐  (starts 75%, ends 50%)
          └─────────
```
- One agent degrading
- Concerning (shouldn't happen)

**High variance:**
```
Win rate: ─┐ ┌─┐ ┌─┐ ┌─  (zigzags up and down)
          └─┘ └─┘ └─┘
```
- Inconsistent play
- High randomness
- Might need more evaluation games

---

### 4. Opening Moves Heatmap

**What it shows:**
Which columns agents prefer for their first move.

**How to read it:**
- **Darker color** = More frequently used
- **Lighter color** = Rarely used
- **One heatmap per agent**

**What good looks like:**
```
Column:  0    1    2    3
        [██] [██] [███] [██]  (prefers center)
```
- Center columns (1-2) preferred
- Strategic play

**What bad looks like:**
```
Column:  0    1    2    3
        [███] [█] [█] [█]  (prefers edge)
```
- Edge columns (0, 3) preferred
- Weak strategy

**What random looks like:**
```
Column:  0    1    2    3
        [██] [██] [██] [██]  (uniform)
```
- All columns equally likely
- No learned preference

---

### 5. Tactical Accuracy Chart

**What it shows:**
Bar chart comparing winning move % and blocking %.

**How to read it:**
- **Two bars per agent**
- **Left bar:** Winning move accuracy
- **Right bar:** Blocking accuracy
- **Higher is better**

**Example:**
```
Agent A: [████████░] 90% wins  [███████░░] 80% blocks
Agent B: [████░░░░░] 50% wins  [█████░░░░] 55% blocks
```

**Interpretation:**
- Agent A: Strong offense (90%) AND defense (80%)
- Agent B: Weak both ways (needs training)

---

### 6. Column Usage Heatmap

**What it shows:**
How often each agent plays each column (across all moves, not just opening).

**How to read it:**
- **Darker** = Played more often
- **Lighter** = Played less often

**What good looks like:**
```
Column:  0    1    2    3
        [██] [███] [███] [██]  (center-heavy)
```
- Center columns used more
- Strategic positioning

**What suboptimal looks like:**
```
Column:  0    1    2    3
        [████] [█] [█] [██]  (edge-heavy)
```
- Edge columns overused
- Poor strategy

---

## 7. Q-TABLE STATISTICS

These metrics analyze the learned Q-tables (not printed by default).

### States Visited

**What it measures:** How many unique board states the agent encountered

**Example:**
```
States Visited: 2,847
```

**What this means:**
- Agent saw 2,847 different board configurations during training
- **More states** = More exploration
- **Fewer states** = Limited exploration

**Interpretation:**
- **Small board (3×4):** 500-5000 states typical
- **Large board (6×7):** 10,000-100,000+ states possible
- **Sparsity = (1 - states_visited / total_possible_states)**

---

### Mean Q-Value

**What it measures:** Average of all Q-values in the table

**Example:**
```
Mean Q-value: 0.234
```

**What this means:**
- Average expected reward across all state-action pairs
- Positive = Optimistic (expects rewards)
- Negative = Pessimistic (expects penalties)
- Near zero = Neutral

**Interpretation:**
- **0.1 to 0.5:** Normal for win reward = 1.0
- **Near 0.0:** Limited learning or symmetric game
- **>0.5:** Might indicate positive initial values

---

### Q-Value Range

**Example:**
```
Q-value Range: [-0.245, 0.856]
```

**What this means:**
- **Min (-0.245):** Worst expected outcome
- **Max (0.856):** Best expected outcome
- **Spread:** Shows diversity in learned values

**Interpretation:**
- **Narrow range (0.1):** Limited learning
- **Medium range (0.5-1.0):** Normal learning
- **Wide range (>1.5):** Might indicate instability

---

### Sparsity

**What it measures:** Percentage of Q-table that's still at initial values (zeros)

**Example:**
```
Sparsity: 78.5%
```

**What this means:**
- 78.5% of state-action pairs were never visited
- Only 21.5% of Q-table was actually updated
- Shows coverage of state space

**Interpretation:**
- **High sparsity (>90%):** Very limited exploration
- **Medium sparsity (50-90%):** Typical for large boards
- **Low sparsity (<50%):** Good exploration, smaller boards

---

## 8. INTERPRETING RESULTS

### Putting It All Together

When you get your results, look at these in order:

#### Step 1: Basic Win Rates
```
Q-Learning wins: 67%
SARSA wins: 27%
```
**Quick take:** Q-Learning is better

---

#### Step 2: Check Position Advantage
```
When Q-Learning is red: 67% win rate
When Q-Learning is black: 58% win rate
```
**Note:** Playing first (red) gives 9% advantage

---

#### Step 3: Look at Outcome Metrics
```
Q-Learning wins in 7.2 moves (fast)
Q-Learning loses in 9.5 moves (fights hard)
```
**Conclusion:** Aggressive attacker, resilient defender

---

#### Step 4: Check Quality Scores
```
Q-Learning: 87.5 / 100 (excellent)
SARSA: 68.3 / 100 (moderate)
```
**Explanation:** Q-Learning has much better tactical awareness

---

#### Step 5: Look at Blunder Rates
```
Q-Learning: 4.2% blunders
SARSA: 11.7% blunders
```
**Explanation:** SARSA makes nearly 3× more mistakes

---

#### Step 6: Statistical Confidence
```
Confidence Interval: [61%, 73%]
p-value: 0.0001
Cohen's h: 0.78 (Large)
```
**Conclusion:** Difference is real, not luck. Q-Learning is genuinely superior.

---

### Example Full Analysis:

```
MATCHUP: Q-Learning vs SARSA (100 games)

Win Rates:
  Q-Learning (red): 67%
  SARSA (black): 27%
  Ties: 6%

Outcome Metrics:
  Q-Learning wins: 7.2 moves (efficient)
  Q-Learning losses: 9.5 moves (resilient)
  SARSA wins: 9.3 moves (slower)
  SARSA losses: 7.4 moves (gets dominated)

Quality Scores:
  Q-Learning: 87.5/100 (excellent)
    - Wins: 45/50 (18/20 opportunities)
    - Blocks: 42.5/50 (17/20 threats)
    - Blunders: 4.2%
  
  SARSA: 68.3/100 (moderate)
    - Wins: 32.5/50 (13/20 opportunities)
    - Blocks: 35.8/50 (14/20 threats)
    - Blunders: 11.7%

Statistics:
  Cohen's h: 0.78 (Large effect)
  p-value: <0.001 (Highly significant)

CONCLUSION:
Q-Learning is significantly better than SARSA in this matchup.
- Wins more often (67% vs 27%)
- Wins faster (7.2 vs 9.3 moves)
- Better tactics (87.5 vs 68.3 quality score)
- Makes fewer mistakes (4.2% vs 11.7% blunders)
- Difference is statistically significant, not luck

Recommendation: Use Q-Learning for this board configuration.
```

---

## Quick Reference Guide

### Quality Score Grading:
- **90-100:** A (Excellent)
- **75-89:** B (Good)
- **60-74:** C (Moderate)
- **40-59:** D (Poor)
- **<40:** F (Very Poor)

### Blunder Rate Grading:
- **0-5%:** Excellent
- **5-10%:** Good
- **10-20%:** Moderate
- **20-30%:** Poor
- **>30%:** Very Poor

### Win Rate Interpretation:
- **>60%:** Strong dominance
- **50-60%:** Moderate advantage
- **45-55%:** Even matchup
- **<45%:** Clear disadvantage

### Effect Size (Cohen's h):
- **<0.2:** Small
- **0.2-0.5:** Medium
- **>0.5:** Large

---

## Common Questions

### Q: Why does an agent with 67% win rate only have a 52.5 quality score?

**A:** Win rate and quality score measure different things:
- **Win rate** = Overall outcomes (did you win?)
- **Quality score** = Tactical execution (did you play well?)

An agent can win frequently due to:
- Better learning algorithm
- More training episodes
- Better hyperparameters

But still have mediocre tactics (missing opportunities).

---

### Q: My agent has 85 quality score but only 45% win rate. Why?

**A:** Possible explanations:
1. **Position disadvantage:** Playing as black (second) is harder
2. **Opponent is better overall:** Good tactics don't guarantee wins against superior strategy
3. **Sample size:** Maybe just bad luck in this sample
4. **Evaluation issues:** Check if games are fair

---

### Q: What's the difference between "blunder rate" and "missed opportunities"?

**A:** Same concept, different framing:
- **Blunder rate** = (mistakes / total moves)
- **Missed opportunities** = Count of specific mistakes
- Blunder rate is a percentage, missed opportunities is a count

---

### Q: Why are ties always exactly 12 moves on a 3×4 board?

**A:** Ties happen when:
- Board is completely filled (all 12 squares)
- No player achieved 3-in-a-row
- Therefore, ties always = rows × columns

If you see ties with different move counts, that's a BUG in the game logic!

---

### Q: What's a "good" sparsity for Q-tables?

**A:** Depends on board size:
- **Small boards (3×4):** 50-80% sparsity is fine
- **Medium boards (4×5):** 70-90% sparsity expected
- **Large boards (6×7):** 95%+ sparsity normal

High sparsity isn't bad—it means you didn't explore useless states.

---

## Further Reading

- **OUTPUT_FILES_GUIDE.md** - Where these metrics are saved
- **RUN_PROJECT_HANDBOOK.md** - How to run evaluations
- **TECHNICAL_SUMMARY.md** - Algorithm details

---

**Bottom Line:** Quality scores tell you HOW WELL agents play tactically. Win rates tell you WHO WINS MORE. Both matter, but they measure different things!
