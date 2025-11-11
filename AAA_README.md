# Reinforcement-Learning-Class

# Connect Four Environment for Reinforcement Learning

A configurable Connect Four game environment designed for RL experiments with variable board sizes.

## Quick Start

```python
from connect_four_env import ConnectFourEnv

# Create 4×3 Connect Three environment (Phase 1)
env = ConnectFourEnv(rows=3, cols=4, connect_n=3)

# Play a game
state = env.reset()
done = False

while not done:
    # Get valid moves
    valid_actions = env.get_valid_actions()
    
    # Agent selects action (or random for testing)
    action = valid_actions[0]
    
    # Execute move
    next_state, reward, done, info = env.make_move(action)
    
    # Visualize
    env.render()
```

## Environment Specifications

### State Space
- **Representation**: NumPy array (rows × cols)
  - `0` = empty position
  - `1` = Player 1 (X)
  - `-1` = Player 2 (O)
- **State Key**: Hashable string for Q-tables via `get_state_key()`

### Action Space
- **Format**: Integer from 0 to (cols-1) representing column selection
- **Validation**: Only non-full columns are valid actions
- **Access**: `env.get_valid_actions()` returns list of valid columns

### Rewards
- **Win**: +1.0
- **Loss**: -1.0
- **Draw**: 0.0
- **Ongoing**: 0.0

### Game Mechanics
- Pieces drop to lowest available position (gravity)
- Players alternate turns (Player 1 starts)
- Win condition: `connect_n` pieces in a row (horizontal/vertical/diagonal)
- Draw condition: Board full with no winner

## Board Configurations

| Configuration | Rows | Cols | Connect-N | States | Use Case |
|--------------|------|------|-----------|---------|----------|
| Tiny | 2 | 2 | 2 | 13 | Testing |
| Phase 1 | 3 | 4 | 3 | 7,157 | Tabular RL |
| Phase 2 | 4 | 4 | 3 | 41,750 | Scaling analysis |
| Large | 4 | 5 | 4 | 3,945,711 | Too large for tabular |
| Standard | 6 | 7 | 4 | ~4.5B (estimated) | Deep RL (Phase 3) |

## API Reference

### Core Methods

**`reset() -> np.ndarray`**
- Resets environment to initial empty state
- Returns: Initial board state

**`make_move(col: int) -> Tuple[state, reward, done, info]`**
- Executes move in specified column
- Returns: (next_state, reward, done, info_dict)

**`get_valid_actions() -> List[int]`**
- Returns list of non-full column indices

**`render(mode='human')`**
- Displays current board state
- mode='human': prints to console
- mode='string': returns string representation

**`get_state_key() -> str`**
- Converts board to hashable string for Q-table indexing
- Format: "board_flattened_currentplayer"

**`copy() -> ConnectFourEnv`**
- Creates deep copy of environment (useful for lookahead/planning)

## State Space Analysis

Actual reachable states (from diagnostics):
- 4×3 Connect-3: 7,157 states
- 4×4 Connect-3: 41,750 states
- 4×5 Connect-4: 3,945,711 states

See `diagnostics.py` for state space enumeration and validation tests.

## Example: Training Loop Structure

```python
# Pseudocode for RL training
env = ConnectFourEnv(rows=3, cols=4, connect_n=3)
q_table = {}

for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        state_key = env.get_state_key()
        
        # Agent selects action (ε-greedy, etc.)
        valid_actions = env.get_valid_actions()
        action = select_action(state_key, valid_actions, q_table)
        
        # Environment step
        next_state, reward, done, info = env.make_move(action)
        
        # Q-learning update
        # ... update q_table ...
        
        state = next_state
```

## Implementation Notes

- **Player perspective**: Rewards are always from current player's perspective (switch sign for opponent)
- **Action masking**: Always constrain to valid actions - never allow invalid moves
- **State hashing**: Use `get_state_key()` for Q-table indexing (includes current player)
- **Deterministic**: No randomness in environment (gravity, win detection are deterministic)

## Testing

Run diagnostics to validate environment:
```bash
python diagnostics.py
```

Tests include:
- Reset functionality
- Move mechanics and gravity
- Win detection (all directions)
- Invalid move handling
- State space enumeration
- Terminal state analysis

## Files

- `connect_four_env.py` - Core environment implementation
- `diagnostics.py` - Validation tests and state space analysis
