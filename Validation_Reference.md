# Connect Four Environment Validation Reference

## Validation Mechanisms Summary

| **Validation** | **Method(s)** | **What It Prevents** |
|---|---|---|
| Column bounds | `is_valid_action()` | Out-of-range columns |
| Full columns | `get_valid_actions()`, `is_valid_action()` | Playing in full column |
| Gravity | `_get_lowest_empty_row()` | Floating pieces |
| Game over | `make_move()` check | Moves after game ends |
| Invalid moves | `make_move()` validation | Any rejected move modifying state |
| Turn order | Automatic player toggle | Same player twice |
| Clean start | `reset()`, `__init__()` | Corrupted initial state |
| Config validation | `__init__()` check | Impossible win conditions |
| Win detection | `_check_win()` | False positives/negatives |
| Draw detection | Valid actions check | Infinite full boards |
| State immutability | `.copy()` returns | External state corruption |
| Move tracking | `move_history` | Lost game context |

---

## Error Handling Reference

| **Scenario** | **Method** | **Returns** | **Board Modified?** |
|---|---|---|---|
| Out of bounds column | `make_move(99)` | `(board_copy, 0, False, {"error": "Invalid action"})` | ❌ No |
| Full column | `make_move(0)` when full | `(board_copy, 0, False, {"error": "Invalid action"})` | ❌ No |
| Move after game over | `make_move(col)` | `(board_copy, 0, True, {"error": "Game already over"})` | ❌ No |
| Impossible config | `ConnectFourEnv(3, 3, 5)` | Raises `ValueError` | N/A - never created |
| Valid move | `make_move(col)` | `(board_copy, reward, done, {"move": (r,c), ...})` | ✅ Yes |
| Check full column | `is_valid_action(0)` when full | `False` | ❌ No |
| Get valid actions | `get_valid_actions()` | List of valid columns (may be empty) | ❌ No |

---

## Key Principles

**Defense in Depth:** Multiple layers of protection ensure valid game states.

1. **Entry validation** - Check inputs before processing
2. **Physics enforcement** - API design makes invalid states impossible
3. **State protection** - Never expose mutable internal state
4. **Automatic management** - Turn switching, game end detection are automatic
5. **Configuration validation** - Catch impossible setups at creation

**Critical insight:** Invalid operations never modify the board - they return errors while maintaining valid state.

---

## Quick Reference: Core Methods

- **`make_move(col)`** - Only way to add pieces; enforces all rules
- **`get_valid_actions()`** - Returns list of legal columns
- **`is_valid_action(col)`** - Checks if specific move is legal
- **`reset()`** - Returns to clean initial state
- **`_get_lowest_empty_row(col)`** - Enforces gravity (internal)
- **`_check_win(row, col)`** - Validates win conditions (internal)

---

*Generated for Connect Four Environment validation documentation*