"""
Perspective-normalized state utilities for single Q-table RL.

Key concept: Always encode the board from the current player's perspective:
  +1 = current player's pieces
  -1 = opponent's pieces
   0 = empty

This ensures Red and Black see identical states in symmetric positions,
enabling a single Q-table to learn both roles simultaneously.
"""

import numpy as np
from typing import Tuple

def normalize_state_perspective(board: np.ndarray, current_player: int) -> str:
    """
    Convert board to perspective-normalized state key.
    
    Args:
        board: Raw board array where Red=+1, Black=-1, Empty=0
        current_player: +1 (Red) or -1 (Black)
    
    Returns:
        State key string where +1 always means "player's [vs opponent] pieces"
    
    Example:
        Raw board (Red to move):     [+1, -1,  0]
        Normalized:                  [+1, -1,  0]  (no change)
        
        Same board (Black to move):  [+1, -1,  0]
        Normalized:                  [-1, +1,  0]  (flipped signs)
        
        Both positions are "Player has piece at 0, opponent at 1" → SAME STATE KEY
    """
    if current_player == 1:
        # Red's perspective: board is already in correct orientation
        normalized = board
    else:
        # Black's perspective: flip all piece signs
        # +1 (Red pieces) → -1 (opponent pieces)
        # -1 (Black pieces) → +1 (my pieces)
        normalized = -board
    
    # Convert to hashable string
    return ''.join(str(int(x)) for x in normalized.flatten())


def denormalize_state_perspective(state_key: str, board_shape: Tuple[int, int], 
                                  current_player: int) -> np.ndarray:
    """
    Convert perspective-normalized state key back to raw board.
    
    Useful for debugging and visualization.
    
    Args:
        state_key: Normalized state string
        board_shape: (rows, cols) tuple
        current_player: +1 (Red) or -1 (Black)
    
    Returns:
        Raw board array where Red=+1, Black=-1
    """
    # Parse state key
    normalized = np.array([int(c) if c != '-' else -1 
                          for c in state_key.replace('-1', '-')], dtype=np.int8)
    normalized = normalized.reshape(board_shape)
    
    if current_player == 1:
        return normalized
    else:
        # Black's perspective was flipped, so flip back
        return -normalized


def test_perspective_normalization():
    """Test that perspective normalization works correctly."""
    
    # Test case 1: Empty board is same for both players
    board = np.zeros((3, 3), dtype=np.int8)
    red_state = normalize_state_perspective(board, 1)
    black_state = normalize_state_perspective(board, -1)
    assert red_state == black_state, "Empty board should be identical"
    print("✓ Test 1 passed: Empty board")
    
    # Test case 2: Symmetric position should produce same state
    board = np.array([
        [1, -1, 0],
        [0,  0, 0],
        [0,  0, 0]
    ], dtype=np.int8)
    
    # Red to move: sees own piece at (0,0), opponent at (0,1)
    red_state = normalize_state_perspective(board, 1)
    
    # Flip the board for Black's symmetric position
    flipped_board = np.array([
        [-1, 1, 0],
        [0,  0, 0],
        [0,  0, 0]
    ], dtype=np.int8)
    
    # Black to move: sees own piece at (0,0), opponent at (0,1)
    black_state = normalize_state_perspective(flipped_board, -1)
    
    assert red_state == black_state, "Symmetric positions should match"
    print("✓ Test 2 passed: Symmetric positions")
    
    # Test case 3: Different positions should produce different states
    board1 = np.array([[1, 0, 0]], dtype=np.int8)
    board2 = np.array([[0, 1, 0]], dtype=np.int8)
    
    state1 = normalize_state_perspective(board1, 1)
    state2 = normalize_state_perspective(board2, 1)
    
    assert state1 != state2, "Different positions should differ"
    print("✓ Test 3 passed: Different positions")
    
    # Test case 4: Roundtrip conversion
    original_board = np.array([
        [1, -1, 0],
        [1,  0, -1],
        [0,  0, 0]
    ], dtype=np.int8)
    
    for player in [1, -1]:
        state_key = normalize_state_perspective(original_board, player)
        reconstructed = denormalize_state_perspective(state_key, (3, 3), player)
        assert np.array_equal(original_board, reconstructed), f"Roundtrip failed for player {player}"
    
    print("✓ Test 4 passed: Roundtrip conversion")
    
    print("\n✅ All perspective normalization tests passed!")


if __name__ == '__main__':
    test_perspective_normalization()
