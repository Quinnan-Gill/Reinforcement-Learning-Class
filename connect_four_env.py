"""
Connect Four Environment
Supports configurable board sizes for reinforcement learning experiments
"""

import numpy as np
from typing import List, Tuple, Optional


class ConnectFourEnv:
    """
    Connect Four game environment for reinforcement learning.
    
    Args:
        rows: Number of rows in the board
        cols: Number of columns in the board
        connect_n: Number of pieces needed in a row to win 
                  (default: min(4, cols, rows) - automatically adjusts for small boards)
    """
    
    def __init__(
            self,
            rows: int = 3,
            cols: int = 4,
            connect_n: Optional[int] = None,
            reward: int=1.0,
            penalty: int=0.0,
            move_cost: int=0.0,
        ):

        self.rows = rows
        self.cols = cols
        self.connect_n = connect_n if connect_n is not None else min(4, cols, rows)

        self.reward = reward
        self.penalty = penalty
        self.move_cost = move_cost
        
        # Validate configuration
        if self.connect_n > max(rows, cols):
            raise ValueError(f"connect_n ({self.connect_n}) cannot exceed max(rows, cols)")
        
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.current_player = 1  # 1 or -1
        self.game_over = False
        self.winner = None
        self.move_history = []
        
    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        return self.board.copy()
    
    def get_valid_actions(self) -> List[int]:
        """Return list of valid column indices (columns not full)."""
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def is_valid_action(self, col: int) -> bool:
        """Check if a column is a valid move."""
        return 0 <= col < self.cols and self.board[0, col] == 0
    
    def make_move(self, col: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Make a move in the specified column.
        
        Returns:
            next_state: Board after move
            reward: Reward for the move (+1 win, -1 loss, 0 otherwise)
            done: Whether game is over
            info: Additional information dict
        """
        if self.game_over:
            return self.board.copy(), 0, True, {"error": "Game already over"}
        
        if not self.is_valid_action(col):
            return self.board.copy(), 0, False, {"error": "Invalid action"}
        
        # Find lowest empty row in column
        row = self._get_lowest_empty_row(col)
        
        # Place piece
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            reward = self.reward
            done = True
        elif len(self.get_valid_actions()) == 0:
            # Draw
            self.game_over = True
            self.winner = 0
            reward = self.penalty
            done = True
        else:
            # Game continues
            reward = self.move_cost
            done = False
        
        # Switch players
        self.current_player *= -1
        
        info = {
            "move": (row, col),
            "winner": self.winner if done else None
        }
        
        return self.board.copy(), reward, done, info
    
    def _get_lowest_empty_row(self, col: int) -> int:
        """Find the lowest empty row in a column."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        return -1  # Should never happen if validation works
    
    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win."""
        player = self.board[row, col]
        
        # Four directions: horizontal, vertical, diagonal /, diagonal \
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # Count the piece just placed
            
            # Check positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= self.connect_n:
                return True
        
        return False
    
    def get_state_key(self) -> str:
        """
        Convert current state to a hashable key for Q-table.
        Format: board_string + current_player
        """
        board_str = ''.join(str(x) for x in self.board.flatten())
        return f"{board_str}_{self.current_player}"
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Print the current board state."""
        if mode == 'human':
            print("\n" + "=" * (self.cols * 4 + 1))
            for row in self.board:
                print("|", end="")
                for cell in row:
                    if cell == 0:
                        print("   |", end="")
                    elif cell == 1:
                        print(" X |", end="")
                    else:
                        print(" O |", end="")
                print()
            print("=" * (self.cols * 4 + 1))
            print(" " + "   ".join(str(i) for i in range(self.cols)))
            
            if self.game_over:
                if self.winner == 1:
                    print("X wins")
                elif self.winner == -1:
                    print("O wins")
                else:
                    print("Draw")
        elif mode == 'string':
            lines = []
            lines.append("=" * (self.cols * 4 + 1))
            for row in self.board:
                line = "|"
                for cell in row:
                    if cell == 0:
                        line += "   |"
                    elif cell == 1:
                        line += " X |"
                    else:
                        line += " O |"
                lines.append(line)
            lines.append("=" * (self.cols * 4 + 1))
            lines.append(" " + "   ".join(str(i) for i in range(self.cols)))
            return "\n".join(lines)
        
        return None
    
    def copy(self) -> 'ConnectFourEnv':
        """Create a deep copy of the environment."""
        new_env = ConnectFourEnv(self.rows, self.cols, self.connect_n)
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.game_over = self.game_over
        new_env.winner = self.winner
        new_env.move_history = self.move_history.copy()
        return new_env