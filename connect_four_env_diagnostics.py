"""
Diagnostic tools for Connect Four environment
Analyzes state space, validates game logic, and provides statistics
"""

import numpy as np
from collections import deque
from connect_four_env import ConnectFourEnv
from typing import Set, Dict, List

# =============================================================================
# CONFIGURATION - Update these values to test different board sizes
# =============================================================================
BOARD_ROWS = 3
BOARD_COLS = 4
CONNECT_N = 3
# =============================================================================


class ConnectFourDiagnostics:
    """Diagnostic and analysis tools for Connect Four environment."""
    
    def __init__(self, env: ConnectFourEnv):
        self.env = env
        
    def count_reachable_states(self, verbose: bool = True) -> int:
        """
        Count all reachable game states using BFS.
        This will be slow for larger boards.
        """
        visited_states = set()
        queue = deque()
        
        # Start from initial state
        initial_env = ConnectFourEnv(self.env.rows, self.env.cols, self.env.connect_n)
        initial_state = initial_env.get_state_key()
        queue.append(initial_env)
        visited_states.add(initial_state)
        
        terminal_states = 0
        non_terminal_states = 0
        
        while queue:
            current_env = queue.popleft()
            
            if current_env.game_over:
                terminal_states += 1
                continue
            
            non_terminal_states += 1
            
            # Try all valid actions
            for action in current_env.get_valid_actions():
                new_env = current_env.copy()
                new_env.make_move(action)
                state_key = new_env.get_state_key()
                
                if state_key not in visited_states:
                    visited_states.add(state_key)
                    queue.append(new_env)
        
        if verbose:
            print(f"State Space Analysis for {self.env.rows}×{self.env.cols} Connect-{self.env.connect_n}:")
            print(f"Total reachable states: {len(visited_states)}")
            print(f"  - Non-terminal states: {non_terminal_states}")
            print(f"  - Terminal states: {terminal_states}")
            print(f"    (Win/loss/draw outcomes)")
        
        return len(visited_states)
    
    def analyze_terminal_states(self) -> Dict[str, int]:
        """Analyze breakdown of terminal states (wins, losses, draws)."""
        visited_states = set()
        queue = deque()
        
        initial_env = ConnectFourEnv(self.env.rows, self.env.cols, self.env.connect_n)
        queue.append(initial_env)
        visited_states.add(initial_env.get_state_key())
        
        player1_wins = 0
        player2_wins = 0
        draws = 0
        
        while queue:
            current_env = queue.popleft()
            
            if current_env.game_over:
                if current_env.winner == 1:
                    player1_wins += 1
                elif current_env.winner == -1:
                    player2_wins += 1
                else:
                    draws += 1
                continue
            
            for action in current_env.get_valid_actions():
                new_env = current_env.copy()
                new_env.make_move(action)
                state_key = new_env.get_state_key()
                
                if state_key not in visited_states:
                    visited_states.add(state_key)
                    queue.append(new_env)
        
        print(f"\nTerminal State Breakdown:")
        print(f"  Player 1 (X) wins: {player1_wins}")
        print(f"  Player 2 (O) wins: {player2_wins}")
        print(f"  Draws: {draws}")
        print(f"  Total terminal states: {player1_wins + player2_wins + draws}")
        
        return {
            "player1_wins": player1_wins,
            "player2_wins": player2_wins,
            "draws": draws
        }
    
    def test_basic_functionality(self):
        """Test basic game mechanics."""
        print("\n" + "="*50)
        print("Testing Basic Functionality")
        print("="*50)
        
        # Test 1: Reset
        print("\n1. Testing reset...")
        self.env.reset()
        assert np.all(self.env.board == 0), "Board should be empty after reset"
        assert self.env.current_player == 1, "Player 1 should start"
        assert not self.env.game_over, "Game should not be over"
        print("   ✓ Reset works correctly")
        
        # Test 2: Valid actions
        print("\n2. Testing valid actions...")
        valid_actions = self.env.get_valid_actions()
        assert len(valid_actions) == self.env.cols, "All columns should be valid initially"
        print(f"   ✓ Valid actions: {valid_actions}")
        
        # Test 3: Make a move
        print("\n3. Testing move mechanics...")
        self.env.reset()
        state, reward, done, info = self.env.make_move(0)
        assert self.env.board[-1, 0] == 1, "Piece should drop to bottom"
        assert self.env.current_player == -1, "Player should switch"
        print("   ✓ Move mechanics work correctly")
        
        # Test 4: Invalid move
        print("\n4. Testing invalid move handling...")
        self.env.reset()
        # Fill a column
        for _ in range(self.env.rows):
            self.env.make_move(0)
        assert 0 not in self.env.get_valid_actions(), "Full column should be invalid"
        print("   ✓ Invalid moves properly rejected")
        
        # Test 5: Win detection
        print("\n5. Testing win detection...")
        self.env.reset()
        # Create a horizontal win for player 1
        for col in range(self.env.connect_n):
            self.env.make_move(col)  # Player 1
            if col < self.env.connect_n - 1:
                self.env.make_move(col)  # Player 2 (same column, stacks)
        
        # Check if game correctly identifies win
        if self.env.game_over and self.env.winner == 1:
            print("   ✓ Win detection works correctly")
        else:
            print("   ! Win detection test inconclusive (may need specific setup)")
        
        print("\n" + "="*50)
        print("Basic functionality tests complete")
        print("="*50)
    
    def analyze_game_tree_depth(self, max_depth: int = 20) -> Dict[int, int]:
        """Analyze distribution of game lengths."""
        depth_counts = {}
        visited_states = set()
        
        def explore(env: ConnectFourEnv, depth: int):
            state_key = env.get_state_key()
            if state_key in visited_states:
                return
            visited_states.add(state_key)
            
            if env.game_over or depth >= max_depth:
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
                return
            
            for action in env.get_valid_actions():
                new_env = env.copy()
                new_env.make_move(action)
                explore(new_env, depth + 1)
        
        initial_env = ConnectFourEnv(self.env.rows, self.env.cols, self.env.connect_n)
        explore(initial_env, 0)
        
        print(f"\nGame Tree Depth Analysis:")
        for depth in sorted(depth_counts.keys()):
            print(f"  Depth {depth}: {depth_counts[depth]} states")
        
        return depth_counts
    
    def sample_random_game(self, render: bool = True) -> List[int]:
        """Play a random game and optionally render it."""
        self.env.reset()
        moves = []
        
        if render:
            print("\nPlaying random game:")
            self.env.render()
        
        while not self.env.game_over:
            valid_actions = self.env.get_valid_actions()
            action = np.random.choice(valid_actions)
            moves.append(action)
            
            self.env.make_move(action)
            
            if render:
                print(f"\nPlayer {'X' if self.env.current_player == -1 else 'O'} plays column {action}")
                self.env.render()
        
        return moves


def run_full_diagnostics():
    """Run complete diagnostic suite."""
    print("\n" + "="*60)
    print("CONNECT FOUR ENVIRONMENT DIAGNOSTICS")
    print("="*60)
    
    # Test with configured values
    print(f"\n\nTesting {BOARD_ROWS}×{BOARD_COLS} Connect {CONNECT_N}")
    env = ConnectFourEnv(rows=BOARD_ROWS, cols=BOARD_COLS, connect_n=CONNECT_N)
    diag = ConnectFourDiagnostics(env)
    
    diag.test_basic_functionality()
    
    print("\nCounting reachable states (this may take a moment)...")
    state_count = diag.count_reachable_states()
    
    diag.analyze_terminal_states()
    diag.analyze_game_tree_depth()
    
    print("\n\nSample random game:")
    diag.sample_random_game(render=True)
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_full_diagnostics()