import numpy as np
from connect_four_env import ConnectFourEnv, PLAYERS
from checkpoints import load_checkpoint


class FrozenAgent:
    """
    Agent that loads Q-tables from a saved checkpoint and uses them to make decisions.
    Does not learn or update Q-values - used as a frozen opponent for curriculum learning.
    """
    
    def __init__(self, env: ConnectFourEnv, checkpoint_path: str, player: int = -1):
        """
        Initialize agent from a saved checkpoint.
        
        Args:
            env: The game environment
            checkpoint_path: Path to saved checkpoint file (.save)
            player: Which player this agent represents (1 for red, -1 for black)
                   Default -1 since checkpoint agents typically play as opponent (black)
        """
        self.env = env
        self.player = player
        self.player_str = PLAYERS[player]
        
        # Load Q-tables from checkpoint
        self.q = load_checkpoint(checkpoint_path)
        
        print(f"CheckpointAgent loaded from: {checkpoint_path}")
        print(f"Playing as: {self.player_str} (player {player})")
    
    def get_q(self, state: str, action: int = None):
        """
        Get Q-value(s) for the current player.
        
        Args:
            state: State key string
            action: Specific action (if None, returns all Q-values for state)
        
        Returns:
            Q-value(s) for the state-action pair(s)
        """
        # Handle unseen states with default Q-values (zeros)
        if state not in self.q[self.player_str]:
            default_q = np.zeros(self.env.get_number_of_actions())
            if action is None:
                return default_q
            return default_q[action]
        
        if action is None:
            return self.q[self.player_str][state]
        return self.q[self.player_str][state][action]
    
    def make_move(self) -> int:
        """
        Select best action based on loaded Q-values (greedy policy, no exploration).
        Uses action masking to only consider valid moves.
        
        Returns:
            int: Column index to play
        """
        state = self.env.get_state_key()
        valid_actions = self.env.get_valid_actions()
        
        # Get Q-values for current state
        q_values = self.get_q(state)
        
        # Mask invalid actions with -inf
        masked_q = np.full(self.env.get_number_of_actions(), -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        
        # Select best valid action (greedy)
        return np.argmax(masked_q)
    
    def reset(self):
        """
        Reset method for compatibility with training loops.
        Checkpoint agent has no state to reset.
        """
        pass