import numpy as np
from connect_four_env import ConnectFourEnv


class RandomAgent:
    """
    Simple random agent that selects valid actions uniformly at random.
    Used as a baseline opponent for training RL agents.
    """
    
    def __init__(self, env: ConnectFourEnv):
        self.env = env
    
    def make_move(self) -> int:
        """
        Select a random valid action (column).
        
        Returns:
            int: Column index to play
        """
        valid_actions = self.env.get_valid_actions()
        return np.random.choice(valid_actions)
    
    def reset(self):
        """
        Reset method for compatibility with training loops.
        Random agent has no state to reset.
        """
        pass
