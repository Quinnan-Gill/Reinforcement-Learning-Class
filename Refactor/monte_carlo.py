from argparse import Namespace
from datetime import datetime
from typing import List, Tuple
import numpy as np

from connect_four_env import ConnectFourEnv
from rl_agent import RLAgentSingleQ, random_argmax

class MonteCarlo(RLAgentSingleQ):
    """
    Monte Carlo with single Q-table and perspective-normalized states.
    
    Key insight for zero-sum return calculation:
        G_t = r_t - γ * G_{t+1}
                  ^
                  MINUS sign alternates perspective!
    
    Why? In self-play:
    - At time t, current player gets reward r_t
    - At time t+1, OPPONENT moves and accumulates G_{t+1}
    - Opponent's gain is player's loss
    - So subtract the future return from immediate reward
    
    This is DIFFERENT from standard MC which uses:
        G_t = r_t + γ * G_{t+1}  (cooperative/single-agent)
    """
    
    def __init__(self, env: ConnectFourEnv, opts: Namespace = None):
        super().__init__(env, opts)
    
    def name(self) -> str:
        return "monte-carlo-single-q"
    
    def get_agent_name(self) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        return f"monte-carlo-single-q--{now_str}"
    
    def generate_episode(self) -> List[Tuple[str, int, float]]:
        """
        Generate a complete episode using epsilon-greedy policy.
        
        Returns:
            List of (state, action, reward) tuples
            Note: state is already perspective-normalized
        """
        self.env.reset()
        episode_data = []
        done = False
        
        while not done:
            state = self.get_state_key()
            action = self.select_action(state)
            _, reward, done, _ = self.env.make_move(action)
            
            # Store transition
            episode_data.append((state, action, reward))
        
        return episode_data
    
    def calculate_returns(self, episode_data: List[Tuple[str, int, float]]) -> List[float]:
        """
        Calculate returns using ZERO-SUM perspective alternation.
        
        Standard MC: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        Zero-sum MC: G_t = r_t - γ*G_{t+1}
        
        The minus sign accounts for opponent moves harming player.
        
        Args:
            episode_data: List of (state, action, reward) tuples
        
        Returns:
            List of returns, one per step
        """
        returns = []
        G = 0.0
        
        # Work backwards through episode
        for i in range(len(episode_data) - 1, -1, -1):
            _, _, reward = episode_data[i]
            
            G = reward - self.gamma * G
            
            returns.insert(0, G)
        
        return returns
    
    def update_q_values(self, episode_data: List[Tuple[str, int, float]], 
                       returns: List[float]):
        """
        First-visit Monte Carlo: Update Q-values for first occurrence of each (s,a).
        
        Args:
            episode_data: List of (state, action, reward) tuples
            returns: List of calculated returns for each step
        """
        visited = set()
        
        for i, (state, action, _) in enumerate(episode_data):
            sa_pair = (state, action)
            
            # First-visit only
            if sa_pair in visited:
                continue
            
            visited.add(sa_pair)
            
            # Update Q-value: Q(s,a) ← Q(s,a) + α[G - Q(s,a)]
            current_q = self.get_q(state, action)
            G = returns[i]
            new_q = current_q + self.alpha * (G - current_q)
            self.set_q(state, action, new_q)
    
    def train_step(self, episode: int) -> dict:
        """
        Train for one episode using Monte Carlo.
        
        Returns:
            Dict with total rewards for red and black
        """
        # Generate episode
        episode_data = self.generate_episode()
        
        # Calculate returns
        returns = self.calculate_returns(episode_data)
        
        # Update Q-values
        self.update_q_values(episode_data, returns)
        
        # Track rewards (alternating red/black)
        total_reward = {"red": 0.0, "black": 0.0}
        for i, (_, _, reward) in enumerate(episode_data):
            player = "red" if i % 2 == 0 else "black"
            total_reward[player] += reward
        
        self.update_epsilon(episode)

        return total_reward
