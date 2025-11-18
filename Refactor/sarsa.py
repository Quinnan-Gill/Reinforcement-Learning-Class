from argparse import Namespace
from datetime import datetime
import numpy as np

from connect_four_env import ConnectFourEnv
from rl_agent import RLAgentSingleQ

class SARSA(RLAgentSingleQ):
    """
    SARSA with single Q-table and perspective-normalized states.
    
    Key update rule for zero-sum games:
        Q(s,a) ← Q(s,a) + α[r - γ * Q(s',a') - Q(s,a)]
                                  ^
                                  MINUS sign for zero-sum!
    
    Difference from Q-Learning:
    - Q-Learning uses max Q(s',a') (opponent's best action)
    - SARSA uses Q(s',a') where a' is the action actually taken
    - On-policy vs off-policy
    """
    
    def __init__(self, env: ConnectFourEnv, opts: Namespace = None):
        super().__init__(env, opts)
    
    def name(self) -> str:
        return "sarsa-single-q"
    
    def get_agent_name(self) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        return f"sarsa-single-q--{now_str}"
    
    def train_step(self, episode: int) -> dict:
        """
        Train for one episode using SARSA with zero-sum updates.
        
        Returns:
            Dict with total rewards for red and black
        """
        self.env.reset()
        state = self.get_state_key()
        action = self.select_action(state)  # Pre-select first action (SARSA requirement)
        done = False
        
        # Track rewards for both players
        total_reward = {"red": 0.0, "black": 0.0}
        current_player_name = "red"
        
        while not done:
            # Take action
            _, reward, done, _ = self.env.make_move(action)
            
            # Get next state
            next_state = self.get_state_key()
            
            # SARSA TD update
            if not done:
                # Select action that will actually be taken (epsilon-greedy)
                next_action = self.select_action(next_state)
                next_q = self.get_q(next_state, next_action)
                
                # CRITICAL: Subtract opponent's Q-value (zero-sum)
                td_target = reward - self.gamma * next_q
            else:
                # Terminal state
                td_target = reward
                next_action = None  # Won't be used
            
            # Update Q-value
            current_q = self.get_q(state, action)
            td_error = td_target - current_q
            new_q = current_q + self.alpha * td_error
            self.set_q(state, action, new_q)
            
            # Track reward
            total_reward[current_player_name] += reward
            
            # Move to next state
            current_player_name = "black" if current_player_name == "red" else "red"
            state = next_state
            action = next_action

        self.update_epsilon(episode)
        
        return total_reward
