from argparse import Namespace
from datetime import datetime
import numpy as np

from connect_four_env import ConnectFourEnv
from rl_agent import RLAgentSingleQ, random_argmax

class QLearning(RLAgentSingleQ):
    """
    Q-Learning with single Q-table and perspective-normalized states.
    
    Key update rule for zero-sum games:
        Q(s,a) ← Q(s,a) + α[r - γ * max Q(s',a') - Q(s,a)]
                                  ^
                                  MINUS sign for zero-sum
    
    Why the minus? After we move:
    - Opponent is now in state s' (from their perspective)
    - They pick their best action: max Q(s',a')
    - Their gain is our loss in zero-sum games
    - So we subtract their expected value, not add it
    """
    
    def __init__(self, env: ConnectFourEnv, opts: Namespace = None):
        super().__init__(env, opts)
    
    def name(self) -> str:
        return "q-learning-single-q"
    
    def get_agent_name(self) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        return f"q-learning-single-q--{now_str}"
    
    def train_step(self, episode: int) -> dict:
        """
        Train for one episode using Q-Learning with zero-sum updates.
        
        Returns:
            Dict with total rewards for red and black
        """
        self.env.reset()
        state = self.get_state_key()
        done = False
        
        # Track rewards for both players
        total_reward = {"red": 0.0, "black": 0.0}
        current_player_name = "red"  # Start with red
        
        while not done:
            # Select and take action
            action = self.select_action(state)
            _, reward, done, _ = self.env.make_move(action)
            
            # Get next state (from opponent's perspective after move)
            next_state = self.get_state_key()
            
            # Q-Learning TD update
            if not done:
                # Opponent picks their best action in next state (from valid actions only)
                next_valid_actions = self.env.get_valid_actions()
                next_q_values = self.get_q(next_state)
                best_next_q = np.max(next_q_values[next_valid_actions])
                
                td_target = reward - self.gamma * best_next_q
            else:
                # Terminal state: no future value
                td_target = reward
            
            # Update Q-value
            current_q = self.get_q(state, action)
            td_error = td_target - current_q
            new_q = current_q + self.alpha * td_error
            self.set_q(state, action, new_q)
            
            # Track reward
            total_reward[current_player_name] += reward
            
            # Switch player tracking
            current_player_name = "black" if current_player_name == "red" else "red"
            state = next_state

        self.update_epsilon(episode)
        
        return total_reward