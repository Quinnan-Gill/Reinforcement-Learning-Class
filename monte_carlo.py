import os
from typing import Optional, List, Tuple
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
import numpy as np
from connect_four_env import ConnectFourEnv, PLAYERS
from checkpoints import save_checkpoint, load_checkpoint


class MonteCarlo:
    def __init__(self, env: ConnectFourEnv, opts: Namespace):
        self.env = env
        q_red = defaultdict(lambda: np.ones(self.env.get_number_of_actions()) * opts.initial_val)
        q_black = defaultdict(lambda: np.ones(self.env.get_number_of_actions()) * opts.initial_val)
        self.q = {
            'red': q_red,
            'black': q_black,
        }

        self.gamma = opts.gamma
        self.alpha = opts.alpha
        self.epsilon = opts.epsilon
        self.episodes = opts.episodes

        self.checkpoints = opts.checkpoint_dir
        if self.checkpoints and not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)
    
    def get_agent_name(self) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        return f"monte-carlo--{now_str}.save"

    def get_q(self, current_player: int, state: str, action: Optional[int] = None) -> float:
        player_str = PLAYERS[current_player]
        if action is None:
            return self.q[player_str][state]
        return self.q[player_str][state][action]
    
    def set_q(self, current_player: int, state: str, action: int, value: float):
        player_str = PLAYERS[current_player]
        self.q[player_str][state][action] = value

    def explore_exploit(self, state: str, current_player: int) -> int:
        """
        Epsilon-greedy exploration/exploitation with valid action masking
        """
        valid_actions = self.env.get_valid_actions()
        
        # Explore: choose random valid action
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploit: choose best valid action
        q_values = self.get_q(current_player, state)
        
        # Mask invalid actions with -inf
        masked_q = np.full(self.env.get_number_of_actions(), -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        
        return np.argmax(masked_q)

    def generate_episode(self, opponent=None) -> List[Tuple[str, int, int, float]]:
        """
        Generate a complete episode using epsilon-greedy policy.
        
        Args:
            opponent: Optional opponent agent (e.g., RandomAgent). If provided:
                      - This agent plays as Red (player 1)
                      - Opponent plays as Black (player -1)
                      If None, do self-play (both players learn)
        
        Returns:
            List of (state, player, action, reward) tuples
        """
        self.env.reset()
        episode_data = []
        done = False

        while not done:
            state = self.env.get_state_key()
            current_player = self.env.current_player
            
            # Select action based on who's playing
            if opponent is not None and current_player == -1:
                # Opponent's turn (Black)
                action = opponent.make_move()
            else:
                # Agent's turn (Red in vs-opponent mode, or both in self-play)
                action = self.explore_exploit(state, current_player)
            
            # Take action
            _, reward, done, _ = self.env.make_move(action)
            
            # Store (state, player, action, reward)
            episode_data.append((state, current_player, action, reward))

        return episode_data

    def calculate_returns(self, episode_data: List[Tuple[str, int, int, float]]) -> List[float]:
        """
        Calculate returns for each step in the episode working backwards.
        G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ...
        
        Returns:
            List of returns, one for each step
        """
        returns = []
        G = 0.0
        
        # Work backwards from end of episode
        for i in range(len(episode_data) - 1, -1, -1):
            _, _, _, reward = episode_data[i]
            G = reward + self.gamma * G
            returns.insert(0, G)  # Insert at beginning to maintain order
        
        return returns

    def update_q_values(self, episode_data: List[Tuple[str, int, int, float]], 
                       returns: List[float], opponent=None):
        """
        First-visit Monte Carlo: Update Q-values for first occurrence of each (state, action) pair.
        
        Args:
            episode_data: List of (state, player, action, reward) tuples
            returns: List of calculated returns for each step
            opponent: If provided, only update Red (player 1). If None, update both players.
        """
        # Track which (player, state, action) pairs we've already updated
        visited = set()
        
        for i, (state, player, action, _) in enumerate(episode_data):
            # If training against opponent, only update the agent (Red, player 1)
            if opponent is not None and player == -1:
                continue
            
            sa_pair = (player, state, action)
            
            # First-visit only: skip if we've seen this state-action pair before in this episode
            if sa_pair in visited:
                continue
            
            visited.add(sa_pair)
            
            # Get current Q-value
            current_q = self.get_q(player, state, action)
            
            # Update using incremental mean: Q(s,a) = Q(s,a) + alpha * [G - Q(s,a)]
            G = returns[i]
            new_q = current_q + self.alpha * (G - current_q)
            
            # Store updated Q-value
            self.set_q(player, state, action, new_q)

    def train(self, opponent=None, phase_name: str = ""):
        """
        Train using first-visit Monte Carlo with epsilon-greedy policy.
        
        Args:
            opponent: Optional opponent agent. If provided, agent plays as Red vs opponent.
                     If None, do self-play (both Red and Black learn).
            phase_name: Optional name for this training phase (for display/logging)
        """
        phase_label = f" ({phase_name})" if phase_name else ""
        training_mode = "vs Opponent" if opponent else "Self-Play"
        print(f"\nTraining Monte Carlo - {training_mode}{phase_label}")
        
        for episode in range(self.episodes):
            print(f"\rEpisode {episode + 1} / {self.episodes}", end="")
            
            # Generate a complete episode
            episode_data = self.generate_episode(opponent)
            
            # Calculate returns for each step
            returns = self.calculate_returns(episode_data)
            
            # Update Q-values using first-visit MC
            self.update_q_values(episode_data, returns, opponent)

        print(f"\nDone training Monte Carlo - {training_mode}{phase_label}")
        
        # Save checkpoint
        if self.checkpoints:
            checkpoint_name = phase_name.lower().replace(" ", "-") if phase_name else "default"
            save_file = os.path.join(self.checkpoints, 
                                    f"monte-carlo--{checkpoint_name}--{datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}.save")
            save_checkpoint(save_file, self.q)
            print(f"Saved checkpoint to {save_file}")
        
    def evaluate(self, filepath: str):
        """
        Load a trained agent from checkpoint.
        """
        if filepath:
            self.q = load_checkpoint(filepath)
            print(f"Loaded checkpoint from {filepath}")
