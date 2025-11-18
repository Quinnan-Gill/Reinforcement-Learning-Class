import os
import sys
import shutil
from argparse import Namespace
from collections import defaultdict
from typing import Optional
import numpy as np

from connect_four_env import ConnectFourEnv
from perspective_state import normalize_state_perspective

def random_argmax(values: np.ndarray):
    """Break ties randomly when selecting argmax"""
    return np.argmax(np.random.random(values.shape) * (values==values.max()))


class RLAgentSingleQ:
    """
    Base RL agent with SINGLE Q-table and perspective-normalized states.
    
    Key architecture:
    - One Q-table: Q[state][action]
    - States are perspective-normalized (always from current player's view)
    - Agent automatically learns to play both Red and Black
    """
    
    def __init__(self, env: ConnectFourEnv, opts: Namespace = None):
        self.env = env
        
        if not opts:
            return
        
        # SINGLE Q-table (no red/black split)
        self.Q = defaultdict(lambda: np.ones(self.env.get_number_of_actions()) * opts.initial_val)
        
        self.gamma = opts.gamma
        self.alpha = opts.alpha
        self.epsilon_start = opts.epsilon_start
        self.epsilon_end = opts.epsilon_end
        self.epsilon_decay = opts.epsilon_decay
        self.epsilon = self.epsilon_start  # Start at epsilon_start
        self.episodes = opts.episodes
        
        self.workspace = opts.workspace
        if self.workspace and not os.path.exists(self.workspace):
            print(f"Creating {self.workspace}")
            os.makedirs(self.workspace)
            self.save_parameters()
        elif self.workspace and not opts.overwrite:
            print(f"Workspace {self.workspace} already exists use `--overwrite` to clear")
            sys.exit(-1)
        elif self.workspace and opts.overwrite:
            print(f"Clearing AND Creating {self.workspace}")
            shutil.rmtree(self.workspace)
            os.makedirs(self.workspace)
            self.save_parameters()
    
    def name(self) -> str:
        raise NotImplementedError()
    
    def get_state_key(self) -> str:
        """Get perspective-normalized state key for current position."""
        return normalize_state_perspective(self.env.board, self.env.current_player)
    
    def get_q(self, state: str, action: Optional[int] = None):
        """Get Q-value(s) for a state."""
        if action is None:
            return self.Q[state]
        return self.Q[state][action]
    
    def set_q(self, state: str, action: int, value: float):
        """Set Q-value for a state-action pair."""
        self.Q[state][action] = value
    
    def select_action(self, state: str) -> int:
        """
        Epsilon-greedy action selection with invalid action masking.
        """
        valid_actions = self.env.get_valid_actions()
        
        if not valid_actions:
            return 0
        
        # Explore: random valid action
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploit: best valid action
        q_values = self.get_q(state)
        
        # Mask invalid actions
        masked_q = np.full(self.env.get_number_of_actions(), -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        
        return random_argmax(masked_q)
    
    def eval_step(self, env: ConnectFourEnv) -> int:
        """
        Greedy action selection for evaluation (no exploration).
        """
        state = normalize_state_perspective(env.board, env.current_player)
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return 0
        
        q_values = self.get_q(state)
        
        # Mask invalid actions
        masked_q = np.copy(q_values)
        for action in range(len(q_values)):
            if action not in valid_actions:
                masked_q[action] = -np.inf
        
        return random_argmax(masked_q)
    
    def save_parameters(self):
        """Save training hyperparameters."""
        import json
        parameters = {
            "env": self.env.get_parameters(),
            "model": {
                "agent-type": self.name(),
                "gamma": self.gamma,
                "alpha": self.alpha,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "episodes": self.episodes
            }
        }
        param_file = os.path.join(self.workspace, "parameters.json")
        with open(param_file, 'w') as f:
            json.dump(parameters, f, indent=2)

    def update_epsilon(self, episode: int):
        """Decay epsilon over episodes."""
        self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** episode))
    
    def save_model(self):
        """Save Q-table to workspace."""
        if self.workspace:
            import numpy as np
            # Convert defaultdict to regular dict for saving
            q_dict = dict(self.Q)
            save_path = os.path.join(self.workspace, "Q_table.npz")
            np.savez(save_path, **q_dict)
            print(f"Saved Q-table to {save_path}")
    
    def load_model(self, workspace: str):
        """Load Q-table from workspace."""
        import json
        self.workspace = workspace
        
        # Load parameters
        param_file = os.path.join(workspace, "parameters.json")
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        self.gamma = params["model"]["gamma"]
        self.alpha = params["model"]["alpha"]
        self.epsilon_start = params["model"]["epsilon_start"]
        self.epsilon_end = params["model"]["epsilon_end"]
        self.epsilon_decay = params["model"]["epsilon_decay"]
        self.epsilon = self.epsilon_start
        self.episodes = params["model"]["episodes"]
        
        # Load Q-table
        q_file = os.path.join(workspace, "Q_table.npz")
        q_data = np.load(q_file, allow_pickle=True)
        self.Q = defaultdict(
            lambda: np.zeros(self.env.get_number_of_actions()),
            {key: q_data[key] for key in q_data.files}
        )
        print(f"Loaded Q-table from {q_file}")
    
    def train_step(self, episode: int) -> dict:
        """Train for one episode. Must be implemented by subclasses."""
        raise NotImplementedError()
    
    def get_agent_name(self) -> str:
        """Get unique agent name for saving."""
        raise NotImplementedError()
