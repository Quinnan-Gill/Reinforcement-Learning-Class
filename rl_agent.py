import os
import sys
import shutil
from argparse import Namespace
from collections import defaultdict
from typing import Optional
import numpy as np

from connect_four_env import ConnectFourEnv, PLAYERS
from checkpoints import save_params, save_checkpoint, load_params, load_checkpoint_dict

def random_argmax(values: np.ndarray):
    return np.argmax(np.random.random(values.shape) * (values==values.max()))


class RLModel:
    def __init__(self, env: ConnectFourEnv, opts: Namespace = None):
        self.env = env
        if not opts:
            return
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

        self.workspace = opts.workspace
        if self.workspace and not os.path.exists(self.workspace):
            print(f"Creating {self.workspace}")
            os.makedirs(self.workspace)
            self.save_parameters()
        elif self.workspace and not opts.overwrite:
            print(f"Workspace {self.workspace} already exists use `--overwrite` to clear the workspace")
            sys.exit(-1)
        elif self.workspace and opts.overwrite:
            print(f"Clearning AND Creating {self.workspace}")
            shutil.rmtree(self.workspace)
            os.makedirs(self.workspace)
            self.save_parameters()

    def name(self) -> str:
        raise NotImplementedError()

    def get_q(self, current_player: int, state: str, action: Optional[int] = None) -> float:
        player_str = PLAYERS[current_player]
        if action is None:
            return self.q[player_str][state]
        return self.q[player_str][state][action]
    
    def set_q(self, current_player: int, state: str, action: int, value: float):
        player_str = PLAYERS[current_player]
        self.q[player_str][state][action] = value

    def save_parameters(self):
        parameters = {
            "env": self.env.get_parameters(),
            "model": {
                "agent-type": self.name(),
                "gamma": self.gamma,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "episodes": self.episodes
            }
        }
        save_params(self.workspace, params=parameters)
    
    def save_model(self):
        if self.workspace:
            save_checkpoint(self.workspace, self.q)
    
    def load_workspace(self, workspace: str):
        self.workspace = workspace
        parameters = load_params(workspace)

        self.gamma = parameters["model"]["gamma"]
        self.alpha = parameters["model"]["alpha"]
        self.epsilon = parameters["model"]["epsilon"]
        self.episodes = parameters["model"]["episodes"]

        self.q = load_checkpoint_dict(workspace, self.env.get_number_of_actions())
        
    def select_action(self, state: str, current_player: int) -> int:
        """
        Epsilon-greedy exploration/exploitation 
        """

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.env.get_number_of_actions())
        else:
            # return np.argmax(self.q[current_player][state])
            return random_argmax(self.get_q(current_player, state))

    def get_agent_name(self) -> str:
        raise NotImplementedError()
    
    def train_step(self) -> int:
        raise NotImplementedError()
    
    def save_model(self):
        raise NotImplementedError()
    
