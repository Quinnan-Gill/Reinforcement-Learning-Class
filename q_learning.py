import os
from typing import Optional
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
import numpy as np
from connect_four_env import ConnectFourEnv, PLAYERS
from checkpoints import save_checkpoint, load_checkpoint

class QLearning:
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
        return f"q-learning--{now_str}.save"

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
        Epsilon-greedy exploration/exploitation 
        """

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.env.get_number_of_actions())
        else:
            # return np.argmax(self.q[current_player][state])
            return np.argmax(self.get_q(current_player, state))

    def train(self):
        for i in range(self.episodes):
            print(f"\rEpisode {i} / {self.episodes}", end="")
            self.env.reset()
            state = self.env.get_state_key()
            current_player = self.env.current_player
            done = False

            while not done:
                action = self.explore_exploit(state, current_player)
                
                _, reward, done, _ = self.env.make_move(action)
                next_state = self.env.get_state_key()

                best_next_action = np.argmax(self.get_q(current_player, state))
                td_target = reward + self.gamma * (
                    self.get_q(current_player, state, action=best_next_action)
                )
                td_error = td_target - (
                    self.get_q(current_player, state, action=action)
                )
                # self.q[current_player][state][action] += self.alpha * td_error
                self.set_q(current_player, state, action, value=self.alpha * td_error)

                current_player = self.env.current_player
                state = next_state

        print("Done training q-learning")
        if self.checkpoints:
            save_file = os.path.join(self.checkpoints, self.get_agent_name())
            save_checkpoint(save_file, self.q)
        
    def evaluate(self, filepath: str):
        if filepath:
            self.q = load_checkpoint(filepath)

    # def make_move(self, )

            

        