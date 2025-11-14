import numpy as np
from argparse import Namespace
from collections import defaultdict
from datetime import datetime

from connect_four_env import ConnectFourEnv, PLAYERS
from rl_agent import RLModel, random_argmax

class ExpectedSarsa(RLModel):
    def __init__(self, env: ConnectFourEnv, opts: Namespace = None):
        super().__init__(env, opts)

    def name(self) -> str:
        return "expected-sarsa"

    def get_agent_name(self) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        return f"expected-sarsa--{now_str}.save"

    def get_expected_q(self, next_player, next_state):
        expected_q = 0
        q_max = np.max(self.get_q(next_player, next_state))
        greedy_actions = 0
        for a in range(self.env.get_number_of_actions()):
            if self.get_q(next_player, next_state, a) == q_max:
                greedy_actions += 1
        
        non_greedy_action_probability = self.epsilon / self.env.get_number_of_actions()
        greedy_action_probability = ((1 - self.epsilon) / greedy_actions) + non_greedy_action_probability

        for a in range(self.env.get_number_of_actions()):
            next_q = self.get_q(next_player, next_state, a)
            if next_q == q_max:
                expected_q += next_q * non_greedy_action_probability
            else:
                expected_q += next_q * greedy_action_probability
        
        return expected_q


    def train_step(self, episode: int) -> int:
        self.env.reset()
        state = self.env.get_state_key()
        current_player = self.env.current_player
        action = self.select_action(state, current_player)
        done = False

        total_reward = defaultdict(float)
        while not done:
            _, reward, done, _ = self.env.make_move(action)
            next_player = self.env.current_player
            next_state = self.env.get_state_key()
            next_action = self.select_action(state, current_player)

            if done:
                td_target = reward
            else:
                expected_q = self.get_expected_q(next_player, next_state)
                td_target = reward + self.gamma * expected_q

            old_q = self.get_q(current_player, state, action=action)
            new_q = old_q + self.alpha * (td_target - old_q)

            self.set_q(current_player, state, action, value=new_q)
            total_reward[PLAYERS[current_player]] += reward

            current_player = self.env.current_player
            state = next_state
            action = next_action
        return total_reward

    def eval_step(self, env: ConnectFourEnv):
        state = env.get_state_key()
        current_player = env.current_player
        
        action = random_argmax(self.get_q(current_player, state))

        return action
