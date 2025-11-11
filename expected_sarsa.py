from argparse import Namespace
from collections import defaultdict
from datetime import datetime

from connect_four_env import ConnectFourEnv, PLAYERS
from rl_agent import RLModel, random_argmax

class Sarsa(RLModel):
    def __init__(self, env: ConnectFourEnv, opts: Namespace = None):
        super().__init__(env, opts)

    def name(self) -> str:
        return "expected-sarsa"

    def get_agent_name(self) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        return f"expected-sarsa--{now_str}.save"

    def train_step(self, episode: int) -> int:
        self.env.reset()
        state = self.env.get_state_key()
        current_player = self.env.current_player
        action = self.select_action(state, current_player)
        done = False

        total_reward = defaultdict(float)
        while not done:
            _, reward, done, _ = self.env.make_move(action)
            next_state = self.env.get_state_key()
            next_action = self.select_action(state, current_player)

            if done:
                td_target = reward
            else:
                q_values = self.get_q(current_player, next_state)
                n_actions = len(q_values)

                # ε-greedy policy probabilities
                probs = [self.epsilon / n_actions] * n_actions
                best_action = random_argmax(q_values)
                probs[best_action] += 1.0 - self.epsilon

                expected_q = sum(p * q for p, q in zip(probs, q_values))
                td_target = reward + self.gamma * expected_q
                
            td_error = td_target - (
                self.get_q(current_player, state, action=action)
            )

            self.set_q(current_player, state, action, value=self.alpha * td_error)
            total_reward[PLAYERS[current_player]] += reward

            current_player = self.env.current_player
            state = next_state
            action = next_action
        return total_reward
