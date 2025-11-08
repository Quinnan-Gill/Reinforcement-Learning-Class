import os
from typing import List
import numpy as np
from copy import copy
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from connect_four_env import ConnectFourEnv
from checkpoints import save_learning_curve

from rl_agent import RLModel
from q_learning import QLearning
from sarsa import Sarsa
from monte_carlo import MonteCarlo

def run_monte_carlo(env: ConnectFourEnv, opts: Namespace):
    agent = MonteCarlo(env, opts)
    agent.train()

def run_q_learning(env: ConnectFourEnv, opts: Namespace):
    agents = []
    env_copy = copy(env)
    base_agent = QLearning(env_copy, opts)
    for _ in range(opts.num_agents):
        agents.append(copy(base_agent))
   
    train_agents(agents, opts)

def run_sarsa(env: ConnectFourEnv, opts: Namespace):
    agents = []
    env_copy = copy(env)
    base_agent = Sarsa(env_copy, opts)
    for _ in range(opts.num_agents):
        agents.append(copy(base_agent))
   
    train_agents(agents, opts)
        
def train_agents(agents: List[RLModel], opts: Namespace) -> np.ndarray:
    learning_curve_red = []
    learning_curve_black = []
    for i in range(opts.episodes):
        print(f"\rEpisode {i} / {opts.episodes}", end="")

        total_rewards = defaultdict(list)
        for agent in agents:
            total_reward = agent.train_step(i)
            for player, val in total_reward.items():
                total_rewards[player].append(val)
        
        learning_curve_red.append(np.array(total_rewards['red']))
        learning_curve_black.append(np.array(total_rewards['black']))
    
    print("\nDone training agent")
    learning_curve_red = np.array(learning_curve_red)
    learning_curve_black = np.array(learning_curve_black)

    save_learning_curve(opts.workspace, player='red', data=learning_curve_red)
    save_learning_curve(opts.workspace, player='black', data=learning_curve_black)

    # save the best checkpoint based on last ten rewards
    best_red_idx = np.argmax(np.sum(learning_curve_red[-10:,], axis=0))
    best_red_agent = agents[best_red_idx].q['red']

    best_black_idx = np.argmax(np.sum(learning_curve_black[-10:,], axis=0))
    best_black_agent = agents[best_black_idx].q['black']

    if opts.workspace:
        np.save(
            os.path.join(opts.workspace, f"best_red_agent"),
            dict(best_red_agent)
        )
        np.save(
            os.path.join(opts.workspace, f"best_black_agent"),
            dict(best_black_agent)
        )

ALGO_MAP = {
    "monte-carlo": run_monte_carlo,
    "q-learning": run_q_learning,
    "sarsa": run_sarsa
}

def main():
    parser = ArgumentParser()
    parser.add_argument("--algo", required=True, choices=list(ALGO_MAP.keys()))

    parser.add_argument('-r', "--rows", type=int, default=3)
    parser.add_argument('-c', "--columns", type=int, default=4)
    parser.add_argument('-n', "--connect-n", type=int, default=3)

    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--move-cost", type=float, default=0.0)

    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--initial-val", type=float, default=0.0)

    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")

    # parser.add_argument("--player", default="agent", choices=["agent", "random", "human"])

    # if set don't save checkpoints (q-table)
    # if set save the checkpoints as EPISODE_AGLO.bin
    parser.add_argument('-o', "--workspace", default="", type=str)

    opts = parser.parse_args()

    env = ConnectFourEnv(
            opts.rows,
            opts.columns,
            opts.connect_n,
            opts.reward,
            opts.penalty,
            opts.move_cost
        )
    
    run_rl_algo = ALGO_MAP[opts.algo]
    run_rl_algo(env, opts)
    
if __name__ == '__main__':
    main()
