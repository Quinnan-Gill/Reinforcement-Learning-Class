from argparse import ArgumentParser, Namespace
from connect_four_env import ConnectFourEnv

from q_learning import QLearning

def run_monte_carlo(env: ConnectFourEnv, opts: Namespace):
    # TODO: Implement this
    pass

def run_q_learning(env: ConnectFourEnv, opts: Namespace):
    # TODO: Implement this
    agent = QLearning(env, opts)
    agent.train()


ALGO_MAP = {
    "monte-carlo": run_monte_carlo,
    "q-learning": run_q_learning,
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

    # parser.add_argument("--player", default="agent", choices=["agent", "random", "human"])

    # if set don't save checkpoints (q-table)
    # if set save the checkpoints as EPISODE_AGLO.bin
    parser.add_argument('-o', "--checkpoint-dir", default="", type=str)

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
