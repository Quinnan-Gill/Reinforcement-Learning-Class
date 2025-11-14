from argparse import ArgumentParser, Namespace
from typing import Tuple

from connect_four_env import ConnectFourEnv, PLAYERS
from checkpoints import load_params
from rl_agent import RLModel
from q_learning import QLearning
from sarsa import Sarsa
from expected_sarsa import ExpectedSarsa
from monte_carlo import MonteCarlo

ALGO_MAP = {
    "q-learning": QLearning,
    "sarsa": Sarsa,
    "expected-sarsa": ExpectedSarsa,
    "monte-carlo": MonteCarlo
}

def load_model(workspace: str, env: ConnectFourEnv = None):
    parameters = load_params(workspace)
    env_params = parameters["env"]
    model_params = parameters["model"]

    agent_type = model_params["agent-type"]
    if agent_type not in ALGO_MAP:
        raise ValueError(f"ERROR: {agent_type} does not support evaluation")
    AgentType = ALGO_MAP[agent_type]

    if env is None:
        env = ConnectFourEnv(
            rows=env_params["rows"],
            cols=env_params["cols"],
            connect_n=env_params["connect_n"],
            reward=env_params["reward"],
            penalty=env_params["penalty"],
            move_cost=env_params["move_cost"],
        )
    else:
        assert (
            env.rows == env_params["rows"] and
            env.cols == env_params["cols"] and
            env.connect_n == env_params["connect_n"]
        )

    # All agents now inherit from RLModel - no special handling needed
    agent: RLModel = AgentType(env)
    agent.load_workspace(workspace)

    return agent, env

def evaluate_game(env: ConnectFourEnv, red_agent: RLModel, black_agent: RLModel) -> Tuple[int, int]:
    env.reset()

    limit = 10_000

    for _ in range(limit):
        current_player = env.current_player
        if PLAYERS[current_player] == 'red':
            action = red_agent.eval_step(env)
        else:
            action = black_agent.eval_step(env)

        _, reward, done, _ = env.make_move(action)
        if done:
            return (env.current_player, reward)
        
    return (env.current_player, 0) # tie
    
def evaluate(env: ConnectFourEnv, red_agent: RLModel, black_agent: RLModel, num_games: int):
    results = {
        "red": 0,
        "black": 0,
        "tie": 0
    }

    for i in range(num_games):
        print(f"\rGame # {i} / {num_games}", end="")
        winner, reward = evaluate_game(env, red_agent, black_agent)

        if reward == env.penalty:
            results["tie"] += 1
        else:
            results[PLAYERS[winner]] += 1
    
    print("\nFinished playing games")
    print(results)
    



def main():
    parser = ArgumentParser()
    parser.add_argument("--red-agent", required=True, type=str)
    parser.add_argument("--black-agent", required=True, type=str)
    parser.add_argument("--num-games", type=int, default=1)
    
    opts = parser.parse_args()
    
    red_agent, env = load_model(opts.red_agent)
    black_agent, _ = load_model(opts.black_agent, env)

    evaluate(env, red_agent, black_agent, opts.num_games)
    


if __name__ == '__main__':
    main()
