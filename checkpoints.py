import os
import json
import tarfile
from typing import Union, Dict
from collections import defaultdict
import numpy as np

def save_checkpoint_numpy(filepath: str, data: np.ndarray):
    np.save(filepath, data)

def save_params(workspace: str, params: Dict):
    param_file = os.path.join(workspace, "parameters.json")
    with open(param_file, 'w') as fout:
        json.dump(params, fout, indent=2)

def load_params(workspace: str) -> Dict:
    if not os.path.exists(workspace):
        raise ValueError(f"ERROR: Unable to find workspace {workspace}")

    param_file = os.path.join(workspace, "parameters.json")
    if not os.path.exists(param_file):
        raise ValueError(f"ERROR: Unable to find parameters {param_file}")

    with open(param_file, 'r') as fin:
        return json.load(fin)

def save_checkpoint_dict(workspace: str, data: Dict):
    # remove default dict
    normal_dict = {}
    
    for player, q in data.items():
        normal_dict[player] = dict(q)
        # Save with just the base filename
        checkpoint_file = os.path.join(workspace, f"{player}.npz")
        np.savez(checkpoint_file, **dict(q))

def save_learning_curve(workspace: str, player: str, data: np.ndarray):
    if workspace:
        np.save(
            file=os.path.join(workspace, f"{player}_learning_curve"),
            arr=data
        )

def load_checkpoint_dict(workspace: str, action_size: int):    
    # Build paths to extracted files
    red_file = os.path.join(workspace, "best_red_agent.npy")
    black_file = os.path.join(workspace, "best_black_agent.npy")

    if not os.path.exists(red_file):
        raise ValueError(f"ERROR: Unable to find {red_file}")
    
    # Load and immediately convert to dict to close the file
    q_red_data = np.load(red_file, allow_pickle=True)
    q_red = defaultdict(
        lambda: np.zeros(action_size),
        q_red_data[()]
    )
    # {key: q_red_data[key] for key in q_red_data}
    # q_red_data.close()  # Explicitly close

    if not os.path.exists(black_file):
        raise ValueError(f"ERROR: Unable to find {black_file}")
    
    q_black_data = np.load(black_file, allow_pickle=True)
    # q_black = {key: q_black_data[key] for key in q_black_data}
    q_black = defaultdict(
        lambda: np.zeros(action_size),
        q_black_data[()]
    )
    # q_black_data.close()  # Explicitly close
    
    return {
        'red': q_red,
        'black': q_black
    }

def save_checkpoint(filepath: str, data: Union[np.ndarray, Dict]):
    if isinstance(data, np.ndarray):
        save_checkpoint_numpy(filepath, data)
    elif isinstance(data, dict):
        save_checkpoint_dict(filepath, data)
    else:
        raise NotImplementedError(f"Error: Unable to handle save of type {type(data)}")

def load_checkpoint(filepath: str, action_size: int=6) -> np.ndarray:
    if not os.path.exists(filepath):
        raise ValueError(f"ERROR: Unable to find {filepath}")

    if tarfile.is_tarfile(filepath):
        return load_checkpoint_dict(filepath, action_size)
    else:
        raise NotImplementedError("Implement other checkpoint methods")

def test():
    q_red = defaultdict(lambda: np.ones(10) * 0.0)
    q_red['a'][1] = 1.0
    q_black = defaultdict(lambda: np.ones(10) * 0.0)
    q_black['b'][2] = 2.0

    q = {
        'red': q_red,
        'black': q_black
    }

    save_checkpoint_dict('test', q)
    result = load_checkpoint('test')
    # import pdb
    # pdb.set_trace()
    assert result['red']['a'][1] == 1.0
    assert result['black']['b'][2] == 2.0


if __name__ == '__main__':
    test()