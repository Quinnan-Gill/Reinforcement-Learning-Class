import os
import shutil
import tarfile
from typing import Union, Dict
from collections import defaultdict
import numpy as np

def save_checkpoint_numpy(filepath: str, data: np.ndarray):
    """
    Save a numpy array to a file
    """
    np.save(filepath, data)

def save_checkpoint_dict(filepath: str, data: Dict):
    # remove default dict
    normal_dict = {}
    files_to_tar = []
    for player, q in data.items():
        normal_dict[player] = dict(q)
        np.savez(f"{filepath}.{player}.npz", **dict(q))
        files_to_tar.append(f"{filepath}.{player}.npz")
    
    with tarfile.open(filepath, "w") as tar:
        for file_path in files_to_tar:
            tar.add(file_path)
    
    for file_path in files_to_tar:
        os.remove(file_path)
    
def load_checkpoint_dict(filepath: str):
    try:
        with tarfile.open(filepath, 'r') as tar:
            # Extract all contents to the specified path
            tar.extractall(path=os.path.dirname(filepath))
        if not os.path.exists(f"{filepath}.red.npz"):
            raise ValueError(f"Error opening {filepath}.red.npz")
        
        q_red = np.load(f"{filepath}.red.npz")
        os.remove(f"{filepath}.red.npz")

        if not os.path.exists(f"{filepath}.black.npz"):
            raise ValueError(f"Error opening {filepath}.black.npz")
        
        q_black = np.load(f"{filepath}.black.npz")
        os.remove(f"{filepath}.black.npz")
        return {
            'red': q_red,
            'black': q_black
        }
        
    except tarfile.ReadError as e:
        raise ValueError(f"Error opening or reading tar file: {e}")
    except FileNotFoundError:
        raise ValueError(f"Error: Tar file '{filepath}' not found.")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")

def save_checkpoint(filepath: str, data: Union[np.ndarray, Dict]):
    if isinstance(data, np.ndarray):
        save_checkpoint_numpy(filepath, data)
    elif isinstance(data, dict):
        save_checkpoint_dict(filepath, data)
    else:
        raise NotImplementedError(f"Error: Unable to handle save of type {type(data)}")

def load_checkpoint(filepath: str) -> np.ndarray:
    if not os.path.exists(filepath):
        raise ValueError(f"ERROR: Unable to find {filepath}")

    if tarfile.is_tarfile(filepath):
        return load_checkpoint_dict(filepath)
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