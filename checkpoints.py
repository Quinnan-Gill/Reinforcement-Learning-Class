import os
import shutil
import tarfile
from typing import Union, Dict
from collections import defaultdict
import numpy as np

def save_checkpoint_numpy(filepath: str, data: np.ndarray):
    np.save(filepath, data)

def save_checkpoint_dict(filepath: str, data: Dict):
    # remove default dict
    normal_dict = {}
    files_to_tar = []
    
    # Extract just the filename without directory
    base_filename = os.path.basename(filepath)
    tar_dir = os.path.dirname(filepath)
    if not tar_dir:
        tar_dir = "."
    
    for player, q in data.items():
        normal_dict[player] = dict(q)
        # Save with just the base filename
        temp_file = f"{base_filename}.{player}.npz"
        temp_path = os.path.join(tar_dir, temp_file)
        np.savez(temp_path, **dict(q))
        files_to_tar.append((temp_path, temp_file))  # (full_path, arcname)
    
    with tarfile.open(filepath, "w") as tar:
        for file_path, arcname in files_to_tar:
            tar.add(file_path, arcname=arcname)  # Store with just filename, not full path
    
    for file_path, _ in files_to_tar:
        os.remove(file_path)
    
def load_checkpoint_dict(filepath: str):
    try:
        # Get the directory where the tar file is located
        tar_dir = os.path.dirname(filepath)
        if not tar_dir:
            tar_dir = "."
            
        with tarfile.open(filepath, 'r') as tar:
            # Extract all contents to the tar file's directory
            tar.extractall(path=tar_dir)
        
        # Build paths to extracted files
        red_file = f"{filepath}.red.npz"
        black_file = f"{filepath}.black.npz"
        
        if not os.path.exists(red_file):
            raise ValueError(f"Error opening {red_file}")
        
        # Load and immediately convert to dict to close the file
        q_red_data = np.load(red_file, allow_pickle=True)
        q_red = {key: q_red_data[key] for key in q_red_data.files}
        q_red_data.close()  # Explicitly close
        os.remove(red_file)

        if not os.path.exists(black_file):
            raise ValueError(f"Error opening {black_file}")
        
        q_black_data = np.load(black_file, allow_pickle=True)
        q_black = {key: q_black_data[key] for key in q_black_data.files}
        q_black_data.close()  # Explicitly close
        os.remove(black_file)
        
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