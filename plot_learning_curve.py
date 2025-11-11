import os
from typing import List
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

# Turn this into a jupiter notebook
def plot_learning_curve(workspaces: List[str]):
    print(workspaces)
    fig = plt.figure()

    for i, workspace in enumerate(workspaces):
        if not os.path.exists(workspace):
            raise ValueError(f"Error: Could not find {workspace}")
        red_file = os.path.join(workspace, "red_learning_curve.npy")
        black_file = os.path.join(workspace, "black_learning_curve.npy")
        
        plt.subplot(2, 1, 1)
        train_data = np.average(np.load(red_file), axis=1)
        plt.plot(range(len(train_data)), train_data)

        plt.subplot(2, 1, 2)
        train_data = np.average(np.load(black_file), axis=1)
        plt.plot(range(len(train_data)), train_data)
    
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--workspace', action='append', help='Workspaces for creating learning curve')
    parser.add_argument('-s', '--smoothing', default=11, type=int)

    opts = parser.parse_args()

    plot_learning_curve(opts.workspace)

if __name__ == '__main__':
    main()
