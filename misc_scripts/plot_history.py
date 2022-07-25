import argparse
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot history of dingo model.",
    )
    parser.add_argument(
        "train_dir",
        type=str,
        help="Path to train dir, which contains history.txt "
             "and where history.pdf is saved.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    history = np.loadtxt(join(args.train_dir, "history.txt"))

    plt.ylabel("Epoch")
    plt.xlabel("Loss")
    plt.plot(history[:,0], history[:,1], label="training loss")
    plt.plot(history[:,0], history[:,2], label="test loss")
    plt.legend()
    plt.savefig(join(args.train_dir, "history.pdf"))