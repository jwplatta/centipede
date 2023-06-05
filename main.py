import argparse
from src import (
    StateMap,
    KMeansFindK,
    KMeansFindKPlot,
    SampleEnv,
    GYM_ENV_NAME,
    ENV_SAMPLES_FILENAME,
    STATE_MAP_FILENAME,
    REPEAT_ACTION_PROBABILITY,
    FRAME_SKIP
)
import gymnasium as gym
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        nargs='+',
        default=None,
        help="mode to run the program in"
    )
    args = parser.parse_args()
    print(args)
    mode = args.mode[0]

    if mode == "run":
       pass
    elif mode == "train":
       pass
    elif mode == "build":
        try:
            n_clusters = int(args.mode[1])
        except IndexError:
            raise Exception("Please provide the number of clusters to build the state map with.")

        if not os.path.exists(ENV_SAMPLES_FILENAME):
            raise Exception(f"File {ENV_SAMPLES_FILENAME} does not exist. Please run the program in sample mode first.")

        if os.path.exists(STATE_MAP_FILENAME):
            overwrite = input(f"File {STATE_MAP_FILENAME} already exists. Overwrite? (y/n) ")
            if overwrite.lower() == "n":
                exit(0)

        env_samples_df = pd.read_csv(ENV_SAMPLES_FILENAME)
        state_map = StateMap.build(env_samples_df, n_clusters=n_clusters, filename=STATE_MAP_FILENAME)
    elif mode == "findk":
        if not os.path.exists(ENV_SAMPLES_FILENAME):
            raise Exception(f"File {ENV_SAMPLES_FILENAME} does not exist. Please run the program in sample mode first.")

        env_samples_df = pd.read_csv(ENV_SAMPLES_FILENAME)

        arg_cnt = len(args.mode)
        if arg_cnt >= 2:
            k_range = np.arange(*[ int(x) for x in args.mode[1:]])
        else:
            k_range = np.arange(2, 20, 2)

        find_k = KMeansFindK(k_range=k_range, verbose=True)
        find_k.run(env_samples_df)
        fig, axs = KMeansFindKPlot(find_k).plot()
        fig.savefig(
            "./findk_curves.png",
            bbox_inches='tight',
            dpi=600
        )

        plt.show()
    elif mode == "sample":
        try:
            n_episodes = int(args.mode[1])
        except IndexError:
            raise Exception("Please provide the number of episodes to run the environment for.")

        if os.path.exists(ENV_SAMPLES_FILENAME):
            overwrite = input(f"File {ENV_SAMPLES_FILENAME} already exists. Overwrite? (y/n) ")
            if overwrite.lower() == "n":
                exit(0)

        filename = SampleEnv.run(n_episodes=n_episodes)
        print("Samples saved to", filename)
    else:
        parser.print_help()
