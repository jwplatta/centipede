import argparse
from src import (
    StateMap,
    KMeansFindK,
    KMeansFindKPlot,
    SampleEnv,
    EpsilonDecay,
    QLearner,
    GYM_ENV_NAME,
    DEFAULT_ENV_SAMPLES_FILENAME,
    DEFAULT_STATE_MAP_FILENAME,
    DEFAULT_QTABLE_FILENAME,
    REPEAT_ACTION_PROBABILITY,
    FRAME_SKIP,
    DEFAULT_RENDER_MODE
)
import gymnasium as gym
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from src.CentipedeAgent import CentipedeAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="run",
        help="mode to run the program in"
    )
    parser.add_argument(
        "--samples_filename",
        type=str,
        default=DEFAULT_ENV_SAMPLES_FILENAME,
        help="filename to store the environment samples in"
    )
    parser.add_argument(
        "--state_map_filename",
        type=str,
        default=DEFAULT_STATE_MAP_FILENAME,
        help="filename to store the state map in"
    )
    parser.add_argument(
        "--qtable_filename",
        type=str,
        default=DEFAULT_QTABLE_FILENAME,
        help="filename to store the qtable in"
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=100,
        help="number of episodes to run the environment for"
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=3,
        help="number of runs to run the environment for"
    )
    parser.add_argument(
        "-c",
        "--clusters",
        type=int,
        default=100,
        help="number of clusters to build the state map with"
    )
    parser.add_argument(
        "--k_range",
        type=str,
        default="10,100,10",
        help="range of k values to run the findk algorithm with"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="policy",
        help="agent type to run the program with"
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=DEFAULT_RENDER_MODE,
        help="render mode to run the environment in"
    )
    args = parser.parse_args()
    print(args)

    if args.mode == "run":
        env = gym.make(
            GYM_ENV_NAME,
            frameskip=FRAME_SKIP,
            render_mode=args.render_mode,
            repeat_action_probability=REPEAT_ACTION_PROBABILITY
        )
        state_map = StateMap.load(args.state_map_filename)
        if args.agent_type == "random":
            agent = CentipedeAgent(env, state_map)
        elif args.agent_type == "policy":
            qtable = pd.read_csv(args.qtable_filename, header=None).to_numpy()
            agent = CentipedeAgent(
                env, state_map, qtable=qtable
            )

        scores = []
        steps = []
        e_times = []
        state_action_counts = []

        for episode in range(args.episodes):
            score, step, e_time, state_action_count = agent.run()
            scores.append(score)
            steps.append(step)
            e_times.append(e_time)
            state_action_counts.append(state_action_count)

        results_df = pd.DataFrame()
        results_df["score"] = scores
        results_df["steps"] = steps
        results_df["e_time"] = e_times

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_filename = f"results-{timestamp}.csv"
        results_df.to_csv(results_filename, index=False, header=True)

        print(
            f"Score: {np.mean(scores)}," + \
                " Steps: {np.mean(steps)}, Time: {np.mean(e_time)}"
        )
    elif args.mode == "train":
        state_map = StateMap.load(args.state_map_filename)
        theta=0.01
        min_episodes=args.episodes
        learning_rate=0.1
        gamma=0.2
        stochastic=True
        n_runs=args.runs
        action_size=18
        proba=0.9

        env = gym.make(
            GYM_ENV_NAME,
            frameskip=FRAME_SKIP,
            render_mode=None,
            repeat_action_probability=REPEAT_ACTION_PROBABILITY
        )
        qlearner = QLearner(
            learning_rate,
            gamma,
            state_map.size(),
            action_size,
            env,
            n_runs=n_runs,
            state_map=state_map,
            qtable_threshold=theta,
            min_episodes=min_episodes,
            epsilon_decay=EpsilonDecay(
                max_epsilon=0.5,
                decay=0.8,
                n_episodes=min_episodes,
                strategy='decay'
            )
        )
        qlearner.fit()
        qlearner.save_qtable(filename=args.qtable_filename)
        print("QTable saved to: ", args.qtable_filename)
    elif args.mode == "build":
        if not os.path.exists(args.samples_filename):
            raise Exception(
                f"File {args.samples_filename} does not exist." + \
                    "Please run the program in sample mode first."
            )

        if os.path.exists(args.state_map_filename):
            overwrite = input(
                f"File {args.state_map_filename} already exists. Overwrite? (y/n)"
            )
            if overwrite.lower() == "n":
                exit(0)

        env_samples_df = pd.read_csv(args.samples_filename, header=None)
        state_map = StateMap.build(
            env_samples_df,
            n_clusters=args.clusters,
            filename=args.state_map_filename
        )
    elif args.mode == "findk":
        if not os.path.exists(args.samples_filename):
            raise Exception(
                f"File {args.samples_filename} does not exist. " \
                  + "Please run the program in sample mode first."
            )

        env_samples_df = pd.read_csv(args.samples_filename, header=None)
        k_range = [int(x) for x in args.k_range.split(",")]
        find_k = KMeansFindK(k_range=np.arange(*k_range), verbose=True)
        find_k.run(env_samples_df)

        fig, axs = KMeansFindKPlot(find_k).plot()
        fig.savefig(
            "./findk_curves.png",
            bbox_inches='tight',
            dpi=400
        )

        plt.show()
    elif args.mode == "sample":
        if os.path.exists(args.samples_filename):
            overwrite = input(
                f"File {args.samples_filename} already exists. Overwrite? (y/n) "
            )

            if overwrite.lower() == "n":
                exit(0)

        fn = SampleEnv.run(n_episodes=args.episodes, filename=args.samples_filename)
        print("Samples saved to: ", fn)
    else:
        parser.print_help()
