import argparse
from src import (
    StateMap,
    KMeansFindK,
    KMeansFindKPlot,
    SampleEnv,
    EpsilonDecay,
    QLearner,
    CentipedeAgent,
    GYM_ENV_NAME,
    DEFAULT_ENV_SAMPLES_FILENAME,
    DEFAULT_STATE_MAP_FILENAME,
    DEFAULT_QTABLE_FILENAME,
    REPEAT_ACTION_PROBABILITY,
    FRAME_SKIP,
    DEFAULT_RENDER_MODE,
    DEFAULT_OBS_TYPE
)
import gymnasium as gym
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


def check_file_dependency(filename, dependency_type):
    if not os.path.exists(filename):
        msg = f"File {filename} does not exist. "
        if dependency_type == "samples":
            msg = msg + "Please run the program with --mode=sample"
        elif dependency_type == "state_map":
            msg = msg + "Please run the program with --mode=build"
        elif dependency_type == "qtable":
            msg = msg + "Please run the program with --mode=train"

        raise Exception(msg)


def check_file_exists(filename):
    if os.path.exists(filename):
        overwrite = input(
            f"File {filename} already exists. Overwrite? (y/n) "
        )

        if overwrite.lower() == "n":
            exit(0)


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
        "--results_filename",
        type=str,
        default="results.csv",
        help="filename to store the results in"
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
        default=0,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed to run the environment with for reproducibility"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="size of the sample size for the state map"
    )

    args = parser.parse_args()
    print(args)

    if args.mode == "run":
        env = gym.make(
            GYM_ENV_NAME,
            frameskip=FRAME_SKIP,
            render_mode=args.render_mode,
            repeat_action_probability=REPEAT_ACTION_PROBABILITY,
            obs_type=DEFAULT_OBS_TYPE
        )

        check_file_dependency(args.state_map_filename, "state_map")
        state_map = StateMap.load(args.state_map_filename)

        if args.agent_type == "random":
            agent = CentipedeAgent(env, state_map)
        elif args.agent_type == "policy":
            check_file_dependency(args.qtable_filename, "qtable")

            qtable = pd.read_csv(args.qtable_filename, header=None).to_numpy()
            agent = CentipedeAgent(
                env, state_map, qtable=qtable
            )

        scores = []
        steps = []
        e_times = []
        state_action_counts = []

        for _ in tqdm(range(args.episodes), ncols=100):
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
        results_df.to_csv(args.results_filename, index=False, header=True)

        print(
            f"Score: {np.mean(scores)}," + \
                f" Steps: {np.mean(steps)}, Time: {np.mean(e_time)}"
        )
    elif args.mode == "train":
        check_file_dependency(args.state_map_filename, "state_map")

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
            repeat_action_probability=REPEAT_ACTION_PROBABILITY,
            obs_type=DEFAULT_OBS_TYPE
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

        check_file_exists(args.qtable_filename)
        qlearner.save_qtable(filename=args.qtable_filename)

        print("QTable saved to: ", args.qtable_filename)
    elif args.mode == "build":
        check_file_dependency(args.samples_filename, "samples")
        check_file_exists(args.state_map_filename)
        env_samples_df = pd.read_csv(args.samples_filename, header=None)

        state_map, clustered_samples = StateMap.build(
            env_samples_df,
            n_clusters=args.clusters,
            size=args.size
        )

        state_map.save(filename=args.state_map_filename)
        clustered_samples.to_csv(
            '{}_cluster_samples.csv'.format(args.clusters),
            index=False
        )

        print("State map saved to: ", args.state_map_filename)
        print(
            "Clustered samples saved to: ",
            '{}_cluster_samples.csv'.format(args.clusters)
        )
    elif args.mode == "findk":
        check_file_dependency(args.samples_filename, "samples")

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
        check_file_exists(args.samples_filename)

        env = gym.make(
            GYM_ENV_NAME,
            frameskip=FRAME_SKIP,
            render_mode=DEFAULT_RENDER_MODE,
            repeat_action_probability=REPEAT_ACTION_PROBABILITY,
            obs_type=DEFAULT_OBS_TYPE
        )

        samples_df = SampleEnv.run(env, n_episodes=args.episodes)
        samples_df.to_csv(args.samples_filename, index=False)
        print("Samples saved to: ", args.samples_filename)
    else:
        parser.print_help()
