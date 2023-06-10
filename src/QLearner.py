import numpy as np
from tqdm import tqdm
import time
import pickle
import os
# from scipy.sparse import csr_matrix
import pandas as pd


"""
Implemntation based on https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
"""
class QLearner:
    @staticmethod
    def load_progress(filepath, env, state_map=None):
        with open(filepath, 'rb') as file:
            state_dict = pickle.load(file)

        ql = QLearner(
            None,
            None,
            None,
            None,
            env,
            None,
            state_dict=state_dict,
            state_map=state_map
        )

        return ql


    def __init__(self,
        learning_rate,
        gamma,
        state_size,
        action_size,
        env,
        qtable_threshold=0.001,
        min_episodes=float('inf'),
        **kwargs
    ):
        if 'state_dict' in kwargs:
            self.__set_state__(kwargs['state_dict'])
            self.env = env
            self.state_map = kwargs.get('state_map', None)
        else:
            self.learning_rate = learning_rate
            self.qtable_threshold = qtable_threshold
            self.gamma = gamma
            self.state_size = state_size
            self.action_size = action_size
            self.map_size = state_size
            self.env = env

            self.rng = np.random.default_rng(101)
            self.seed = 101
            self.state_map = kwargs.get('state_map', None)
            self.reward_model = kwargs.get('reward_model', None)

            self.epsilon = kwargs.get('epsilon', 0.1)
            self.epsilon_decay = kwargs.get('epsilon_decay', None)
            self.progress_file = kwargs.get('progress_file', False)
            self.n_runs = kwargs.get('n_runs', 1)
            self.min_episodes = min_episodes

            # For progress
            self.start_run = 0
            self.start_episode = 0

            self.rewards = np.zeros((self.min_episodes, self.n_runs))
            self.qtable_deltas = np.zeros((self.min_episodes, self.n_runs))
            self.steps = np.zeros((self.min_episodes, self.n_runs))
            self.qtables = np.zeros((self.n_runs, self.state_size, self.action_size))
            self.all_states = []
            self.all_actions = []
            self.transition_counts = {}

            self.__reset_qtable()


    def fit(self):
        # NOTE: Run several times to account for stochasticity
        for run in range(self.start_run, self.n_runs):
            self.__reset_qtable()
            prev_qtable = np.array([])
            self.start_exploit_episode = None

            for episode in tqdm(
                range(self.start_episode, self.min_episodes),
                ncols=100,
                desc="Run {0}/{1}".format(run, self.n_runs)
            ):
                state = self.env.reset(seed=self.seed)[0]

                if self.state_map:
                    state = self.state_map.predict(state)

                step = 0
                done = False
                total_rewards = 0

                while not done:
                    action, epsilon = self.__pick_action(
                        action_space=self.env.action_space,
                        state=state,
                        qtable=self.qtable,
                        episode=episode
                    )

                    if epsilon == 0.0 and not(self.start_exploit_episode):
                        self.start_exploit_episode = episode

                    # Log all states and actions
                    # self.all_states.append(state)
                    # self.all_actions.append(action)

                    # Take the action (a) and observe the outcome
                    # state(s') and reward (r)
                    observation, reward, terminated, truncated, _ = self.env.step(
                        action
                    )


                    if self.state_map:
                        new_state = self.state_map.predict(observation)
                    else:
                        new_state = observation

                    # NOTE: update reward if reward model provided
                    if self.reward_model:
                        for next_state_tuple in self.reward_model[state][action]:
                            if next_state_tuple[1] == new_state:
                                reward = next_state_tuple[2]

                    done = terminated or truncated

                    # NOTE: tracking state transitions for transition model
                    if state not in self.transition_counts:
                        self.transition_counts[state] = {}

                    if action not in self.transition_counts[state]:
                        self.transition_counts[state][action] = {}

                    if new_state not in self.transition_counts:
                        self.transition_counts[new_state] = {}

                    if new_state not in self.transition_counts[state][action]:
                        self.transition_counts[state][action][new_state] = {
                          'count': 0, 'terminal': False, 'total_reward': 0
                        }


                    self.transition_counts[state][action][new_state]['count'] += 1
                    self.transition_counts[state][action][new_state]['terminal'] = done
                    self.transition_counts[state][action] \
                        [new_state]['total_reward'] += reward

                    # NOTE: update Q table and store update
                    q_update, delta = self.__update(state, action, reward, new_state)
                    self.qtable[state, action] = q_update

                    total_rewards += reward
                    step += 1

                    # NOTE: update the current state
                    state = new_state

                # Log all rewards and steps
                self.rewards[episode, run] = total_rewards
                self.steps[episode, run] = step
                self.start_episode += 1

                if self.progress_file and (episode % 1000) == 0:
                    self.save_progress()

                # # NOTE: check for convergence
                if not(prev_qtable.any()):
                    qtable_delta = self.qtable_threshold * 1.1
                else:
                    qtable_delta = np.max(np.abs(prev_qtable - self.qtable))
                    self.qtable_deltas[episode, run] = qtable_delta

                # if (episode > self.min_episodes):
                if (qtable_delta < self.qtable_threshold \
                    and episode > self.min_episodes):
                    print(
                        'Converged episode {0} - Q-table delta {1} - last epsilon {2}' \
                            .format(episode, qtable_delta, epsilon)
                    )
                    break
                else:
                    prev_qtable = self.qtable.copy()

            self.qtables[run, :, :] = self.qtable.copy()
            self.start_run += 1

            # # NOTE: reset for next run
            self.start_episode = 0

        # self.qtable = self.qtables.mean(axis=0)
        # qtable_mean = qtables.mean(axis=0)
        # rewards = qtable_mean.max(axis=1)
        # self.best_actions = np.argmax(self.qtable, axis=1) \
        #   .reshape(self.map_size, self.map_size)
        # self.qtable_val_max = self.qtable.max(axis=1) \
        #   .reshape(self.map_size, self.map_size)

        return self.rewards, self.steps, self.qtables, self.all_states, \
            self.all_actions, self.transition_counts


    def __update(self, state, action, reward, new_state):
        target = reward + self.gamma * np.max(self.qtable[new_state, :])
        delta = target - self.qtable[state, action]
        q_update = self.qtable[state, action] + self.learning_rate * delta

        return q_update, delta


    def __pick_action(self, action_space, state, qtable, episode):
        """
        Implements an epsilon greedy approach to the explore-exploit trade-off.
        """
        explore_exploit_tradeoff = self.rng.uniform(0, 1)

        # NOTE: Exploration
        if self.epsilon_decay:
            epsilon = self.epsilon_decay.get(episode)
        else:
            epsilon = self.epsilon

        if explore_exploit_tradeoff < epsilon:
            action = action_space.sample()
        else:
            # NOTE: Exploit
            # NOTE: Break ties randomly
            # If all actions are the same for this state, choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

        return action, epsilon


    def __reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))


    def save_qtable(self, filename=None, path="./"):
        qtable_df = pd.DataFrame(self.qtable)

        if filename:
            filepath = os.path.join(path, filename)
        else:
            filepath = os.path.join(path, "qtable_{}.csv".format(time.time()))

        qtable_df.to_csv(filepath, index=False, sep=",", header=False)

        return filepath


    def save_progress(self):
        curr_state = self.__get_state__()

        # NOTE: rotate progress file
        if os.path.exists(self.progress_file):
            os.rename(self.progress_file, "backup_" + self.progress_file)

        import gc
        gc.disable()

        with open(self.progress_file, 'wb') as file:
            pickle.dump(curr_state, file)


    def __get_state__(self):
        state_dict = {}
        state_dict['rewards'] = self.rewards
        state_dict['steps'] = self.steps
        state_dict['learning_rate'] = self.learning_rate
        state_dict['gamma'] = self.gamma
        state_dict['state_size'] = self.state_size
        state_dict['action_size'] = self.action_size
        state_dict['map_size'] = self.map_size
        state_dict['rng'] = self.rng
        state_dict['reward_model'] = self.reward_model
        state_dict['epsilon'] = self.epsilon
        state_dict['epsilon_decay'] = self.epsilon_decay
        state_dict['progress_file'] = self.progress_file
        state_dict['start_run'] = self.start_run
        state_dict['start_episode'] = self.start_episode
        state_dict['qtable'] = self.qtable
        state_dict['qtables'] = self.qtables
        state_dict['transition_counts'] = self.transition_counts # json.dumps()
        state_dict['qtable_threshold'] = self.qtable_threshold
        state_dict['min_episodes'] = self.min_episodes

        return state_dict


    def __set_state__(self, state_dict):
        self.rewards = state_dict['rewards']
        self.learning_rate = state_dict['learning_rate']
        self.gamma = state_dict['gamma']
        self.qtable_threshold = state_dict.get('qtable_threshold', None)
        self.min_episodes = state_dict.get('min_episodes', None)
        self.state_size = state_dict['state_size']
        self.action_size = state_dict['action_size']
        self.map_size = state_dict['map_size']
        self.rng = state_dict['rng']
        self.reward_model = state_dict['reward_model']
        self.epsilon = state_dict['epsilon']
        self.epsilon_decay = state_dict['epsilon_decay']
        self.progress_file = state_dict['progress_file']
        self.start_run = state_dict['start_run']
        self.start_episode = state_dict['start_episode']
        self.steps = state_dict['steps']
        self.qtable = state_dict['qtable']
        self.qtables = state_dict['qtables']
        self.transition_counts = state_dict['transition_counts']