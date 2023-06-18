from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

class SampleEnv:
    @staticmethod
    def run(env, n_episodes=100):
        samples = []
        episode_steps = []

        for _ in tqdm(range(n_episodes), ncols=100):
          _ = env.reset()
          done = False
          steps = 0

          while not done:
              action = env.action_space.sample()
              new_observation, _, terminated, truncated, _ = env.step(action)
              done = terminated or truncated

              if type(new_observation) == np.ndarray:
                  state_key = tuple(new_observation.tolist())
              else:
                  state_key = new_observation

              steps += 1
              samples.append(state_key)

          episode_steps.append(steps)

        episode_steps = np.array(episode_steps)
        env.close()

        samples_df = pd.DataFrame(samples)
        return samples_df, episode_steps


    @staticmethod
    def record_states(env, state_map, target_state, max_sample_count=3):
        sample_count = 0
        while sample_count < max_sample_count:
            _ = env.reset()
            done = False

            while not done:
                action = env.action_space.sample()
                observation, _, terminated, truncated, _ = env.step(action)
                state = state_map.predict(observation)
                done = terminated or truncated

                if state == target_state:
                    filepath = "state_{}_{}.pkl".format(state, sample_count)

                    with open(filepath, 'wb') as f:
                        pickle.dump(env.render(), f)
                        sample_count += 1


        return sample_count