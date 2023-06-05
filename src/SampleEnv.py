import gymnasium as gym
from tqdm import tqdm
import numpy as np
import pandas as pd
from .constants import ENV_SAMPLES_FILENAME, GYM_ENV_NAME, FRAME_SKIP, RENDER_MODE, REPEAT_ACTION_PROBABILITY

class SampleEnv:
  @staticmethod
  def run(n_episodes=100):
      env = gym.make(
        GYM_ENV_NAME,
        frameskip=FRAME_SKIP,
        render_mode=None,
        repeat_action_probability=REPEAT_ACTION_PROBABILITY
      )
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
      samples_df.to_csv(ENV_SAMPLES_FILENAME, index=False)

      return ENV_SAMPLES_FILENAME