import numpy as np
import time


class CentipedeAgent:
    def __init__(self, env, state_map, qtable=np.array([])):
        """
        Initialize the agent
        :param env: The environment to run the agent on
        :param state_map: The state map to use
        :param qtable: The Q-table to use
        """
        self.env = env
        self.state_map = state_map
        if qtable.any():
            self.policy = qtable.argmax(axis=1)
        else:
            self.policy = np.array([])


    def run(self, render=False):
        """
        Run the agent on the environment
        :param render: Whether to render the environment
        :return: Score, steps, elapsed time, state-action counts
        """
        state_action_counts = np.zeros((self.state_map.size(), self.env.action_space.n))
        steps = 0
        score = 0
        done = False
        start_time = time.time()

        observation = self.env.reset()[0]

        if render and self.env.render_mode == 'human':
            self.env.render()

        if self.state_map:
            state = self.state_map.predict(observation)

        while not done:
            if self.policy.any():
                action = self.policy[state]
            else:
                action = self.env.action_space.sample()

            state_action_counts[state, action] += 1

            observation, reward, terminated, truncated, info = self.env.step(action)
            if self.state_map:
                state = self.state_map.predict(observation)
            else:
                state = observation

            score += reward
            done = terminated or truncated

            if not(done):
                steps += 1

        e_time = time.time() - start_time
        self.env.close()

        self.score = score
        self.steps = steps
        self.e_time = e_time

        return score, steps, e_time, state_action_counts