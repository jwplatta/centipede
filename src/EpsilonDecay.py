class EpsilonDecay:
    def __init__(
        self,
        n_episodes=0,
        decay=0.0,
        min_epsilon=0.0,
        max_epsilon=0.0,
        strategy='constant'
    ):
        """
        Strategies:
        1. Greedy: max_epsilon=0.0
        2. Greedy epsilon: max_epsilon>0.0 and decay=0.0
        3. Epsilon decay: max_epsilon>0.0 and decay>0.0
        """
        self.min_epsilon = max(min_epsilon, 0.0)
        self.max_epsilon = max_epsilon
        self.__decay = decay
        self.__epsilon = max_epsilon
        self.epsilons = [self.__epsilon]
        self.strategy = strategy
        self.n_episodes = n_episodes

        if n_episodes == 0:
            self.decay_steps = 1_000 * decay
        else:
            self.decay_steps = n_episodes * decay

        if self.strategy != 'constant' and n_episodes == 0:
            raise Exception('n_episodes cannot be zero.')


    def get(self, episode):
        if self.max_epsilon == 0.0 or (self.max_epsilon > 0.0 and self.__decay == 0.0):
            return self.max_epsilon
        else:
            return max(
                self.max_epsilon * (1.0 - episode / self.decay_steps),
                max(0.0, self.min_epsilon)
            )


    def name(self):
        if self.max_epsilon == 0.0:
            return "greedy"
        elif self.__decay > 0.0:
            return "epsilon={0} duration={1}".format(self.max_epsilon, self.__decay)
        else:
            return "greedy epsilon={0}".format(self.max_epsilon)


    def __repr__(self):
        return self.name()