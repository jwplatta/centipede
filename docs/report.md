# Centipede

## Setup

- follow instructions here to setup Atari games for Gymnasium: https://gymnasium.farama.org/environments/atari/#atari

```sh
$ pip install "gymnasium[atari, accept-rom-license]"
```

## Run Experiments

1. Build the StateMap
   1. Sampling from the env
   2. Find an optimal number for the clusters for the sampled env states
   3. Get the cluster centers
2. Learn the qtable
3. Run an agent with the qtable and compare the average scores with a random agent