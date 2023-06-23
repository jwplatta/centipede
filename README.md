# State Generalization for Centipede

## Summary

Repo contains code for an example of generalizing a large state space (Centipede). The generalization is built using supervised (k-nearest neighbors) and unsupervised (k-means clustering) learning. The agent is trained using Q-learning. A more detailed description of the approach can be found [here](https://jwplatta.github.io/machine-learning/reinforcement-learning/2023/06/19/generalizing-centipede-game-states-for-reinforcement-learning.html).

## Setup

```sh
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Example Usage

Sample the Centipede environment for 100 episodes:
```sh
python centipede.py --mode sample --episodes 100 --samples_filename "samples.csv"
```

Find the optimal number of clusters:
```sh
python centipede.py --mode findk --k_range "10,100,10" --samples_filename "samples.csv"
```

Build the state map:
```sh
python centipede.py --mode build --clusters 20 --samples_filename "samples.csv" --state_map_filename "state_map.csv"
```

Train for 5 runs and 100 episodes per run:
```sh
python centipede.py --mode train --episodes 100 --runs 5 --state_map_filename "state_map.csv" --qtable_filename "qtable.csv"
```

Run the trained agent for 10 episodes using the Q-table:
```sh
python centipede.py --mode run --agent_type policy --episodes 10 --qtable_filename "qtable.csv"
```