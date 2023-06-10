# State Generalization for Centipede

## Summary

TODO

## Setup

```sh
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip install - requirements.txt
```

## Example Usage

Sample the Centipede environment for 100 episodes:
```
python centipede.py --mode sample --episodes 100 --samples_filename "samples.csv"
```

Find the optimal number of clusters:
```
python centipede.py --mode findk --k_range "10,100,10" --samples_filename "samples.csv"
```

Build the state map:
```
python centipede.py --mode build --clusters 20 --samples_filename "samples.csv" --state_map_filename "state_map.csv"
```

Train for 5 runs and 100 episodes per run:
```
python centipede.py --mode train --episodes 100 --runs 5 --state_map_filename "state_map.csv" --qtable_filename "qtable.csv"
```

Run the trained agent for 10 episodes using the Q-table:
```
python centipede.py --mode run --agent_type policy --episodes 10
```