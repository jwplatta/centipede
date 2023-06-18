#!/bin/bash

#####################################
### Sample Centipede Observations ###
#####################################
python centipede.py --mode sample --episodes 1000 --samples_filename "samples_1000_episodes.csv"

########################
### Build state maps ###
########################
python centipede.py --mode build --samples_filename "samples_1000_episodes.csv" --state_map_filename "state_map_no_generalization.csv"
python centipede.py --mode build --size 20 --samples_filename "samples_1000_episodes.csv" --state_map_filename "state_map_20_samples.csv"
python centipede.py --mode build --size 200 --samples_filename "samples_1000_episodes.csv" --state_map_filename "state_map_200_samples.csv"
python centipede.py --mode build --clusters 20 --samples_filename "samples_1000_episodes.csv" --state_map_filename "state_map_20_clusters.csv"
python centipede.py --mode build --clusters 200 --samples_filename "samples_1000_episodes.csv" --state_map_filename "state_map_200_clusters.csv"

####################
### Train agents ###
####################
python centipede.py \
  --mode train \
  --episodes 1000 \
  --runs 5 \
  --state_map_filename "state_map_no_generalization.csv" \
  --qtable_filename "qtable_no_generalization.csv"
python centipede.py \
  --mode train \
  --episodes 1000 \
  --runs 5 \
  --state_map_filename "state_map_20_samples.csv" \
  --qtable_filename "qtable_20_samples.csv"
python centipede.py \
  --mode train \
  --episodes 1000 \
  --runs 5 \
  --state_map_filename "state_map_200_samples.csv" \
  --qtable_filename "qtable_200_samples.csv"
python centipede.py \
  --mode train \
  --episodes 1000 \
  --runs 5 \
  --state_map_filename "state_map_20_clusters.csv" \
  --qtable_filename "qtable_20_clusters.csv"
python centipede.py \
  --mode train \
  --episodes 1000 \
  --runs 5 \
  --state_map_filename "state_map_200_clusters.csv" \
  --qtable_filename "qtable_200_clusters.csv"


##################
### Run agents ###
##################
python centipede.py \
  --mode run \
  --agent_type policy \
  --episodes 1000 \
  --state_map_filename "state_map_no_generalization.csv" \
  --qtable_filename "qtable_no_generalization.csv" \
  --results_filename "results_no_generalization.csv"
python centipede.py \
  --mode run \
  --agent_type policy \
  --episodes 1000 \
  --state_map_filename "state_map_20_samples.csv" \
  --qtable_filename "qtable_20_samples.csv" \
  --results_filename "results_20_samples.csv"
python centipede.py \
  --mode run \
  --agent_type policy \
  --episodes 1000 \
  --state_map_filename "state_map_200_samples.csv" \
  --qtable_filename "qtable_200_samples.csv" \
  --results_filename "results_200_samples.csv"
python centipede.py \
  --mode run \
  --agent_type policy \
  --episodes 1000 \
  --state_map_filename "state_map_20_clusters.csv" \
  --qtable_filename "qtable_20_clusters.csv" \
  --results_filename "results_20_clusters.csv"
python centipede.py \
  --mode run \
  --agent_type policy \
  --episodes 1000 \
  --state_map_filename "state_map_200_clusters.csv" \
  --qtable_filename "qtable_200_clusters.csv" \
  --results_filename "results_200_clusters.csv"