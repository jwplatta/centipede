import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_no_generalization_describe_df = pd.read_csv(
  "./results/results_no_generalization.csv", header=0
).describe()
results_20_samples_describe_df = pd.read_csv(
  "./results/results_20_samples.csv", header=0
).describe()
results_200_samples_describe_df = pd.read_csv(
  "./results/results_200_samples.csv", header=0
).describe()
results_20_clusters_describe_df = pd.read_csv(
  "./results/results_20_clusters.csv", header=0
).describe()
results_200_clusters_describe_df = pd.read_csv(
  "./results/results_200_clusters.csv", header=0
).describe()

mean_scores = [
  results_no_generalization_describe_df.loc['mean', 'score'],
  results_20_samples_describe_df.loc['mean', 'score'],
  results_200_samples_describe_df.loc['mean', 'score'],
  results_20_clusters_describe_df.loc['mean', 'score'],
  results_200_clusters_describe_df.loc['mean', 'score']
]

mean_steps = [
  results_no_generalization_describe_df.loc['mean', 'steps'],
  results_20_samples_describe_df.loc['mean', 'steps'],
  results_200_samples_describe_df.loc['mean', 'steps'],
  results_20_clusters_describe_df.loc['mean', 'steps'],
  results_200_clusters_describe_df.loc['mean', 'steps']
]

min_scores = [
  results_no_generalization_describe_df.loc['min', 'score'],
  results_20_samples_describe_df.loc['min', 'score'],
  results_200_samples_describe_df.loc['min', 'score'],
  results_20_clusters_describe_df.loc['min', 'score'],
  results_200_clusters_describe_df.loc['min', 'score']
]

max_scores = [
  results_no_generalization_describe_df.loc['max', 'score'],
  results_20_samples_describe_df.loc['max', 'score'],
  results_200_samples_describe_df.loc['max', 'score'],
  results_20_clusters_describe_df.loc['max', 'score'],
  results_200_clusters_describe_df.loc['max', 'score']
]

min_steps = [
  results_no_generalization_describe_df.loc['min', 'steps'],
  results_20_samples_describe_df.loc['min', 'steps'],
  results_200_samples_describe_df.loc['min', 'steps'],
  results_20_clusters_describe_df.loc['min', 'steps'],
  results_200_clusters_describe_df.loc['min', 'steps']
]

max_steps = [
  results_no_generalization_describe_df.loc['max', 'steps'],
  results_20_samples_describe_df.loc['max', 'steps'],
  results_200_samples_describe_df.loc['max', 'steps'],
  results_20_clusters_describe_df.loc['max', 'steps'],
  results_200_clusters_describe_df.loc['max', 'steps']
]

results_names = [
  "No Generalization",
  "20 Samples",
  "200 Samples",
  "20 Clusters",
  "200 Clusters"
]

###################################
### Mean Scores and Steps Plots ###
###################################
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].bar(results_names, mean_scores, color='violet')
axs[0].set_title("Mean Scores")
axs[0].set_ylabel("Score")
axs[0].set_xlabel("State Representation")
axs[0].set_xticklabels(results_names, rotation=45, fontsize=8)
axs[0].grid(color='k', linestyle='-', alpha=0.1)

axs[1].bar(results_names, mean_steps, color='darkviolet')
axs[1].set_title("Mean Steps")
axs[1].set_ylabel("Steps")
axs[1].set_xlabel("State Representation")
axs[1].set_xticklabels(results_names, rotation=45, fontsize=8)
axs[1].grid(color='k', linestyle='-', alpha=0.1)

fig.tight_layout()
fig.savefig(
  "mean_scores_and_steps.png",
  bbox_inches='tight'
)
plt.show()

##########################################
### Max and Min Scores and Steps Plots ###
##########################################
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

bar_width = 0.35
axs[0].bar(
  np.arange(len(results_names)),
  max_scores,
  bar_width,
  color='blue',
  label="Max Scores"
)
axs[0].bar(
  np.arange(len(results_names)) + bar_width,
  min_scores,
  bar_width,
  color='orange',
  label="Min Scores"
)
axs[0].set_title("Max and Min Scores")
axs[0].set_ylabel("Score")
axs[0].set_xlabel("State Representation")
axs[0].set_xticks(np.arange(len(results_names)) + bar_width / 2)
axs[0].set_xticklabels(results_names, rotation=45, fontsize=8)
axs[0].legend()
axs[0].grid(color='k', linestyle='-', alpha=0.1)

axs[1].bar(
  np.arange(len(results_names)),
  max_steps,
  bar_width,
  color='blue',
  label="Max Steps"
)
axs[1].bar(
  np.arange(len(results_names)) + bar_width,
  min_steps,
  bar_width,
  color='orange',
  label="Min Steps"
)
axs[1].set_title("Max and Min Steps")
axs[1].set_ylabel("Steps")
axs[1].set_xlabel("State Representation")
axs[1].set_xticks(np.arange(len(results_names)) + bar_width / 2)
axs[1].set_xticklabels(results_names, rotation=45, fontsize=8)
axs[1].legend()
axs[1].grid(color='k', linestyle='-', alpha=0.1)

fig.tight_layout()
fig.savefig(
  "max_min_scores_steps.png",
  bbox_inches='tight'
)
plt.show()

