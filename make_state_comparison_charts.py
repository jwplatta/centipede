from src import (
    StateMap,
    SampleEnv,
    GYM_ENV_NAME,
    REPEAT_ACTION_PROBABILITY,
    FRAME_SKIP,
    DEFAULT_OBS_TYPE
)
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
import pandas as pd

env = gym.make(
    GYM_ENV_NAME,
    frameskip=FRAME_SKIP,
    render_mode="rgb_array",
    repeat_action_probability=REPEAT_ACTION_PROBABILITY,
    obs_type=DEFAULT_OBS_TYPE
)

state_map = StateMap.load("./results/state_map_20_samples.csv")
target_state = 19
max_sample_count = 3

##############################
### Centipede State Images ###
##############################
# TODO: uncomment after saving images
# sample_count = SampleEnv.record_states(
#   env, state_map, target_state, max_sample_count=max_sample_count
# )

# if sample_count < max_sample_count:
#     print("Not enough samples")
#     exit()

state_filenames = [
  "state_{}_{}.pkl".format(target_state, sample) for sample in range(max_sample_count)
]

fig, axs = plt.subplots(1, max_sample_count, figsize=(12, 5))
for idx, filename in enumerate(state_filenames):
    with open(filename, 'rb') as f:
        rendered_env = pickle.load(f)

    axs[idx].imshow(rendered_env)
    axs[idx].axis('off')

fig.suptitle("Examples from State Cluster {}".format(target_state))
fig.tight_layout()
fig.savefig(
    "state_cluster_examples.png",
    bbox_inches='tight'
)
plt.show()

##############################
### Compare State Clusters ###
##############################
clustered_samples = pd.read_csv("./results/20_cluster_samples.csv", header=0)

samples_per_cluster = clustered_samples['cluster'].value_counts().sort_index()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].bar(samples_per_cluster.index, samples_per_cluster)
axs[0].set_xticks(samples_per_cluster.index)
axs[0].set_xticklabels(samples_per_cluster.index)
axs[0].set_xlabel("State Cluster")
axs[0].set_ylabel("Sample Count")
axs[0].grid(color='k', linestyle='-', alpha=0.1)

pca = PCA(n_components=2)
samples_transformed = pca.fit_transform(clustered_samples.drop('cluster', axis=1))
samples_transformed_df = pd.DataFrame(samples_transformed)
samples_transformed_df['cluster'] = clustered_samples['cluster']

selected_clusters = [0, 10, 15, 19] #list(range(1, 7))
samples_subset = samples_transformed_df[
    samples_transformed_df['cluster'].isin(selected_clusters)
].reset_index(drop=True)
unique_clusters = samples_subset['cluster'].unique()

scatter = axs[1].scatter(
    samples_subset.iloc[:, 0],
    samples_subset.iloc[:, 1],
    c=samples_subset['cluster'],
    alpha=0.6
)

axs[1].set_xlabel("1st principal component")
axs[1].set_ylabel("2nd principal component")

handles, labels = scatter.legend_elements(
    prop='colors',
    alpha=0.6,
)
legend2 = axs[1].legend(
    handles, unique_clusters, loc="upper left", title="State Cluster"
)
axs[1].grid(color='k', linestyle='-', alpha=0.1)

fig.suptitle("State Cluster Comparison")

fig.savefig(
    "state_cluster_comparison.png",
    bbox_inches='tight'
)
plt.show()