import matplotlib.pyplot as plt
import numpy as np


class KMeansFindKPlot():
    def __init__(self, experiment):
        self.k_range = experiment.k_range
        self.wcss = experiment.wcss
        self.bcss = experiment.bcss
        self.cluster_size_stds = experiment.cluster_size_stds
        self.metric_scores = []


    def plot(self, axs=np.array([]), figsize=(20,4), title=None, ymin=None, ymax=None, title_desc=None):
        if axs.any():
            fig = None
        else:
            fig, axs = plt.subplots(1, 3, figsize=figsize)

        plt.style.use('seaborn')

        axs[0].plot(self.k_range, self.wcss)
        axs[0].set_xlabel('Number of Clusters')
        axs[0].set_ylabel('Distance')
        if title_desc:
            axs[0].set_title('Within Cluster Sum of Squares ({0})'.format(title_desc))
        else:
            axs[0].set_title('Within Cluster Sum of Squares')

        axs[1].plot(self.k_range, self.bcss)
        axs[1].set_xlabel('Number of Clusters')
        axs[1].set_ylabel('Distance')
        if title_desc:
            axs[1].set_title('Between Cluster Sum of Squares ({0})'.format(title_desc))
        else:
            axs[1].set_title('Between Cluster Sum of Squares')

        axs[2].plot(self.k_range, self.cluster_size_stds)
        axs[2].set_xlabel('Number of Clusters')
        axs[2].set_ylabel('Std')
        if title_desc:
            axs[2].set_title('Cluster Size Variance ({0})'.format(title_desc))
        else:
            axs[2].set_title('Cluster Size Variance')


        return fig, axs