import numpy as np
from sklearn.cluster import KMeans
import time

class KMeansFindK:
    def __init__(self, k_range=np.arange(2, 20, 2), init='k-means++', verbose=False, **kwargs):
        self.k_range = k_range
        self.init = init
        self.verbose = verbose
        self.labels = []
        self.models = []
        self.cluster_centers = []
        self.cluster_names = []
        self.fit_times = []
        self.wcss = []
        self.bcss = []
        self.cluster_sizes = []
        self.cluster_size_stds = []
        self.avg_intra_cluster_distances = []
        self.avg_inter_cluster_distances = []


    def run(self, X):
        for n_clusters in self.k_range:
            kmeans = KMeans(n_clusters=n_clusters, init=self.init)

            start_time = time.time()
            kmeans.fit(X)
            fit_time = time.time() - start_time
            self.fit_times.append(fit_time)

            if self.verbose:
                print('fit time {0}'.format(fit_time))

            wcss = kmeans.inertia_
            centroids = kmeans.cluster_centers_
            self.cluster_centers.append(centroids)

            # NOTE: calculate the average intra-cluster sum squared errors for each cluster
            # avg_wcss = wcss / X.shape[0]
            self.labels.append(kmeans.labels_)
            self.cluster_names.append(np.unique(kmeans.labels_).tolist())
            self.wcss.append(wcss)

            # NOTE: compute BCSS
            bcss = []
            cluster_sizes = {}
            overall_mean = np.mean(X, axis=0) # NOTE: calculate the overall mean of the data
            for cluster in range(n_clusters):
                # STEP: get the number of points in the i-th cluster
                cluster_size = np.count_nonzero(kmeans.labels_ == cluster, axis=0)
                cluster_sizes[cluster] = cluster_size

                # STEP: get the i-th cluster centroid
                centroid = kmeans.cluster_centers_[cluster]

                # STEP: calculate the distance between the i-th cluster centroid and the overall mean
                dist = np.linalg.norm(centroid - overall_mean)

                # STEP: add the BCSS contribution from the i-th cluster to the total BCSS
                bcss.append(cluster_size * (dist ** 2))

            self.cluster_size_stds.append(np.std([c_size for _, c_size in cluster_sizes.items()]))
            self.cluster_sizes.append(cluster_sizes)
            self.bcss.append(np.sum(bcss))

            # NOTE: Compute the inter-cluster distances
            inter_cluster_distances = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    inter_cluster_distances.append(distance)

            self.avg_inter_cluster_distances.append(np.mean(inter_cluster_distances))
            self.models.append(kmeans)

            if self.verbose:
                print("\n------------\n{0}".format(kmeans))
                print(
                    'n_clusters: {0} / avg inter cluster dist: {1} / fit_time: {2}'.format(
                        n_clusters, self.avg_inter_cluster_distances[-1], fit_time
                    )
                )

        return self.models