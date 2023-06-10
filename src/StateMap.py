import collections
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from warnings import simplefilter
import time


class StateMap:
    @staticmethod
    def build(samples_df, n_clusters=200, filename=None):
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        _ = kmeans.fit_predict(samples_df)

        state_map = StateMap()
        for cluster_center in kmeans.cluster_centers_:
            state_map.add_state(tuple(cluster_center))

        state_map.save(filename=filename)

        return state_map


    @staticmethod
    def load(filepath):
        df = pd.read_csv(filepath, sep=",", header=None)
        sm = StateMap()
        [sm.add_state(tuple(state[1])) for state in df.iterrows()]
        return sm


    def __init__(self):
        self.states_to_index = {}
        self.indices_to_state = {}
        self.rewards = {}
        self.states_to_cluster = {}
        self.cluster_estimator = None
        self.neighbor_estimator = None


    def add_state(self, state, reward=0):
        if isinstance(state, collections.abc.Hashable):
            if state not in self.states_to_index.keys():
                state_index = len(self.states_to_index)
                self.states_to_index[state] = state_index
                self.indices_to_state[state_index] = state
                self.rewards[state_index] = reward
        else:
            raise Exception('State object is not hashable')


    def predict(self, observation):
        simplefilter(action='ignore', category=FutureWarning)

        state_idx = self.get_index(tuple(observation))
        if state_idx != None:
            return state_idx

        if not(self.neighbor_estimator):
            self.reset_estimator()

        state_idx = self.neighbor_estimator.predict(np.array(observation).reshape(1, -1))

        self.states_to_index[tuple(observation)] = state_idx[0]

        return state_idx[0]


    def reset_estimator(self):
        knn_clf = KNeighborsClassifier(n_neighbors=1)
        knn_clf.fit(self.to_array(), self.indices())
        self.neighbor_estimator = knn_clf


    def states(self):
        return list(self.states_to_index.keys())


    def get_state(self, index):
        return self.indices_to_state.get(index, None)


    def get_index(self, state):
        if type(state) != tuple:
            state = tuple(state)

        return self.states_to_index.get(state, None)


    def to_array(self):
        return np.array(list(self.states_to_index.keys()))


    def indices(self):
        return list(self.indices_to_state)


    def size(self):
        return len(self.indices_to_state)


    def to_df(self):
        return pd.DataFrame(list(self.states_to_index.keys()))


    def save(self, folder=".", filename=None):
        if filename:
            filepath = os.path.join(folder, filename)
        else:
            filepath = os.path.join(folder, "state_map_{0}.csv".format(time.time()))

        df = self.to_df()
        df.to_csv(filepath, index=False, sep=",", header=False)

        return filepath