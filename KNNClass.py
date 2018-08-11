# -*- coding: utf8 -*-


import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean


class BaseEstimator(object):

    X = None
    y = None
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('Number of feature must be > 0')

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('Missed required argument y')
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if y.size == 0:
            raise ValueError('Number of feature must be > 0')

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            raise NotImplementedError()
        else:
            raise ValueError('You must call "fit" before "predict')


class KNNBase(BaseEstimator):

    def __init__(self, k=5, dist_fun=euclidean):
        self.k = None if k == 0 else k
        self.dist_fun = dist_fun

    def aggregate(self, neighbor_targets):
        raise NotImplementedError()

    def _predict(self, X=None):
        predictions = [self._predict_x(x) for x in X]

        return np.array(predictions)

    def _predict_x(self, x):

        distance = (self.dist_fun(x, example) for example in self.X)

        neighbors = sorted(((dist, target) for (dist, target) in zip(distance, self.y)),
                           key=lambda x: x[0])

        neighbors_target = [target for (_, target) in neighbors[:self.k]]

        return self.aggregate(neighbors_target)


class KNNClassifier(KNNBase):

    def aggregate(self, neighbors_target):
        most_common_label = Counter(neighbors_target).most_common(1)[0][0]
        return most_common_label


class KNNRegressor(KNNBase):

    def aggregate(self, neighbors_target):
        return np.array(neighbors_target)
