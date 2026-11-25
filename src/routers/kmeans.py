"""k-means cluster routing (PRD 7.5).

Cluster the training prompts; a cluster's score is the share of its
training prompts that the cheap model got right. A new prompt gets the
score of the nearest cluster centre. Every prompt in a cluster gets the
same score, so the scores repeat; the routing curve handles those ties as
blocks (src/metrics.py), and tests/test_routers.py checks that the curve
doesn't depend on the order of tied rows.

Clustering and the per-cluster shares use the training split only (PRD
0.4). Scaling (hand-built features only, config.ROUTER_FEATURE_PARAMS) sits
inside the pipeline, so it's fit on the training rows too.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.routers.base import Router


class KMeansRouter(Router):
    name = "kmeans"

    def __init__(self, seed, n_clusters=25, n_init=10, scale=False):
        super().__init__(seed, n_clusters=n_clusters, n_init=n_init, scale=scale)
        # The seed fixes k-means' random starting centres; n_init restarts
        # keep the best of several, so one bad start can't decide the result.
        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
        self.model = make_pipeline(StandardScaler(), km) if scale else make_pipeline(km)
        self.n_clusters = n_clusters

    def fit(self, X_train, y_train):
        labels = self.model.fit_predict(X_train)
        y = np.asarray(y_train, dtype=float)
        size = np.bincount(labels, minlength=self.n_clusters)
        right = np.bincount(labels, weights=y, minlength=self.n_clusters)
        # A cluster can end up with no training prompts. It then has no
        # evidence of its own, so it gets the overall training share.
        self.overall_ = float(y.mean())
        self.cluster_score_ = np.where(size > 0, right / np.maximum(size, 1), self.overall_)
        self.cluster_size_ = size
        return self

    def score(self, X):
        return self.cluster_score_[self.model.predict(X)]
