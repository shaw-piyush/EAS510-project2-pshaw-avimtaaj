"""kNN router (PRD 7.5): score = share of the k nearest training prompts
that the cheap model got right (distance-weighted if weights="distance").

kNN is the method a recent paper found competitive with complex learned
routers (arXiv:2505.12601), which is why it's in the comparison.

The distance metric and scaling depend on the feature set and come from
config.ROUTER_FEATURE_PARAMS: cosine on TF-IDF and embeddings, standardized
Euclidean on the hand-built features. Scaling sits inside the pipeline, so
it's fit on the training rows only.

kNN has no randomness: neighbours are found by exact search, and equal
distances are broken by training-row order, which is fixed by the split.
The seed is kept only so every router has the same interface.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.routers.base import Router, proba_of_one


class KNNRouter(Router):
    name = "knn"

    def __init__(self, seed, n_neighbors=15, weights="uniform", metric="cosine", scale=False):
        super().__init__(seed, n_neighbors=n_neighbors, weights=weights, metric=metric,
                         scale=scale)
        # sklearn's tree indexes don't support cosine, so cosine needs brute
        # force (PRD 7.5); "auto" lets sklearn pick a tree for Euclidean.
        algorithm = "brute" if metric == "cosine" else "auto"
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric,
                                   algorithm=algorithm)
        self.model = make_pipeline(StandardScaler(), knn) if scale else make_pipeline(knn)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def score(self, X):
        return proba_of_one(self.model, X)
