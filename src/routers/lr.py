"""L2 logistic regression: the baseline router (PRD 7.5).

Scaling sits inside the pipeline, so it's fit on the training rows only.
It matters for the hand-built features, whose counts run from 0 to tens of
thousands; on the unit-length TF-IDF and embedding vectors it's harmless.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.routers.base import Router, proba_of_one


class LRRouter(Router):
    name = "lr"

    def __init__(self, seed, C=1.0, max_iter=2000):
        super().__init__(seed, C=C, max_iter=max_iter)
        # lbfgs's default penalty is L2. The seed only matters for solvers
        # that shuffle, but setting it keeps every router seeded the same way.
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=max_iter, random_state=seed),
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def score(self, X):
        return proba_of_one(self.model, X)
