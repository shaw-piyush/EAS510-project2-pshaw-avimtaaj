"""The interface every router follows (agreed with Avimta, PRD 5.3).

A router learns from training prompts whether the cheap model got them
right (y = 1) and gives each new prompt a score: its estimate of
P(cheap model is correct). Avimta's metrics then send the lowest-scored
prompts to the strong model.

    class MyRouter(Router):
        def __init__(self, seed, **params): ...
        def fit(self, X_train, y_train): ...; return self
        def score(self, X): ...   # 1-D array, higher = safer for cheap

Hyperparameters come in as keyword arguments from the config grids, and
`seed` from config.SEEDS, so a run is fully described by config.py.
"""

import numpy as np


class Router:
    name = "base"

    def __init__(self, seed, **params):
        self.seed = seed
        self.params = params

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError


def proba_of_one(model, X):
    """P(y = 1) from a fitted classifier, whichever column holds class 1.

    Looking the column up by class, not position, stays right even if a
    training split only had one class (then P is 0 or 1 everywhere).
    """
    p = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return p[:, classes.index(1)]
    return np.zeros(len(p))
