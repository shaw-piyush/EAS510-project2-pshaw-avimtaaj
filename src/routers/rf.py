"""Random forest router: a check on the gradient boosting result (PRD 8.5).

Score = share of trees voting "cheap model is correct" (predict_proba).

Why a second tree model: boosting and random forests build trees in
different ways (one after another fixing errors vs. many independent trees
averaged). If both land close together, the result is about trees on these
features, not one particular algorithm's settings.

Design choices:
- No scaler, for the same reason as HGB: tree splits ignore feature scale.
- n_estimators=300: enough trees that the averaged score is stable; more
  trees only cost time.
- max_depth and min_samples_leaf limit how closely each tree can fit single
  training prompts, which matters here because correctness is noisy (the
  baseline only slightly beats random). They're tuned on validation by
  Piyush's runner.
- n_jobs=-1 uses every CPU core; random_state=seed makes the bootstrap
  samples and feature choices the same for a seed.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.routers.base import Router, proba_of_one


class RFRouter(Router):
    name = "rf"

    def __init__(self, seed, n_estimators=300, max_depth=None, min_samples_leaf=1, n_jobs=-1):
        super().__init__(seed, n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, n_jobs=n_jobs)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=seed,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def score(self, X):
        # With n_jobs=-1 the threads add the trees' votes in a varying order,
        # so the same forest can give scores that differ in the 16th decimal
        # (checked: up to 3e-16). The routing curve treats equal scores as one
        # tied block, so that noise could split true ties and make a seed's
        # results not exactly repeatable. Rounding to 10 decimals removes it;
        # real score differences between prompts are far larger than 1e-10.
        return np.round(proba_of_one(self.model, X), 10)
