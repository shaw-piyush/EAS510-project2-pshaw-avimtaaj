"""Histogram gradient boosting router (PRD 8.5).

Score = predicted P(cheap model is correct), from sklearn's
HistGradientBoostingClassifier.

Design choices:
- No scaler: tree splits only compare a feature with a threshold, so the
  scale of a feature doesn't change which split is best. This is why HGB
  can use the raw hand-built counts directly.
- The "histogram" version bins each feature into at most 255 buckets before
  building trees, which makes it fast enough for ~20,000 training prompts
  with 256-384 features on a CPU.
- early_stopping=True holds out part of the TRAINING rows (sklearn's default
  validation_fraction=0.1) and stops adding trees once the loss on them
  stops improving. That holdout comes from train only, so it never sees our
  validation or test split; max_iter=300 is just the upper limit.
- random_state=seed fixes the early-stopping holdout, so a seed gives the
  same model every time.
- Hyperparameters (learning_rate, max_leaf_nodes) are tuned by Piyush's
  runner on validation curve AUC using the "hgb" grids in config.py.
"""

from sklearn.ensemble import HistGradientBoostingClassifier

from src.routers.base import Router, proba_of_one


class HGBRouter(Router):
    name = "hgb"

    def __init__(self, seed, learning_rate=0.1, max_leaf_nodes=31, max_iter=300,
                 early_stopping=True):
        super().__init__(seed, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                         max_iter=max_iter, early_stopping=early_stopping)
        self.model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=seed,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def score(self, X):
        return proba_of_one(self.model, X)
