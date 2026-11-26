"""Multi-layer perceptron router: the project's neural model (PRD 8.5).

Score = predicted P(cheap model is correct), from sklearn's MLPClassifier
(a small fully connected network trained with Adam on log-loss).

Design choices:
- StandardScaler inside the pipeline, fit on the training rows only. Unlike
  trees, a network's training is sensitive to feature scale: the hand-built
  counts run into the thousands next to 0/1 flags, and unscaled inputs make
  gradient descent slow and unstable. On the unit-length TF-IDF and
  embedding vectors scaling is harmless.
- Small networks only: (128,) or (256, 128) hidden units. With ~20,000
  training prompts and a weak signal (the LR baseline barely beats random),
  a bigger network would mostly memorize noise.
- alpha is the L2 penalty on the weights, the main guard against
  overfitting; it's tuned on validation curve AUC by Piyush's runner.
- early_stopping=True holds out 10% of the TRAINING rows (sklearn default)
  and stops when the score on them stops improving for 10 epochs, so
  max_iter=200 is only an upper limit. That holdout never sees our
  validation or test split.
- random_state=seed fixes the starting weights, the batch order and the
  early-stopping holdout, so a seed always gives the same network.
- Frozen features only: the MLP learns from TF-IDF, hand-built or MiniLM
  vectors; it never fine-tunes the embedding model (scope rule).
"""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.routers.base import Router, proba_of_one


class MLPRouter(Router):
    name = "mlp"

    def __init__(self, seed, hidden_layer_sizes=(128,), alpha=1e-4, early_stopping=True,
                 max_iter=200):
        # Grids come from config (and best_params are stored as JSON, which
        # turns tuples into lists), so accept a list and make it a tuple.
        hidden_layer_sizes = tuple(hidden_layer_sizes)
        super().__init__(seed, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                         early_stopping=early_stopping, max_iter=max_iter)
        self.model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                early_stopping=early_stopping,
                max_iter=max_iter,
                random_state=seed,
            ),
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def score(self, X):
        return proba_of_one(self.model, X)
