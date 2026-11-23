"""Tests for the router interface, the LR baseline and experiment.run_one."""

import numpy as np
import pandas as pd
import pytest

import config
from src import experiment, routers, splits
from src.routers.base import Router


def test_registry_has_lr_and_rejects_unknown():
    assert isinstance(routers.make("lr", seed=0, C=1.0), Router)
    with pytest.raises(ValueError):
        routers.make("crystal_ball", seed=0)


def test_lr_scores_are_probabilities_of_cheap_correct():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0).astype(int)          # cheap is right when feature 0 > 0
    r = routers.make("lr", seed=0, C=1.0).fit(X, y)
    s = r.score(X)
    assert s.shape == (400,) and ((0 <= s) & (s <= 1)).all()
    assert s[y == 1].mean() > s[y == 0].mean()


def test_one_class_training_gives_a_constant_score():
    # Only possible on tiny splits, but it must not crash or flip classes.
    from sklearn.dummy import DummyClassifier
    from src.routers.base import proba_of_one
    m = DummyClassifier().fit(np.zeros((3, 1)), [0, 0, 0])
    assert (proba_of_one(m, np.zeros((2, 1))) == 0).all()


def test_grid_points_cover_every_combination_in_order():
    pts = experiment.grid_points({"C": [0.1, 1], "max_iter": [100]})
    assert pts == [{"C": 0.1, "max_iter": 100}, {"C": 1, "max_iter": 100}]


def synthetic_frame(n=600, seed=0):
    """Prompts whose wording predicts whether the cheap model is right."""
    rng = np.random.default_rng(seed)
    easy = rng.random(n) < 0.5
    text = np.where(easy, "simple short question about facts",
                    "hard multi step proof with integrals and lemmas")
    y = np.where(rng.random(n) < 0.9, easy, ~easy).astype(int)
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "group": np.where(rng.random(n) < 0.5, "g1", "g2"),
        "text": text,
        "y": y,
        "q_cheap": y.astype(float),
        "q_strong": np.ones(n),
        "c_cheap": np.full(n, 0.1),
        "c_strong": np.full(n, 1.0),
    })


@pytest.mark.parametrize("feature_set", ["tfidf", "handbuilt"])
def test_run_one_selects_on_val_and_beats_random(feature_set):
    df = synthetic_frame()
    split = splits.random_split(df, seed=0)
    profile = dict(config.SAMPLE, svd_components=2)
    row, curve = experiment.run_one(df, split, feature_set, "lr",
                                    {"C": [0.1, 1], "max_iter": [2000]}, 0, profile)
    assert row["n_test"] == len(split["test"]) and len(curve) >= 2
    assert row["best_params"] in ('{"C": 0.1, "max_iter": 2000}', '{"C": 1, "max_iter": 2000}')
    # Wording predicts correctness, so the router should beat random routing
    # and the curve should start at always-cheap and end at always-strong.
    assert row["lift"] > 0
    assert curve["x"].iloc[0] == pytest.approx(0) and curve["x"].iloc[-1] == pytest.approx(1)
    assert 0 <= row["cost_pct_at_t"] <= 100
