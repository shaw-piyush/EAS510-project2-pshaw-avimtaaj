"""Smoke tests for Avimta's routers (PRD 8.5): each fits a small synthetic
array, returns one probability per row, learns a real signal, and is
reproducible for a seed. Kept apart from tests/test_routers.py (Piyush's)."""

import numpy as np
import pytest

import config
from src import routers
from src.routers.base import Router

# Small settings so the tests run in a second or two.
FAST = {
    "hgb": {"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 50},
    "rf": {"n_estimators": 30, "min_samples_leaf": 5},
}


def signal_data(n=600, seed=0):
    """Cheap is right (y=1) mostly when feature 0 > 0; the others are noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.where(rng.random(n) < 0.9, X[:, 0] > 0, X[:, 0] <= 0).astype(int)
    return X, y


@pytest.mark.parametrize("name", ["hgb", "rf"])
def test_registered_and_follows_the_interface(name):
    assert isinstance(routers.make(name, seed=0, **FAST[name]), Router)


@pytest.mark.parametrize("name", ["hgb", "rf"])
def test_scores_are_probabilities_that_track_the_signal(name):
    X, y = signal_data()
    X_test, y_test = signal_data(n=300, seed=1)
    s = routers.make(name, seed=0, **FAST[name]).fit(X, y).score(X_test)
    assert s.shape == (len(X_test),)
    assert np.isfinite(s).all() and ((0 <= s) & (s <= 1)).all()
    assert s[y_test == 1].mean() > s[y_test == 0].mean() + 0.3


@pytest.mark.parametrize("name", ["hgb", "rf"])
def test_same_seed_gives_same_scores(name):
    X, y = signal_data()
    s1 = routers.make(name, seed=3, **FAST[name]).fit(X, y).score(X)
    s2 = routers.make(name, seed=3, **FAST[name]).fit(X, y).score(X)
    assert np.array_equal(s1, s2)


@pytest.mark.parametrize("name", ["hgb", "rf"])
def test_every_grid_point_builds_a_router(name):
    # A typo in a config grid key would only show up deep inside a full run.
    from src.experiment import grid_points
    for profile in (config.GRIDS_FULL, config.GRIDS_SAMPLE):
        for params in grid_points(profile[name]):
            assert isinstance(routers.make(name, seed=0, **params), Router)
