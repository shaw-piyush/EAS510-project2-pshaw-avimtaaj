"""Tests for the router interface, the LR, kNN and k-means routers, and
experiment.run_one."""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src import experiment, routers, splits
from src.routers.base import Router


def test_registry_has_piyushs_routers_and_rejects_unknown():
    assert isinstance(routers.make("lr", seed=0, C=1.0), Router)
    assert isinstance(routers.make("knn", seed=0, n_neighbors=5), Router)
    assert isinstance(routers.make("kmeans", seed=0, n_clusters=2), Router)
    with pytest.raises(ValueError):
        routers.make("crystal_ball", seed=0)


def test_fixed_settings_exist_for_every_feature_set():
    # run_one adds these to every grid point, so a missing entry would
    # silently fall back to the router's defaults.
    for name in ("knn", "kmeans"):
        assert set(config.ROUTER_FEATURE_PARAMS[name]) == set(config.FEATURE_SETS)


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


# --- kNN ---------------------------------------------------------------------------
def two_blobs(n=200, seed=0):
    """Two well-separated groups: cheap right on 80% of A, 20% of B."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[5, 0], scale=0.3, size=(n, 2))
    b = rng.normal(loc=[0, 5], scale=0.3, size=(n, 2))
    y = np.r_[(np.arange(n) % 5 != 0), (np.arange(n) % 5 == 0)].astype(int)  # 80% / 20%
    return np.vstack([a, b]), y


@pytest.mark.parametrize("metric,scale", [("cosine", False), ("euclidean", True)])
def test_knn_scores_are_neighbour_shares(metric, scale):
    X, y = two_blobs()
    r = routers.make("knn", seed=0, n_neighbors=50, metric=metric, scale=scale).fit(X, y)
    s = r.score(X)
    assert s.shape == (len(X),) and ((0 <= s) & (s <= 1)).all()
    # Uniform weights: a score is (cheap right among the 50 neighbours) / 50,
    # and the neighbours all come from the prompt's own blob.
    assert np.allclose(s * 50, np.round(s * 50))
    assert s[:200].mean() == pytest.approx(0.8, abs=0.05)
    assert s[200:].mean() == pytest.approx(0.2, abs=0.05)


def test_knn_cosine_ignores_vector_length_and_uses_brute_force():
    X, y = two_blobs()
    r = routers.make("knn", seed=0, n_neighbors=15, metric="cosine").fit(X, y)
    assert r.model[-1].algorithm == "brute"
    assert np.array_equal(r.score(X), r.score(3.0 * X))


def test_knn_scaling_is_fit_on_training_rows_only():
    X, y = two_blobs()
    r = routers.make("knn", seed=0, n_neighbors=5, metric="euclidean", scale=True).fit(X, y)
    assert np.allclose(r.model[0].mean_, X.mean(axis=0))


# --- k-means -----------------------------------------------------------------------
def test_kmeans_score_is_the_clusters_training_share():
    X, y = two_blobs()
    r = routers.make("kmeans", seed=0, n_clusters=2, n_init=10).fit(X, y)
    assert sorted(np.round(r.cluster_score_, 6)) == [0.2, 0.8]
    s = r.score(X)
    assert np.allclose(s[:200], 0.8) and np.allclose(s[200:], 0.2)


def test_kmeans_is_reproducible_for_a_seed():
    X, y = two_blobs()
    s1 = routers.make("kmeans", seed=3, n_clusters=5, n_init=2).fit(X, y).score(X)
    s2 = routers.make("kmeans", seed=3, n_clusters=5, n_init=2).fit(X, y).score(X)
    assert np.array_equal(s1, s2)


def test_kmeans_empty_cluster_gets_the_overall_share():
    # Two distinct points but three clusters: one cluster ends up empty.
    X = np.array([[0.0], [0.0], [0.0], [10.0], [10.0]])
    y = np.array([1, 1, 0, 0, 0])
    from sklearn.exceptions import ConvergenceWarning
    with pytest.warns(ConvergenceWarning):
        r = routers.make("kmeans", seed=0, n_clusters=3, n_init=1).fit(X, y)
    empty = r.cluster_size_ == 0
    assert empty.any()
    assert np.allclose(r.cluster_score_[empty], y.mean())
    assert np.isfinite(r.score(X)).all()


@pytest.mark.parametrize("name,params", [
    ("kmeans", {"n_clusters": 8, "n_init": 3}),
    ("knn", {"n_neighbors": 10}),
])
def test_curve_does_not_depend_on_the_order_of_tied_rows(name, params):
    # PRD (Wed Nov 25): k-means (and uniform kNN) scores repeat, so the
    # routing curve must give the same result whatever the row order.
    from src import metrics
    rng = np.random.default_rng(1)
    X_train, y_train = two_blobs(seed=1)
    X_test, _ = two_blobs(n=150, seed=2)
    n = len(X_test)
    q_cheap = rng.integers(0, 2, n).astype(float)
    q_strong = rng.integers(0, 2, n).astype(float)
    c_cheap, c_strong = rng.uniform(0.1, 0.2, n), rng.uniform(1, 2, n)
    s = routers.make(name, seed=0, **params).fit(X_train, y_train).score(X_test)
    assert len(np.unique(s)) < n  # there really are ties

    def summary(order):
        curve = metrics.routing_curve(s[order], q_cheap[order], q_strong[order],
                                      c_cheap[order], c_strong[order])
        return (metrics.curve_auc(curve),
                metrics.cost_at_95_test(curve, q_strong.mean(), c_strong.mean()))

    base = summary(np.arange(n))
    for seed in range(5):
        auc, cost95 = summary(np.random.default_rng(seed).permutation(n))
        assert auc == pytest.approx(base[0], abs=1e-12)
        assert cost95 == pytest.approx(base[1], abs=1e-9)


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


@pytest.mark.parametrize("feature_set", ["tfidf", "handbuilt"])
@pytest.mark.parametrize("router_name,grid", [
    ("knn", {"n_neighbors": [5, 15], "weights": ["uniform", "distance"]}),
    ("kmeans", {"n_clusters": [2], "n_init": [10]}),   # the frame has 2 distinct texts
])
def test_run_one_with_knn_and_kmeans_uses_the_fixed_settings(feature_set, router_name, grid):
    df = synthetic_frame()
    split = splits.random_split(df, seed=0)
    profile = dict(config.SAMPLE, svd_components=2)
    row, _ = experiment.run_one(df, split, feature_set, router_name, grid, 0, profile)
    chosen = json.loads(row["best_params"])
    # The feature set's fixed settings are recorded with the tuned ones.
    for key, value in config.ROUTER_FEATURE_PARAMS[router_name][feature_set].items():
        assert chosen[key] == value
    assert row["lift"] > 0
