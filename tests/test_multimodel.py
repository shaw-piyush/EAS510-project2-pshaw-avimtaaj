"""Tests for src/multimodel.py on synthetic data (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

import config
from src import multimodel, splits

COST = "|total_cost"


def test_frontier_drops_dominated_models():
    means = pd.DataFrame({"mean_score": [0.3, 0.5, 0.4, 0.9],
                          "mean_cost": [1.0, 2.0, 3.0, 10.0]},
                         index=["cheap", "mid", "worse_mid", "strong"])
    on = multimodel.frontier(means)
    # worse_mid costs more than mid and scores lower: dominated.
    assert on.to_dict() == {"cheap": True, "mid": True, "worse_mid": False, "strong": True}


def test_route_lambda_zero_takes_most_likely_and_large_lambda_cheapest():
    P = np.array([[0.2, 0.5, 0.9],
                  [0.6, 0.6, 0.6]])          # row 2: exact tie
    rel_cost = np.array([0.01, 0.1, 1.0])     # ordered cheapest first
    assert multimodel.route(P, rel_cost, 0.0).tolist() == [2, 0]   # tie -> cheaper
    assert multimodel.route(P, rel_cost, 1e6).tolist() == [0, 0]


def test_oracle_picks_cheapest_correct_model():
    scores = np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=float)
    costs = np.array([[1, 2, 3], [5, 2, 3], [4, 1, 9]], dtype=float)
    assert multimodel.oracle_choice(scores, costs).tolist() == [1, 1, 1]


def synthetic_frame(n=900, seed=0):
    """Four models; whether the cheap ones are right depends on the wording."""
    rng = np.random.default_rng(seed)
    easy = rng.random(n) < 0.5
    df = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "group": np.where(rng.random(n) < 0.5, "g1", "g2"),
        "text": np.where(easy, "simple short fact question", "hard multi step integral proof"),
    })
    p_right = {"tiny": np.where(easy, 0.7, 0.1), "small": np.where(easy, 0.85, 0.3),
               "bad": np.full(n, 0.2), "big": np.full(n, 0.9)}
    cost = {"tiny": 0.01, "small": 0.05, "bad": 0.5, "big": 1.0}  # "bad" is dominated
    for m in p_right:
        df[m] = (rng.random(n) < p_right[m]).astype(float)
        df[m + COST] = cost[m] * rng.uniform(0.9, 1.1, n)
    return df


@pytest.fixture(scope="module")
def result():
    df = synthetic_frame()
    split = splits.random_split(df, seed=0)
    profile = dict(config.SAMPLE, svd_components=2)
    return multimodel.run_one(df, split, "tfidf", "lr", {"C": [0.1, 1], "max_iter": [2000]},
                              0, profile), df, split


def test_run_one_candidates_and_outputs(result):
    res, _, _ = result
    cand = res["candidates"].set_index("model")["on_frontier"]
    assert not cand["bad"] and cand[["tiny", "small", "big"]].all()
    assert res["strongest"] == "big"
    assert list(res["brier"]["model"]) == ["tiny", "small", "big"]  # cheapest first
    assert len(res["curve"]) == len(config.MULTIMODEL_LAMBDAS)
    # The prompts' wording predicts the cheap models, so their classifiers
    # beat a constant guess.
    b = res["brier"].set_index("model")
    assert (b.loc[["tiny", "small"], "test_brier"] < b.loc[["tiny", "small"], "test_brier_constant"]).all()


def test_curve_spans_strong_to_cheap_and_oracle_is_best(result):
    res, _, _ = result
    curve, pol = res["curve"], res["policies"].set_index("policy")
    # Largest lambda: every prompt goes to the cheapest candidate.
    assert curve["cost"].iloc[-1] == pytest.approx(pol.loc["always_cheapest", "cost"])
    assert curve["share:tiny"].iloc[-1] == 1.0
    # Cost never rises as lambda grows (cheaper models only get more attractive).
    assert (np.diff(curve["cost"]) <= 1e-12).all()
    assert pol.loc["oracle", "quality"] >= curve["quality"].max() - 1e-12


def test_candidates_use_training_rows_only(result):
    res, df, split = result
    # Make "bad" look perfect on validation and test: the candidate list
    # must not change, because it is built from training means only.
    df2 = df.copy()
    rest = np.r_[split["val"], split["test"]]
    df2.loc[df2.index[rest], "bad"] = 1.0
    profile = dict(config.SAMPLE, svd_components=2)
    res2 = multimodel.run_one(df2, split, "tfidf", "lr", {"C": [1], "max_iter": [2000]},
                              0, profile)
    assert list(res2["brier"]["model"]) == list(res["brier"]["model"])
