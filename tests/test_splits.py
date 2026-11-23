"""Tests for src/splits.py on synthetic frames shaped like src/data.py output."""

import numpy as np
import pandas as pd
import pytest

import config
from src import splits


def frame(sizes, seed=0):
    """sizes: {group: n_rows}. Labels alternate so every group has both."""
    rows = []
    for g, n in sizes.items():
        rows += [{"sample_id": f"{g}.{i}", "group": g, "y": i % 2} for i in range(n)]
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


SIZES = {"big": 1000, "mid": 300, "small": 40}


def test_random_split_fractions_and_stratification():
    df = frame(SIZES)
    s = splits.random_split(df, seed=0)
    n = len(df)
    assert abs(len(s["train"]) / n - config.TRAIN_FRAC) < 0.01
    assert abs(len(s["val"]) / n - config.VAL_FRAC) < 0.01
    assert abs(len(s["test"]) / n - config.TEST_FRAC) < 0.01
    # Stratified: every group keeps its share in every part.
    for part in s.values():
        share = df.iloc[part]["group"].value_counts(normalize=True)
        overall = df["group"].value_counts(normalize=True)
        assert (abs(share - overall) < 0.02).all()


def test_random_split_is_disjoint_and_complete():
    df = frame(SIZES)
    s = splits.random_split(df, seed=3)
    ids = [set(df.iloc[p]["sample_id"]) for p in s.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(p) for p in s.values()) == len(df)


def test_same_seed_same_split_new_seed_new_split():
    df = frame(SIZES)
    a, b, c = (splits.random_split(df, seed) for seed in (1, 1, 2))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])


def test_lobo_holds_one_group_out_completely():
    df = frame(SIZES)
    s = splits.lobo_split(df, "mid", seed=0)
    groups = df["group"].to_numpy()
    assert set(groups[s["test"]]) == {"mid"} and len(s["test"]) == SIZES["mid"]
    assert "mid" not in set(groups[s["train"]]) | set(groups[s["val"]])
    rest = len(df) - SIZES["mid"]
    assert abs(len(s["val"]) / rest - config.LOBO_VAL_FRAC) < 0.01


def test_lobo_unknown_group_raises():
    with pytest.raises(ValueError):
        splits.lobo_split(frame(SIZES), "nope", seed=0)


def test_tiny_groups_do_not_break_stratification():
    # A 3-row group leaves at most one row in val+test; it must not crash.
    df = frame({"big": 500, "mid": 200, "tiny": 3, "tinier": 2})
    for seed in range(5):
        splits.random_split(df, seed)


def test_check_split_catches_overlap_and_leaks():
    df = frame(SIZES)
    s = splits.random_split(df, seed=0)
    bad = dict(s, val=np.concatenate([s["val"], s["test"][:1]]))
    with pytest.raises(AssertionError, match="both"):
        splits.check_split(df, bad)
    lobo = splits.lobo_split(df, "mid", seed=0)
    leak = {"train": np.concatenate([lobo["train"], lobo["test"][:1]]),
            "val": lobo["val"], "test": lobo["test"][1:]}
    with pytest.raises(AssertionError, match="appears in train"):
        splits.check_split(df, leak, heldout="mid")


def test_lobo_groups_follow_the_d3_rule():
    df = frame({"big": 400, "small": config.HELDOUT_MIN_ROWS - 1})
    assert splits.lobo_groups(df) == ["big"]
