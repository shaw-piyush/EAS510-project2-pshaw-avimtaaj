"""Tests for data/make_sample.py. The allocation and drawing tests use
synthetic data; the last test checks the real data/sample/ files if they
have been built, and is skipped otherwise (e.g. on a fresh clone)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_sample", ROOT / "data" / "make_sample.py")
make_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_sample)

SIZES = {"big": 5000, "mid": 900, "small": 120, "tiny": 30}


def frame(sizes=SIZES):
    groups = [g for g, n in sizes.items() for _ in range(n)]
    return pd.DataFrame({"sample_id": [f"{g}.{i}" for i, g in enumerate(groups)],
                         "group": groups})


def test_allocation_totals_and_minimum():
    take = make_sample.allocate(SIZES, 600, 40)
    assert take.sum() == 600
    assert take["tiny"] == 30                    # smaller than the minimum: all of it
    assert (take[["mid", "small"]] >= 40).all()
    assert (take <= pd.Series(SIZES)).all()
    assert take["big"] > take["mid"] > take["small"]  # the rest is proportional


def test_allocation_rejects_impossible_requests():
    with pytest.raises(ValueError):
        make_sample.allocate(SIZES, 10_000, 40)   # more rows than exist
    with pytest.raises(ValueError):
        make_sample.allocate(SIZES, 100, 40)      # minimums alone exceed the total


def test_draw_is_reproducible_and_ignores_input_order():
    df = frame()
    a = make_sample.draw_sample(df, 600, 40, seed=0)
    b = make_sample.draw_sample(df.sample(frac=1, random_state=7), 600, 40, seed=0)
    assert a["sample_id"].tolist() == b["sample_id"].tolist()
    assert a["sample_id"].is_unique and len(a) == 600
    c = make_sample.draw_sample(df, 600, 40, seed=1)
    assert a["sample_id"].tolist() != c["sample_id"].tolist()


def test_built_sample_loads_in_the_pipeline():
    if not config.SAMPLE_CSV.exists():
        pytest.skip("data/sample/ not built; run `python data/make_sample.py`")
    from src.data import load_sample
    from src.features import EmbeddingFeatures

    df = load_sample()
    assert "eval_name" not in pd.read_csv(config.SAMPLE_CSV, nrows=1).columns
    ids = np.load(config.SAMPLE_IDS)               # must load without allow_pickle
    assert ids.dtype.kind == "U" and ids.tolist() == df["sample_id"].tolist()
    X = EmbeddingFeatures(config.SAMPLE_EMBEDDINGS, config.SAMPLE_IDS).transform(df)
    assert X.shape == (len(df), 384) and X.dtype == np.float32
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4)
    for path in (config.SAMPLE_CSV, config.SAMPLE_EMBEDDINGS, config.SAMPLE_IDS):
        assert path.stat().st_size < 5e6
