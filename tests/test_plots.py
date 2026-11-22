"""Smoke tests for src/plots.py: each figure renders to a PNG from synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on the import path (Piyush's pytest.ini also does this once merged).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import routing_curve  # noqa: E402
from src.plots import (  # noqa: E402
    color_of,
    plot_multimodel,
    plot_per_benchmark,
    plot_quality_cost,
)
from src.policies import reference_policies  # noqa: E402


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    qc = (rng.random(n) < 0.5).astype(float)
    qs = (rng.random(n) < 0.85).astype(float)
    cc = rng.uniform(0.1, 0.5, n)
    cs = cc + rng.uniform(1.0, 3.0, n)
    return qc, qs, cc, cs


def is_png(path):
    return Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_quality_cost_figure_renders(tmp_path):
    qc, qs, cc, cs = make_data()
    rng = np.random.default_rng(1)
    curves = {
        "lr": routing_curve(qc + rng.normal(0, 0.6, len(qc)), qc, qs, cc, cs),
        "knn": routing_curve(np.round(qc + rng.normal(0, 0.8, len(qc)), 1), qc, qs, cc, cs),
    }
    out = plot_quality_cost(curves, reference_policies(qc, qs, cc, cs),
                            tmp_path / "fig" / "qc.png", "synthetic | random | tfidf")
    assert is_png(out)


def test_per_benchmark_figure_renders(tmp_path):
    table = pd.DataFrame({
        "group": ["mmlu", "mmlu", "mbpp", "mbpp"],
        "router": ["lr", "hgb", "lr", "hgb"],
        "auc_lift": [0.03, 0.05, -0.01, 0.02],
        "auc_lift_std": [0.01, 0.01, 0.02, 0.02],
    })
    assert is_png(plot_per_benchmark(table, "auc_lift", tmp_path / "pb.png", "synthetic"))


def test_multimodel_figure_renders(tmp_path):
    curve = pd.DataFrame({"cost": [1.0, 2.0, 4.0, 8.0], "quality": [0.5, 0.65, 0.75, 0.8]})
    pol = pd.DataFrame({"cost": [0.5, 8.0, 3.0], "quality": [0.45, 0.8, 0.9]},
                       index=["always_cheapest", "always_strongest", "oracle"])
    assert is_png(plot_multimodel(curve, pol, tmp_path / "mm.png", "synthetic"))


def test_colors_are_fixed_per_router():
    assert color_of("lr") == color_of("LR") == color_of("lr_tfidf")
    assert color_of("hgb") != color_of("lr")
    assert color_of("something_new").startswith("#")
