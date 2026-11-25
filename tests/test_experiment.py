"""Tests for the resumable experiment runner (experiment.run_all) and main.py --full.

Synthetic prompts in two benchmark groups, small grids, so every test runs
in seconds and needs no data download.
"""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

import config
import main
from src import experiment

FEATURES = ["tfidf", "handbuilt"]
ROUTERS = ["lr", "knn", "kmeans"]


def frame(n=800, seed=0):
    """Two groups, each big enough to hold out (D3), wording predicts y."""
    rng = np.random.default_rng(seed)
    easy = rng.random(n) < 0.5
    text = np.where(easy, "simple short question about facts",
                    "hard multi step proof with integrals and lemmas")
    y = np.where(rng.random(n) < 0.9, easy, ~easy).astype(int)
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "group": np.where(np.arange(n) % 2 == 0, "g1", "g2"),
        "text": text,
        "y": y,
        "q_cheap": y.astype(float),
        "q_strong": np.ones(n),
        "c_cheap": np.full(n, 0.1),
        "c_strong": np.full(n, 1.0),
    })


def profile(**grids):
    g = {"lr": {"C": [1], "max_iter": [2000]},
         "knn": {"n_neighbors": [5], "weights": ["uniform"]},
         "kmeans": {"n_clusters": [2], "n_init": [2]}}   # the frame has 2 distinct texts
    g.update(grids)
    return dict(config.SAMPLE, svd_components=2, grids=g)


def run(out, prof=None, **kw):
    kw.setdefault("feature_sets", FEATURES)
    kw.setdefault("seeds", [0, 1])
    return experiment.run_all(frame(), prof or profile(), out, log=lambda *a: None, **kw)


def test_runs_every_cell_of_both_settings(tmp_path):
    counts = run(tmp_path)
    # random: 2 seeds x 2 feature sets x 3 routers; lobo: x 2 held-out groups
    assert counts == {"total": 36, "ran": 36, "skipped": 0}
    rnd = pd.read_csv(tmp_path / "metrics_random.csv", keep_default_na=False)
    lobo = pd.read_csv(tmp_path / "metrics_lobo.csv", keep_default_na=False)
    assert len(rnd) == 12 and len(lobo) == 24
    assert not rnd.duplicated(experiment.KEY).any() and not lobo.duplicated(experiment.KEY).any()
    assert set(lobo["heldout"]) == {"g1", "g2"} and set(rnd["heldout"]) == {""}
    assert len(list((tmp_path / "curves").glob("*.csv"))) == 36
    assert len(list((tmp_path / "predictions").glob("*.csv.gz"))) == 36
    refs = pd.read_csv(tmp_path / "reference_policies.csv", keep_default_na=False)
    assert len(refs) == (2 + 4) * 4   # 6 test splits x 4 policies
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["invocations"][-1]["ran"] == 36


def test_heldout_benchmark_is_the_whole_test_set_and_nothing_else(tmp_path):
    run(tmp_path, settings=["lobo"], seeds=[0])
    lobo = pd.read_csv(tmp_path / "metrics_lobo.csv", keep_default_na=False)
    for _, r in lobo.iterrows():
        name = experiment.cell_name("lobo", r["heldout"], 0, r["feature_set"], r["router"])
        with gzip.open(tmp_path / "predictions" / f"{name}.csv.gz", "rt") as f:
            pred = pd.read_csv(f)
        assert set(pred["group"]) == {r["heldout"]}
        assert len(pred) == r["n_test"] == 400
        # Train and validation come from the other group only.
        assert r["n_train"] + r["n_val"] == 400


def test_rerun_skips_finished_cells_and_changes_nothing(tmp_path):
    run(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
              and p.name != "run_info.json"}
    counts = run(tmp_path)
    assert counts == {"total": 36, "ran": 0, "skipped": 36}
    after = {p: p.read_bytes() for p in before}
    assert before == after
    assert len(json.loads((tmp_path / "run_info.json").read_text())["invocations"]) == 2


def test_interrupted_or_partial_run_resumes_with_the_rest(tmp_path):
    first = run(tmp_path, router_names=["lr"])
    assert first["ran"] == 12
    second = run(tmp_path)
    assert second == {"total": 36, "ran": 24, "skipped": 12}
    lobo = pd.read_csv(tmp_path / "metrics_lobo.csv", keep_default_na=False)
    assert len(lobo) == 24 and not lobo.duplicated(experiment.KEY).any()


def test_rows_made_under_other_settings_stop_the_run(tmp_path):
    run(tmp_path, router_names=["lr"], settings=["random"], seeds=[0])
    changed = profile(lr={"C": [0.1], "max_iter": [2000]})
    with pytest.raises(RuntimeError, match="different settings"):
        run(tmp_path, prof=changed, router_names=["lr"], settings=["random"], seeds=[0])


def test_router_without_a_grid_is_skipped_not_guessed(tmp_path):
    counts = run(tmp_path, prof=profile(kmeans=None), settings=["random"], seeds=[0])
    rnd = pd.read_csv(tmp_path / "metrics_random.csv")
    assert set(rnd["router"]) == {"lr", "knn"} and counts["total"] == 4


def test_missing_embedding_cache_runs_the_other_feature_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SAMPLE_EMBEDDINGS", tmp_path / "nope.npy")
    counts = run(tmp_path, feature_sets=["tfidf", "embedding"], router_names=["lr"],
                 settings=["random"], seeds=[0])
    rnd = pd.read_csv(tmp_path / "metrics_random.csv")
    assert set(rnd["feature_set"]) == {"tfidf"} and counts["total"] == 1


def test_unknown_router_or_setting_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, router_names=["crystal_ball"])
    with pytest.raises(ValueError):
        run(tmp_path, settings=["sideways"])


def test_thin_curve_keeps_exact_points_and_both_ends():
    curve = pd.DataFrame({"x": np.linspace(0, 1, 5000), "quality": np.arange(5000.0)})
    thin = experiment.thin_curve(curve, points=201)
    assert len(thin) == 201
    assert thin.iloc[0].equals(curve.iloc[0]) and thin.iloc[-1].equals(curve.iloc[-1])
    assert thin["quality"].isin(curve["quality"]).all()
    short = curve.iloc[:50]
    assert experiment.thin_curve(short, points=201).equals(short)


def test_fingerprint_changes_with_the_grid_only_for_that_router():
    a, b = profile(), profile(knn={"n_neighbors": [7], "weights": ["uniform"]})
    assert experiment.fingerprint("knn", "tfidf", a) != experiment.fingerprint("knn", "tfidf", b)
    assert experiment.fingerprint("lr", "tfidf", a) == experiment.fingerprint("lr", "tfidf", b)


def test_main_full_without_the_data_prints_the_download_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RAW_PICKLE", config.ROOT / "data" / "raw" / "missing.pkl")
    assert main.main(["--full"]) == 1
    assert "python data/download.py" in capsys.readouterr().err


def test_main_parses_the_development_flags():
    args = main.parse_args(["--full", "--routers", "lr,knn", "--settings", "lobo",
                            "--seeds", "0,2"])
    assert (args.routers, args.settings, args.seeds) == (["lr", "knn"], ["lobo"], [0, 2])
