"""Experiments (PRD 7.9): one cell, or the whole resumable matrix.

    from src import experiment
    row, curve = experiment.run_one(df, split, "tfidf", "lr", grid, seed, profile)
    experiment.run_all(df, config.FULL, config.RESULTS_FULL)   # python main.py --full

A cell is one setting x held-out benchmark x seed x feature set x router:
1. Fit the feature set on the training rows only; transform val and test.
2. For every grid point: fit the router on train and score validation. Keep
   the one with the highest validation curve AUC (D9).
3. Touch test once, with that router: the test curve, AUC, lift over
   random, Cost@95% read off the test curve, and the threshold chosen on
   validation applied to test (D8), plus classifier metrics.

run_all covers both settings: the random split, and leave-one-benchmark-out
for every group big enough to hold out (D3). It fits each feature set once
per split and reuses it for every router. Each finished cell appends one row
to metrics_<setting>.csv, so an interrupted run resumes where it stopped and
Avimta's routers can be added later without rerunning the others.

Metrics are Avimta's (src/metrics.py, src/policies.py).
"""

import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import features, metrics, policies, routers, runinfo, splits

# Columns that identify a cell; a finished cell has exactly one row.
KEY = ["setting", "heldout", "seed", "feature_set", "router"]

# Saved curves keep this many of their points (see thin_curve).
CURVE_POINTS = 201


def grid_points(grid):
    """Every combination of a grid dict, in a fixed order."""
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def outcome_arrays(part):
    """The four per-prompt arrays the metrics need, for one part of a split."""
    return (part["q_cheap"].to_numpy(), part["q_strong"].to_numpy(),
            part["c_cheap"].to_numpy(), part["c_strong"].to_numpy())


def fit_features(df, split, feature_set, profile, seed):
    """Fit the feature set on the training rows only; returns (X_train, X_val, X_test)."""
    train, val, test = (df.iloc[split[k]] for k in ("train", "val", "test"))
    feats = features.make(feature_set, profile, seed)
    X_train = feats.fit_transform(train)
    return X_train, feats.transform(val), feats.transform(test)


def run_router(df, split, X, feature_set, router_name, grid, seed, setting="random",
               heldout=None):
    """Select on validation, then test once, on features already fitted.

    Returns (metrics row, test curve, test scores, threshold chosen on validation).
    """
    t0 = time.perf_counter()
    X_train, X_val, X_test = X
    train, val, test = (df.iloc[split[k]] for k in ("train", "val", "test"))
    y_train = train["y"].to_numpy()
    val_out, test_out = outcome_arrays(val), outcome_arrays(test)

    # Settings fixed by the feature set (e.g. kNN's distance metric), added
    # to every grid point; they're not tuned (config.ROUTER_FEATURE_PARAMS).
    fixed = config.ROUTER_FEATURE_PARAMS.get(router_name, {}).get(feature_set, {})

    # Selection on validation only (D9).
    best = None
    for tuned in grid_points(grid):
        params = {**fixed, **tuned}
        router = routers.make(router_name, seed, **params).fit(X_train, y_train)
        s_val = router.score(X_val)
        auc = metrics.curve_auc(metrics.routing_curve(s_val, *val_out))
        if best is None or auc > best[0]:  # ties keep the earlier grid point
            best = (auc, params, router, s_val)
    val_auc, params, router, s_val = best

    # Test, once.
    s_test = router.score(X_test)
    curve = metrics.routing_curve(s_test, *test_out)
    q_cheap, q_strong, _, c_strong = test_out
    test_auc = metrics.curve_auc(curve)
    rand_auc = metrics.random_auc(q_cheap.mean(), q_strong.mean())
    t_star = metrics.choose_threshold(s_val, *val_out)
    at_t = metrics.apply_threshold(s_test, t_star, *test_out)
    clf = metrics.classifier_metrics(test["y"].to_numpy(), s_test, t_star)
    ref = policies.reference_policies(*test_out)

    row = {
        "setting": setting,
        "heldout": heldout or "",
        "seed": seed,
        "feature_set": feature_set,
        "router": router_name,
        "best_params": json.dumps(params, sort_keys=True),
        "val_auc": val_auc,
        "test_auc": test_auc,
        "random_auc": rand_auc,
        "lift": test_auc - rand_auc,
        "cost95_test_pct": metrics.cost_at_95_test(curve, q_strong.mean(), c_strong.mean()),
        "t_star": t_star,
        "cost_pct_at_t": at_t["cost_pct"],
        "quality_at_t": at_t["quality"],
        "met_target_at_t": at_t["met_target"],
        **clf,
        "q_always_cheap": ref.loc["always_cheap", "quality"],
        "q_always_strong": ref.loc["always_strong", "quality"],
        "q_oracle": ref.loc["oracle", "quality"],
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return row, curve, s_test, t_star


def run_one(df, split, feature_set, router_name, grid, seed, profile, setting="random",
            heldout=None):
    """One cell, features included. Returns (metrics row as a dict, test curve)."""
    t0 = time.perf_counter()
    X = fit_features(df, split, feature_set, profile, seed)
    row, curve, _, _ = run_router(df, split, X, feature_set, router_name, grid, seed,
                                  setting, heldout)
    row["seconds"] = round(time.perf_counter() - t0, 2)  # features + router
    return row, curve


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------
def _feature_spec(feature_set, profile):
    if feature_set == "tfidf":
        return {"tfidf": config.TFIDF_PARAMS, "svd_components": profile["svd_components"]}
    if feature_set == "handbuilt":
        return {"features": config.HANDBUILT_FEATURES, "code_symbols": config.CODE_SYMBOLS,
                "math_symbols": config.MATH_SYMBOLS, "code_keywords": config.CODE_KEYWORDS}
    return {"model": config.EMBEDDING_MODEL, "normalize": config.EMBEDDING_NORMALIZE,
            "max_tokens": config.EMBEDDING_MAX_TOKENS,
            "turn_order": config.PROMPT_TURN_ORDER}


def fingerprint(router_name, feature_set, profile):
    """Short hash of every setting that decides a cell's result.

    Stored with each row. On resume a finished row is skipped only if its
    fingerprint still matches; a row made under other settings stops the
    run instead of being mixed into new results.
    """
    spec = {
        "router": router_name,
        "grid": profile["grids"][router_name],
        "fixed": config.ROUTER_FEATURE_PARAMS.get(router_name, {}).get(feature_set, {}),
        "feature_set": feature_set,
        "features": _feature_spec(feature_set, profile),
        "data": {
            "revision": config.DATASET_REVISION,
            "excluded_eval_names": config.EXCLUDED_EVAL_NAMES,
            "excluded_groups": config.EXCLUDED_GROUPS,
            "group_prefixes": config.GROUP_PREFIXES,
            "turn_order": config.PROMPT_TURN_ORDER,
            "turn_separator": config.PROMPT_TURN_SEPARATOR,
            "dedupe": [config.DEDUPE_WITHIN_GROUP, config.DROP_CONFLICTING_DUPLICATES],
            "heldout_rule": [config.HELDOUT_MIN_ROWS, config.HELDOUT_MIN_CHEAP_RIGHT,
                             config.HELDOUT_MIN_CHEAP_WRONG],
        },
        "pair": [config.CHEAP_MODEL, config.STRONG_MODEL],
        "target": config.TARGET_QUALITY_FRAC,
        "splits": [config.TRAIN_FRAC, config.VAL_FRAC, config.TEST_FRAC,
                   config.LOBO_TRAIN_FRAC, config.LOBO_VAL_FRAC],
    }
    blob = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def thin_curve(curve, points=CURVE_POINTS):
    """Keep `points` rows of a test curve, evenly spaced, first and last included.

    Every kept row is an exact point of the curve, so plots look the same.
    The metrics are computed on the full curve before this; it only keeps
    the saved files small (a held-out benchmark's curve has up to 14,000
    rows, and the matrix has hundreds of curves).
    """
    if len(curve) <= points:
        return curve
    idx = np.unique(np.linspace(0, len(curve) - 1, points).round().astype(int))
    return curve.iloc[idx]


def cell_name(setting, heldout, seed, feature_set, router_name):
    parts = [setting] + ([heldout] if heldout else []) + [feature_set, router_name, f"seed{seed}"]
    return "_".join(parts)


def _read_rows(path):
    if not path.exists():
        return pd.DataFrame()
    # keep_default_na=False keeps the random setting's empty `heldout` as "".
    return pd.read_csv(path, dtype={"heldout": str}, keep_default_na=False)


def _append_row(path, row):
    """Append one row; refuse if the file's columns differ from the row's."""
    if path.exists():
        with path.open() as f:
            header = f.readline().rstrip("\n").split(",")
        if header != list(row):
            raise RuntimeError(f"{path} has different columns from this code's rows; "
                               "write to a new output folder.")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _embeddings_available(profile):
    if profile["name"] == "sample":
        return config.SAMPLE_EMBEDDINGS.exists() and config.SAMPLE_IDS.exists()
    return config.EMB_CACHE.exists() and config.EMB_IDS.exists()


def _blocks(df, settings, seeds):
    """(setting, held-out group, seed) for every split, seed-major, so a
    partial run has every held-out benchmark for the first seeds."""
    blocks = []
    for setting in settings:
        if setting == "random":
            blocks += [("random", "", s) for s in seeds]
        elif setting == "lobo":
            groups = splits.lobo_groups(df)
            blocks += [("lobo", g, s) for s in seeds for g in groups]
        else:
            raise ValueError(f"unknown setting {setting!r}; expected {config.SETTINGS}")
    return blocks


def run_all(df, profile, out_dir, router_names=None, feature_sets=None, settings=None,
            seeds=None, log=print):
    """Run every unfinished cell of the matrix; returns counts.

    Writes to out_dir:
        metrics_random.csv, metrics_lobo.csv   one row per finished cell
        reference_policies.csv                 always-cheap/strong, random, oracle per test split
        curves/<cell>.csv                      test curve, thinned (thin_curve)
        predictions/<cell>.csv.gz              per test prompt: score and whether it went
                                               to strong at the validation threshold
        run_info.json                          one entry per invocation
    """
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    router_names = list(router_names or routers.ROUTERS)
    feature_sets = list(feature_sets or config.FEATURE_SETS)
    settings = list(settings or config.SETTINGS)
    seeds = list(profile["seeds"] if seeds is None else seeds)

    unknown = [r for r in router_names if r not in routers.ROUTERS]
    if unknown:
        raise ValueError(f"unknown routers {unknown}; registered: {sorted(routers.ROUTERS)}")
    # A router without a grid (e.g. Avimta's, until she fills config) is
    # skipped, never run with guessed settings.
    ready = [r for r in router_names if profile["grids"].get(r)]
    skipped_routers = [r for r in router_names if r not in ready]
    for r in skipped_routers:
        log(f"Skipping router {r}: no grid for it in config yet.")
    # Without the embedding cache, run the other feature sets (PRD 5.1).
    if "embedding" in feature_sets and not _embeddings_available(profile):
        log("Skipping feature set embedding: the embedding cache is missing "
            "(run `python -m src.embed` first).")
        feature_sets = [f for f in feature_sets if f != "embedding"]

    for sub in ("curves", "predictions"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    fps = {(r, f): fingerprint(r, f, profile) for r in ready for f in feature_sets}
    done = {}
    for setting in settings:
        for _, r in _read_rows(out_dir / f"metrics_{setting}.csv").iterrows():
            done[(r["setting"], r["heldout"], int(r["seed"]), r["feature_set"], r["router"])] = \
                r["fingerprint"]
    ref_path = out_dir / "reference_policies.csv"
    refs_done = {(r["setting"], r["heldout"], int(r["seed"]))
                 for _, r in _read_rows(ref_path).iterrows()}

    commit = runinfo.git_commit()
    blocks = _blocks(df, settings, seeds)
    counts = {"total": len(blocks) * len(feature_sets) * len(ready), "ran": 0, "skipped": 0}
    for setting, heldout, seed in blocks:
        split = None
        for fs in feature_sets:
            todo = []
            for r in ready:
                key = (setting, heldout, seed, fs, r)
                if key not in done:
                    todo.append(r)
                elif done[key] != fps[(r, fs)]:
                    raise RuntimeError(
                        f"{out_dir / f'metrics_{setting}.csv'} already has {key} made with "
                        "different settings (config changed since). Results are never "
                        "overwritten: write to a new output folder.")
                else:
                    counts["skipped"] += 1
            if not todo:
                continue

            if split is None:
                split = (splits.random_split(df, seed) if setting == "random"
                         else splits.lobo_split(df, heldout, seed))
                if (setting, heldout, seed) not in refs_done:
                    test = df.iloc[split["test"]]
                    ref = policies.reference_policies(*outcome_arrays(test))
                    ref = ref.rename_axis("policy").reset_index()
                    for col, val in reversed(list(zip(["setting", "heldout", "seed"],
                                                      [setting, heldout, seed]))):
                        ref.insert(0, col, val)
                    ref.to_csv(ref_path, mode="a", header=not ref_path.exists(), index=False)
                    refs_done.add((setting, heldout, seed))

            t0 = time.perf_counter()
            X = fit_features(df, split, fs, profile, seed)
            feature_seconds = round(time.perf_counter() - t0, 2)
            test = df.iloc[split["test"]]
            for r in todo:
                row, curve, s_test, t_star = run_router(
                    df, split, X, fs, r, profile["grids"][r], seed, setting, heldout or None)
                row.update(feature_seconds=feature_seconds, fingerprint=fps[(r, fs)],
                           git_commit=commit,
                           timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                name = cell_name(setting, heldout, seed, fs, r)
                thin_curve(curve).to_csv(out_dir / "curves" / f"{name}.csv", index=False,
                                         float_format="%.8g")
                pd.DataFrame({
                    "sample_id": test["sample_id"].to_numpy(),
                    "group": test["group"].to_numpy(),
                    "score": s_test,
                    # apply_threshold: cheap if score >= t*, else strong.
                    "to_strong": s_test < t_star,
                }).to_csv(out_dir / "predictions" / f"{name}.csv.gz", index=False,
                          float_format="%.8g",
                          compression={"method": "gzip", "mtime": 0})  # reproducible bytes
                # The metrics row goes last: it marks the cell as finished.
                _append_row(out_dir / f"metrics_{setting}.csv", row)
                counts["ran"] += 1
                log(f"[{counts['ran'] + counts['skipped']}/{counts['total']}] "
                    f"{setting} {heldout or '-'} seed {seed} {fs} {r}: test AUC "
                    f"{row['test_auc']:.4f}, lift {row['lift']:+.4f} "
                    f"({feature_seconds + row['seconds']:.0f} s)")

    info = {
        "profile": profile["name"],
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.perf_counter() - t_start, 1),
        "peak_memory_mb": runinfo.peak_memory_mb(),
        **counts,
        "routers": ready,
        "routers_skipped_no_grid": skipped_routers,
        "feature_sets": feature_sets,
        "settings": settings,
        "seeds": seeds,
        "lobo_groups": splits.lobo_groups(df) if "lobo" in settings else [],
        "grids": {r: profile["grids"][r] for r in ready},
        "fixed_params": {r: config.ROUTER_FEATURE_PARAMS.get(r, {}) for r in ready},
        "svd_components": profile["svd_components"],
        "n_rows": len(df),
        "cheap_model": config.CHEAP_MODEL,
        "strong_model": config.STRONG_MODEL,
        **runinfo.versions(),
    }
    info_path = out_dir / "run_info.json"
    history = json.loads(info_path.read_text())["invocations"] if info_path.exists() else []
    info_path.write_text(json.dumps({"invocations": history + [info]}, indent=2) + "\n")
    return counts
