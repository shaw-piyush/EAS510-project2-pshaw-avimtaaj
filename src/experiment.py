"""Running one experiment cell (PRD 7.9): seed x split x feature set x router.

    from src import experiment
    row, curve = experiment.run_one(df, split, "tfidf", "lr", grid, seed, profile)

One call does, in this order:
1. Fit the feature set on the training rows only; transform val and test.
2. For every point of the grid: fit the router on train and score
   validation. Keep the one with the highest validation curve AUC (D9).
3. Touch test once, with that router: the test curve, AUC, lift over
   random, Cost@95% read off the test curve, and the threshold chosen on
   validation applied to test (D8), plus classifier metrics.

Metrics are Avimta's (src/metrics.py, src/policies.py). The full runner
over every cell, with resume, comes later (PRD 9, Fri Nov 27).
"""

import itertools
import json
import time

from src import features, metrics, policies, routers


def grid_points(grid):
    """Every combination of a grid dict, in a fixed order."""
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def outcome_arrays(part):
    """The four per-prompt arrays the metrics need, for one part of a split."""
    return (part["q_cheap"].to_numpy(), part["q_strong"].to_numpy(),
            part["c_cheap"].to_numpy(), part["c_strong"].to_numpy())


def run_one(df, split, feature_set, router_name, grid, seed, profile, setting="random",
            heldout=None):
    """Returns (one metrics row as a dict, the test curve as a DataFrame)."""
    t0 = time.perf_counter()
    train, val, test = (df.iloc[split[k]] for k in ("train", "val", "test"))
    y_train = train["y"].to_numpy()
    val_out, test_out = outcome_arrays(val), outcome_arrays(test)

    feats = features.make(feature_set, profile, seed)
    X_train = feats.fit_transform(train)
    X_val, X_test = feats.transform(val), feats.transform(test)

    # Selection on validation only (D9).
    best = None
    for params in grid_points(grid):
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
    return row, curve
