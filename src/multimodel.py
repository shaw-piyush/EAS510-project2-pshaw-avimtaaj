"""Multi-model routing: choose among several LLMs, not just two (PRD 8.6).

    python -m src.multimodel                          # sample, LR, TF-IDF
    python -m src.multimodel --routers lr,hgb --features embedding
    python -m src.multimodel --full --routers lr,hgb  # full data, every seed

Method, one seed at a time:
1. Candidates: the models on the TRAINING cost-quality frontier, i.e. no
   other model has both a lower mean cost and a higher mean score on the
   training rows. A dominated model is never worth routing to on average.
2. One classifier per candidate predicts P(that model is correct) from the
   prompt's features, trained on the same train split and features as the
   two-model setting. Its hyperparameters are chosen on VALIDATION by Brier
   score (lower is better): the routing rule below compares probabilities
   across models, so their accuracy as probabilities is what matters, and
   with several models there is no single two-model curve to take an AUC of.
3. Routing rule: send each prompt to the candidate maximizing
       P_m(x) - lambda * cbar_m / cbar_strong
   where cbar_m is model m's mean TRAINING cost and cbar_strong that of the
   strongest candidate (highest mean training score). lambda = 0 ignores
   cost; larger lambda pushes prompts to cheaper models. Sweeping lambda
   traces the test quality-vs-cost curve. No lambda is selected, so the
   test split is only used to draw the curve and score the classifiers.
4. Reference policies on test: always-cheapest and always-strongest
   candidate (picked by TRAINING means), and the multi-model oracle (each
   prompt goes to the cheapest candidate that answers it correctly).
5. Calibration check: each classifier's Brier score on test. No calibration
   step is added (PRD 8.6: only if Avimta decides to).

Nothing here uses eval_name, group or sample_id as a feature, and every
number that picks a model, a hyperparameter or a cost scale comes from
train or validation rows.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import config
from src import data, features, routers, splits
from src.data import COST_SUFFIX, model_names
from src.experiment import grid_points


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------
def model_means(df, models):
    """Mean score and mean cost of each model over df's rows."""
    return pd.DataFrame({
        "mean_score": [df[m].mean() for m in models],
        "mean_cost": [df[m + COST_SUFFIX].mean() for m in models],
    }, index=pd.Index(models, name="model"))


def frontier(means):
    """True for models that no other model beats on BOTH cost and score.

    Strict on both sides (PRD 8.6), so two models with exactly the same
    mean cost both stay in; the routing rule then decides between them.
    """
    cost, score = means["mean_cost"].to_numpy(), means["mean_score"].to_numpy()
    dominated = [((cost < c) & (score > s)).any() for c, s in zip(cost, score)]
    return pd.Series(~np.array(dominated), index=means.index, name="on_frontier")


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def route(P, rel_cost, lam):
    """Index of the chosen candidate for each prompt.

    P: (n_prompts, n_candidates) probabilities; rel_cost: cbar_m / cbar_strong.
    Candidates must be ordered by increasing cost, so that np.argmax, which
    returns the first maximum, breaks exact ties toward the cheaper model.
    """
    return np.argmax(P - lam * rel_cost[None, :], axis=1)


def outcome(choice, scores, costs):
    """Mean quality and mean cost on test when prompt i goes to choice[i]."""
    rows = np.arange(len(choice))
    return scores[rows, choice].mean(), costs[rows, choice].mean()


def oracle_choice(scores, costs):
    """Per prompt: the cheapest candidate that is correct; if none is, the
    cheapest candidate overall (quality is 0 either way, so pay least)."""
    big = np.where(scores >= 1, costs, np.inf)
    any_right = np.isfinite(big).any(axis=1)
    return np.where(any_right, np.argmin(big, axis=1), np.argmin(costs, axis=1))


# --------------------------------------------------------------------------
# One seed
# --------------------------------------------------------------------------
def run_one(df, split, feature_set, router_name, grid, seed, profile,
            lambdas=config.MULTIMODEL_LAMBDAS):
    """Returns a dict of DataFrames: candidates, brier, curve, policies."""
    t0 = time.perf_counter()
    train, val, test = (df.iloc[split[k]] for k in ("train", "val", "test"))
    models = model_names(df.columns)

    # 1. Candidates and cost scale, from training rows only.
    means = model_means(train, models)
    means["on_frontier"] = frontier(means)
    cand = means[means["on_frontier"]].sort_values(["mean_cost", "mean_score"])
    names = list(cand.index)
    strongest = cand["mean_score"].idxmax()
    rel_cost = (cand["mean_cost"] / cand.loc[strongest, "mean_cost"]).to_numpy()

    # Features fit on train only, as in the two-model setting.
    feats = features.make(feature_set, profile, seed)
    X_train = feats.fit_transform(train)
    X_val, X_test = feats.transform(val), feats.transform(test)
    fixed = config.ROUTER_FEATURE_PARAMS.get(router_name, {}).get(feature_set, {})

    # 2. One classifier per candidate, chosen on validation Brier score.
    P = np.zeros((len(test), len(names)))
    brier_rows = []
    for j, m in enumerate(names):
        y_tr = (train[m] == 1).astype(int).to_numpy()
        y_va = (val[m] == 1).astype(int).to_numpy()
        best = None
        for tuned in grid_points(grid):
            params = {**fixed, **tuned}
            r = routers.make(router_name, seed, **params).fit(X_train, y_tr)
            b = brier_score_loss(y_va, r.score(X_val))
            if best is None or b < best[0]:  # ties keep the earlier grid point
                best = (b, params, r)
        val_brier, params, r = best
        P[:, j] = r.score(X_test)
        y_te = (test[m] == 1).astype(int).to_numpy()
        brier_rows.append({
            "model": m,
            "best_params": json.dumps(params, sort_keys=True, default=list),
            "val_brier": val_brier,
            "test_brier": brier_score_loss(y_te, P[:, j]),
            # Brier of always predicting the training accuracy: a classifier
            # that can't beat this has learned nothing about the prompts.
            "test_brier_constant": brier_score_loss(
                y_te, np.full(len(y_te), y_tr.mean())),
            "test_accuracy_rate": y_te.mean(),
        })

    # 3. Lambda sweep on test.
    scores = test[names].to_numpy(dtype=float)
    costs = test[[m + COST_SUFFIX for m in names]].to_numpy(dtype=float)
    curve_rows = []
    for lam in lambdas:
        choice = route(P, rel_cost, lam)
        q, c = outcome(choice, scores, costs)
        shares = np.bincount(choice, minlength=len(names)) / len(choice)
        curve_rows.append({"lambda": lam, "quality": q, "cost": c,
                           **{f"share:{m}": s for m, s in zip(names, shares)}})
    curve = pd.DataFrame(curve_rows)

    # 4. Reference policies on test (which model is "cheapest"/"strongest"
    #    was decided on train above).
    n = len(test)
    policy_choice = {
        "always_cheapest": np.zeros(n, dtype=int),           # names[0] is cheapest
        "always_strongest": np.full(n, names.index(strongest)),
        "oracle": oracle_choice(scores, costs),
    }
    policies = pd.DataFrame(
        [dict(zip(("quality", "cost"), outcome(ch, scores, costs))) for ch in policy_choice.values()],
        index=pd.Index(list(policy_choice), name="policy"))
    base = policies.loc["always_strongest", "cost"]
    curve["cost_pct"] = 100 * curve["cost"] / base
    policies["cost_pct"] = 100 * policies["cost"] / base

    common = {"seed": seed, "feature_set": feature_set, "router": router_name}
    return {
        "candidates": means.reset_index().assign(**common),
        "brier": pd.DataFrame(brier_rows).assign(**common),
        "curve": curve.assign(**common),
        "policies": policies.reset_index().assign(**common),
        "seconds": round(time.perf_counter() - t0, 2),
        "strongest": strongest,
    }


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-model routing (PRD 8.6)")
    parser.add_argument("--full", action="store_true", help="full data and every seed")
    parser.add_argument("--routers", default="lr",
                        help="comma-separated router names (PRD: the best two-model router plus lr)")
    parser.add_argument("--features", default="tfidf", choices=config.FEATURE_SETS)
    args = parser.parse_args(argv)

    from src import plots  # matplotlib only needed here, not for tests of the logic

    profile = config.FULL if args.full else config.SAMPLE
    df = data.load_data(verbose=False) if args.full else data.load_sample()
    out = profile["results_dir"] / "multimodel"
    (out / "curves").mkdir(parents=True, exist_ok=True)

    tables = {"candidates": [], "brier": [], "policies": []}
    for router_name in args.routers.split(","):
        grid = profile["grids"][router_name]
        for seed in profile["seeds"]:
            split = splits.random_split(df, seed)
            res = run_one(df, split, args.features, router_name, grid, seed, profile)
            stem = f"{args.features}_{router_name}_seed{seed}"
            res["curve"].to_csv(out / "curves" / f"{stem}.csv", index=False)
            for k in tables:
                tables[k].append(res[k])
            if seed == profile["seeds"][0]:
                pol = res["policies"].set_index("policy")
                plots.plot_multimodel(
                    res["curve"], pol, out / "figures" / f"multimodel_{stem}.png",
                    f"Multi-model routing | {profile['name']} | random split | "
                    f"{args.features} | {router_name} | seed {seed}")
            print(f"{router_name} seed {seed}: {res['seconds']} s, "
                  f"{len(res['brier'])} candidates, strongest = {res['strongest']}")

    suffix = f"{args.features}_{args.routers.replace(',', '-')}"
    for k, parts in tables.items():
        pd.concat(parts).to_csv(out / f"{k}_{suffix}.csv", index=False)
    print(f"Wrote {out.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
