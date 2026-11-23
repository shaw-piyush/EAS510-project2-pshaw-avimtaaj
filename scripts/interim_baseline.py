"""PRELIMINARY full-data baseline for the interim report (PRD 9, Sun Nov 22).

    python scripts/interim_baseline.py

Logistic regression on TF-IDF and on the hand-built features, random split,
every seed in config.SEEDS, FULL grids, hyperparameters chosen on
validation. Embeddings, the other routers and leave-one-benchmark-out come
later, so nothing here is a final result.

Writes to results/full/interim/:
    metrics_random.csv        one row per seed x feature set
    reference_policies.csv    always-cheap / always-strong / random / oracle per seed (test)
    curves/random_<features>_lr_seed<k>.csv   test curves
    summary.md                mean and std over seeds, as a Markdown table
    figures/quality_cost_lr_seed0.png         one figure for the interim PDF
    run_info.json             versions, commit, runtime, peak memory

Offline; needs only the pickle.
"""

import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

import config  # noqa: E402
from src import data, experiment, policies, splits  # noqa: E402

OUT = config.RESULTS_INTERIM
FEATURE_SETS = ["tfidf", "handbuilt"]
ROUTER = "lr"
LABEL = {"tfidf": "LR, TF-IDF", "handbuilt": "LR, hand-built"}

# Chart colours: categorical slots 1-2 of the dataviz reference palette
# (validated: CVD separation and contrast pass on the light surface), plus
# its neutral ink, grid and axis tones.
COLOR = {"tfidf": "#2a78d6", "handbuilt": "#eb6834"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=config.ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "config.py",
                                "scripts"], capture_output=True, text=True,
                               cwd=config.ROOT).stdout.strip() != ""
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def peak_memory_mb():
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (2**20 if sys.platform == "darwin" else 2**10), 1)


def mean_std(s, pct=False):
    m, sd = s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0
    return f"{m:.1f} ± {sd:.1f}" if pct else f"{m:.4f} ± {sd:.4f}"


def write_summary(metrics, refs, n_rows, runtime_s):
    seeds = sorted(metrics["seed"].unique())
    lines = [
        "# PRELIMINARY: logistic regression baseline, random split",
        "",
        f"Interim results, not final. Full data after cleaning ({n_rows:,} prompts), "
        f"cheap = `{config.CHEAP_MODEL}`, strong = `{config.STRONG_MODEL}`. "
        f"Random 70/15/15 split stratified by benchmark; {len(seeds)} seeds ({', '.join(map(str, seeds))}); "
        "`C` chosen on validation curve AUC; every number is measured on the test split, "
        "as mean ± standard deviation over seeds. Source: `metrics_random.csv` and "
        "`reference_policies.csv` in this folder.",
        "",
        "| Router | Test AUC | Lift over random | Cost@95%, test curve (% of always-strong) | "
        "Cost at validation threshold (%) | Met 95% target | ROC AUC | Chosen C |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fs in FEATURE_SETS:
        m = metrics[metrics["feature_set"] == fs]
        chosen = ", ".join(f"{json.loads(p)['C']:g}" for p in m.sort_values("seed")["best_params"])
        lines.append(
            f"| {LABEL[fs]} | {mean_std(m['test_auc'])} | {mean_std(m['lift'])} | "
            f"{mean_std(m['cost95_test_pct'], pct=True)} | {mean_std(m['cost_pct_at_t'], pct=True)} | "
            f"{int(m['met_target_at_t'].sum())} of {len(m)} seeds | {mean_std(m['roc_auc'])} | {chosen} |"
        )
    r = refs.groupby("policy")["quality"]
    rand = metrics.drop_duplicates("seed")["random_auc"]
    lines += [
        "",
        "Reference points on the same test splits (quality = mean score of the chosen model):",
        "",
        "| Policy | Quality | Cost (% of always-strong) |",
        "|---|---|---|",
    ]
    cost_pct = refs.assign(pct=100 * refs["cost"] / refs["strong_cost"])
    for p in ["always_cheap", "always_strong", "random", "oracle"]:
        q = r.get_group(p)
        c = cost_pct.loc[cost_pct["policy"] == p, "pct"]
        lines.append(f"| {p.replace('_', '-')} | {mean_std(q)} | {mean_std(c, pct=True)} |")
    lines += [
        "",
        f"Random routing's AUC (the baseline that lift is measured from): {mean_std(rand)}.",
        "",
        f"Runtime {runtime_s:.0f} s. Embeddings, the other routers and "
        "leave-one-benchmark-out are not run yet.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))


def plot(curves, refs, path):
    """Quality vs normalized cost on seed 0's test split."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ref = refs.set_index("policy")

    # Random routing: the straight line from always-cheap to always-strong.
    ax.plot([0, 1], [ref.loc["always_cheap", "quality"], ref.loc["always_strong", "quality"]],
            color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=2)
    # Direct labels sit just above each curve, at different x so they don't
    # meet; the text stays in neutral ink (the legend carries the colour).
    label_at = {"tfidf": 0.6, "handbuilt": 0.3}
    for fs in FEATURE_SETS:
        c = curves[fs]
        ax.plot(c["x"], c["quality"], color=COLOR[fs], lw=2, zorder=3, label=LABEL[fs])
        i = int(np.argmin(np.abs(c["x"].to_numpy() - label_at[fs])))
        ax.annotate(LABEL[fs], (c["x"].iloc[i], c["quality"].iloc[i]), xytext=(-4, 10),
                    textcoords="offset points", ha="right", fontsize=8, color=INK_2)

    # Reference points, labelled directly in neutral ink, placed away from
    # the curves (which start and end on the always-cheap/strong points).
    pts = {"always-cheap": ("always_cheap", (8, -12), "left"),
           "always-strong": ("always_strong", (-10, 8), "right"),
           "oracle": ("oracle", (8, 2), "left")}
    for text, (p, off, ha) in pts.items():
        x, y = ref.loc[p, "x"], ref.loc[p, "quality"]
        ax.scatter([x], [y], s=64, color=INK_2, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha,
                    fontsize=8, color=INK_2)
    mid = (ref.loc["always_cheap", "quality"] + ref.loc["always_strong", "quality"]) / 2
    ax.annotate("random routing", (0.5, mid), xytext=(4, -14), textcoords="offset points",
                fontsize=8, color=MUTED)

    ax.set_xlabel("Normalized cost (0 = always cheap, 1 = always strong)", color=INK_2, fontsize=9)
    ax.set_ylabel("Quality (mean score of the chosen model)", color=INK_2, fontsize=9)
    ax.set_title("PRELIMINARY: routing quality vs cost, logistic regression\n"
                 "test split, seed 0", color=INK, fontsize=10, loc="left")
    ax.set_xlim(-0.02, 1.02)
    ax.margins(y=0.08)  # room for the always-cheap label under its point
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    df = data.load_data(verbose=True)
    grid = config.GRIDS_FULL[ROUTER]
    rows, refs, seed0_curves = [], [], {}
    (OUT / "curves").mkdir(parents=True, exist_ok=True)

    for seed in config.SEEDS:
        split = splits.random_split(df, seed)
        test = df.iloc[split["test"]]
        ref = policies.reference_policies(*experiment.outcome_arrays(test))
        ref = ref.rename_axis("policy").reset_index()
        ref.insert(0, "seed", seed)
        ref["strong_cost"] = ref.loc[ref["policy"] == "always_strong", "cost"].iloc[0]
        refs.append(ref)
        for fs in FEATURE_SETS:
            row, curve = experiment.run_one(df, split, fs, ROUTER, grid, seed, config.FULL)
            rows.append(row)
            curve.to_csv(OUT / "curves" / f"random_{fs}_{ROUTER}_seed{seed}.csv", index=False)
            if seed == config.SEEDS[0]:
                seed0_curves[fs] = curve
            print(f"seed {seed} {fs:9s} C={json.loads(row['best_params'])['C']:<5g} "
                  f"test AUC {row['test_auc']:.4f}  lift {row['lift']:+.4f}  "
                  f"cost@95 {row['cost95_test_pct']:.1f}%  ({row['seconds']:.0f} s)")

    metrics = pd.DataFrame(rows)
    refs = pd.concat(refs, ignore_index=True)
    metrics.to_csv(OUT / "metrics_random.csv", index=False)
    refs.drop(columns="strong_cost").to_csv(OUT / "reference_policies.csv", index=False)
    runtime = time.perf_counter() - t0
    write_summary(metrics, refs, len(df), runtime)
    plot(seed0_curves, refs[refs["seed"] == config.SEEDS[0]],
         OUT / "figures" / "quality_cost_lr_seed0.png")

    sha, dirty = git_commit()
    info = {
        "status": "PRELIMINARY (interim report); not final results",
        "script": "scripts/interim_baseline.py",
        "git_commit": sha,
        "uncommitted_code_changes": dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(runtime, 1),
        "peak_memory_mb": peak_memory_mb(),
        "seeds": config.SEEDS,
        "feature_sets": FEATURE_SETS,
        "router": ROUTER,
        "grid": grid,
        "svd_components": config.FULL["svd_components"],
        "n_rows": len(df),
        "cheap_model": config.CHEAP_MODEL,
        "strong_model": config.STRONG_MODEL,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
    }
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print((OUT / "summary.md").read_text())
    print(f"Wrote results to {OUT.relative_to(config.ROOT)}/ in {runtime:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
