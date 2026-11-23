"""Figures for the report (PRD 8.4). matplotlib only, never plt.show().

    plot_quality_cost(curves, policies, path, title)    two-model curves
    plot_per_benchmark(table, value_col, path, title)   grouped bars per benchmark
    plot_multimodel(curve, policies, path, title)       multi-model lambda sweep

Design choices:
- The "Agg" backend draws to files only, so `python main.py` never opens a
  window or waits for one (the sample run must finish unattended).
- Every router keeps the same color in every figure (ROUTER_COLORS), so a
  reader can follow "kNN" across the report without re-reading legends.
- The quality-vs-cost axis is NORMALIZED cost (0 = always-cheap,
  1 = always-strong), the same x that metrics.curve_auc integrates over,
  so the area a reader sees is the AUC we report.
- Functions take DataFrames and a path and only draw: no metric is computed
  here, so a figure can never disagree with the CSV it was drawn from.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; must come before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# One fixed color per router (colorblind-safe Okabe-Ito palette). Unknown
# names fall back to grey, so a new router still plots without a crash.
ROUTER_COLORS = {
    "lr": "#0072B2",      # blue: the baseline
    "knn": "#E69F00",     # orange
    "kmeans": "#009E73",  # green
    "hgb": "#D55E00",     # vermillion
    "rf": "#CC79A7",      # pink
    "mlp": "#56B4E9",     # sky blue
}
FALLBACK_COLOR = "#777777"
POLICY_STYLE = {
    "always_cheap": dict(marker="s", color="black", label="always-cheap"),
    "always_strong": dict(marker="^", color="black", label="always-strong"),
    "oracle": dict(marker="*", color="#B8860B", label="oracle", markersize=13),
}
FIGSIZE = (6.4, 4.8)  # readable at full text width in report.md
DPI = 150


def color_of(name):
    """Color for a router; also accepts names like 'lr_tfidf' by prefix."""
    key = str(name).lower()
    if key in ROUTER_COLORS:
        return ROUTER_COLORS[key]
    for router, color in ROUTER_COLORS.items():
        if key.startswith(router + "_"):
            return color
    return FALLBACK_COLOR


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)  # free memory: the full run draws many figures
    return path


def plot_quality_cost(curves, policies, path, title):
    """Quality vs normalized cost: one line per router plus reference points.

    curves:   {router name: DataFrame from metrics.routing_curve} (needs x, quality)
    policies: DataFrame from policies.reference_policies (index = policy
              names; needs quality, and x for the markers)
    title:    should name the split, setting and feature set (PRD 8.4)
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Random routing: the straight line between the two endpoints, dashed.
    qc = policies.loc["always_cheap", "quality"]
    qs = policies.loc["always_strong", "quality"]
    ax.plot([0, 1], [qc, qs], linestyle="--", color="grey", linewidth=1, label="random")

    for name, curve in curves.items():
        ax.plot(curve["x"], curve["quality"], color=color_of(name), linewidth=1.8, label=name)

    for policy, style in POLICY_STYLE.items():
        if policy in policies.index:
            style = dict(style)
            ax.plot(policies.loc[policy, "x"], policies.loc[policy, "quality"],
                    linestyle="none", markersize=style.pop("markersize", 9), **style)

    ax.set_xlabel("Normalized cost (0 = always-cheap, 1 = always-strong)")
    ax.set_ylabel("Quality (mean correctness)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_per_benchmark(table, value_col, path, title, group_col="group", router_col="router",
                       ylabel=None):
    """Grouped bar chart: one cluster per benchmark, one bar per router.

    table: long format, one row per (benchmark, router), e.g. per-benchmark
           AUC lift for Piyush's error analysis. An optional `<value_col>_std`
           column is drawn as error bars (std over seeds).
    """
    wide = table.pivot(index=group_col, columns=router_col, values=value_col)
    std_col = f"{value_col}_std"
    err = table.pivot(index=group_col, columns=router_col, values=std_col) \
        if std_col in table.columns else None

    routers = list(wide.columns)
    n_groups, n_routers = len(wide.index), len(routers)
    width = 0.8 / max(n_routers, 1)
    fig, ax = plt.subplots(figsize=(max(FIGSIZE[0], 0.7 * n_groups * max(n_routers, 1) * 0.5 + 2),
                                    FIGSIZE[1]))
    for i, router in enumerate(routers):
        xs = [g + (i - (n_routers - 1) / 2) * width for g in range(n_groups)]
        ax.bar(xs, wide[router], width=width, color=color_of(router), label=router,
               yerr=None if err is None else err[router], capsize=2)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(wide.index, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel or value_col)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_multimodel(curve, policies, path, title):
    """Multi-model routing: quality vs cost as lambda is swept (PRD 8.6).

    curve:    DataFrame with cost and quality, one row per lambda
    policies: DataFrame indexed by always_cheapest, always_strongest, oracle
              with cost and quality
    Cost is shown as a percentage of always-strongest cost, because with
    several models there is no single "cheap" end to normalize against.
    """
    base = policies.loc["always_strongest", "cost"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(100 * curve["cost"] / base, curve["quality"], marker="o", markersize=3,
            color=ROUTER_COLORS["lr"], label="multi-model router")
    styles = {
        "always_cheapest": dict(marker="s", color="black", label="always-cheapest"),
        "always_strongest": dict(marker="^", color="black", label="always-strongest"),
        "oracle": dict(marker="*", color="#B8860B", label="oracle", markersize=13),
    }
    for policy, style in styles.items():
        if policy in policies.index:
            style = dict(style)
            ax.plot(100 * policies.loc[policy, "cost"] / base, policies.loc[policy, "quality"],
                    linestyle="none", markersize=style.pop("markersize", 9), **style)
    ax.set_xlabel("Cost (% of always-strongest)")
    ax.set_ylabel("Quality (mean correctness)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)
