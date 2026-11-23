"""D2 check: is GPT-4/Mixtral still the right pair on the TRAINING split?

    python scripts/d2_check.py

D2 was chosen from whole-dataset means (results/full/data_check/
model_summary.csv), which include rows that later become test. This
recomputes each model's mean score and cost on the training part of every
seed's random split only (PRD 0.4) and checks that:
- the strong model has the highest mean training score, and
- the cheap model costs a small fraction of the strong one and trails it by
  a clear gap.

Writes results/full/d2_check/model_ranking_train.csv (one row per seed and
model) and prints a verdict. Offline; needs only the pickle.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data, splits  # noqa: E402

OUT = config.D2_CHECK_DIR / "model_ranking_train.csv"


def main():
    df = data.load_data(verbose=False)
    models = data.model_names(df.columns)
    rows = []
    for seed in config.SEEDS:
        train = df.iloc[splits.random_split(df, seed)["train"]]
        strong_cost = train[config.STRONG_MODEL + data.COST_SUFFIX].mean()
        means = pd.DataFrame({
            "model": models,
            "mean_score_train": [train[m].mean() for m in models],
            "mean_cost_train": [train[m + data.COST_SUFFIX].mean() for m in models],
        })
        means["cost_vs_strong"] = means["mean_cost_train"] / strong_cost
        means = means.sort_values("mean_score_train", ascending=False)
        means.insert(0, "rank_by_score", range(1, len(means) + 1))
        means.insert(0, "seed", seed)
        means.insert(2, "n_train", len(train))
        rows.append(means)
    table = pd.concat(rows, ignore_index=True)

    config.D2_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)

    pick = table[table["model"].isin([config.STRONG_MODEL, config.CHEAP_MODEL])]
    print(pick.to_string(index=False))
    strong_first = (table.loc[table["model"] == config.STRONG_MODEL, "rank_by_score"] == 1).all()
    cheap = table[table["model"] == config.CHEAP_MODEL]
    strong = table[table["model"] == config.STRONG_MODEL].set_index("seed")
    gap = strong.loc[cheap["seed"], "mean_score_train"].to_numpy() - cheap["mean_score_train"].to_numpy()
    print(f"\n{config.STRONG_MODEL} has the highest training score in every seed: {strong_first}")
    print(f"{config.CHEAP_MODEL}: score gap {gap.min():.3f} to {gap.max():.3f}, "
          f"cost {cheap['cost_vs_strong'].min():.1%} to {cheap['cost_vs_strong'].max():.1%} of strong")
    print(f"Wrote {OUT.relative_to(config.ROOT)}")
    return 0 if strong_first else 1


if __name__ == "__main__":
    sys.exit(main())
