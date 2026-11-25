"""Build the small bundled sample that `python main.py` runs on (D10, PRD 8.3).

    python data/make_sample.py

Reads the cleaned full data (src.data.load_data) and the full embedding
cache (src/embed.py's output), and writes to data/sample/:

    sample.csv       sample_id, prompt, group, and the score and cost
                     columns of every model. No model responses.
    embeddings.npy   float32 MiniLM vectors, row-aligned with sample.csv
    sample_ids.npy   the sample_id of each row, as plain strings
    SAMPLE_INFO.md   how the rows were chosen, rows per group, file sizes,
                     source and license note

Design choices:
- Rows come from load_data(), so D1 (0/1 scores only), D3 and D11 are
  already applied. That means GSM8K is NOT in the sample: D1 drops it
  because its scores aren't 0/1. SAMPLE_INFO.md says so.
- Stratified by group with a per-group minimum (config.SAMPLE_MIN_PER_GROUP),
  so every benchmark appears, including the small ones reported separately
  (MBPP, accounting_audit). See config.py for why.
- Fixed seed (config.SAMPLE_SEED): rerunning gives the identical sample.
- Vectors are copied from data/cache/ by sample_id, never recomputed, so
  this script never needs the network and the sample sees exactly the
  vectors the full run uses.
- Score and cost columns are kept for ALL models, not only the cheap/strong
  pair: the multi-model extension needs them, and they're small numbers.
- eval_name is left out: the sample needs only `group`, and the brief's
  minimal-columns rule (D10) plus the no-benchmark-id-features rule mean
  there's no reason to ship it.

Don't stage data/sample/ until the TA confirms we may share RouterBench rows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run as a script from anywhere: make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.data import COST_SUFFIX, ID_COL, PROMPT_COL, model_names  # noqa: E402

MAX_FILE_MB = 5  # repo rule: no committed file over 5 MB


def allocate(group_sizes, n_rows, min_per_group):
    """Rows to draw from each group.

    Each group first gets min(its size, min_per_group). The rows left over
    are shared out in proportion to group size (largest remainder, so the
    total is exactly n_rows), never more than a group has.
    """
    sizes = pd.Series(group_sizes).sort_index()
    if n_rows > sizes.sum():
        raise ValueError(f"asked for {n_rows} rows but the data has {sizes.sum()}")
    take = np.minimum(sizes, min_per_group)
    if take.sum() > n_rows:
        raise ValueError("SAMPLE_MIN_PER_GROUP x groups is more than SAMPLE_N_ROWS")
    # Hand out the rest a few rows at a time until it's all placed; a group
    # that runs out of rows simply stops receiving more.
    while take.sum() < n_rows:
        room = sizes - take
        left = n_rows - take.sum()
        share = left * room / room.sum()
        add = np.floor(share).astype(int)
        if add.sum() == 0:  # last few rows: biggest fractional shares first
            order = (share - add).sort_values(ascending=False).index
            add[order[:left]] = 1
        take = take + np.minimum(add, room)
    return take.astype(int)


def draw_sample(df, n_rows, min_per_group, seed):
    """Stratified sample of df's rows, in a stable order (group, then id)."""
    counts = allocate(df["group"].value_counts(), n_rows, min_per_group)
    rng = np.random.default_rng(seed)
    parts = []
    # Groups in sorted order and rows in sample_id order before drawing, so
    # the result depends only on the seed, not on the input row order.
    for group, k in counts.items():
        rows = df[df["group"] == group].sort_values(ID_COL)
        pick = rng.choice(len(rows), size=k, replace=False)
        parts.append(rows.iloc[np.sort(pick)])
    return pd.concat(parts).reset_index(drop=True)


def sample_columns(df):
    models = model_names(df.columns)
    return [ID_COL, PROMPT_COL, "group"] + [c for m in models for c in (m, m + COST_SUFFIX)]


def lookup_embeddings(ids, emb_path, ids_path):
    """Vectors for `ids` from the full cache, in the same order."""
    if not emb_path.exists() or not ids_path.exists():
        sys.exit(f"{emb_path} not found. Run `python -m src.embed` first.")
    emb = np.load(emb_path)
    cache_ids = np.load(ids_path)  # plain strings: no allow_pickle needed
    row = pd.Series(np.arange(len(cache_ids)), index=cache_ids.astype(str))
    missing = ~pd.Index(ids).isin(row.index)
    if missing.any():
        sys.exit(f"{int(missing.sum())} sample rows have no cached embedding")
    return emb[row.loc[ids].to_numpy()].astype(np.float32)


def write_info(sample, sizes_mb, path, n_full):
    per_group = sample["group"].value_counts().sort_index()
    lines = [
        "# data/sample: bundled RouterBench sample",
        "",
        "Built by `python data/make_sample.py` (decision D10). Used by `python main.py`.",
        "",
        "## How the rows were chosen",
        f"- Source: the cleaned data from `src.data.load_data()` ({n_full:,} rows after "
        "D1, D3 and D11).",
        f"- {len(sample):,} rows, stratified by benchmark group: every group gets at least "
        f"{config.SAMPLE_MIN_PER_GROUP} rows (or all of its rows if it has fewer), and the "
        "rest are shared out in proportion to group size.",
        f"- Random seed: {config.SAMPLE_SEED} (`config.SAMPLE_SEED`). Rerunning the script "
        "gives the identical sample.",
        "- GSM8K (`grade-school-math`) is not included: decision D1 drops it from the whole "
        "project because its scores are not 0/1.",
        "",
        "## Rows per group",
        "",
        "| Group | Rows |",
        "|---|---|",
        *[f"| {g} | {n} |" for g, n in per_group.items()],
        "",
        "## Files",
        "",
        "| File | Contents | Size (MB) |",
        "|---|---|---|",
        f"| sample.csv | sample_id, prompt, group, score and cost of each model; no model "
        f"responses | {sizes_mb['sample.csv']:.2f} |",
        f"| embeddings.npy | float32 {config.EMBEDDING_MODEL} vectors (normalized), "
        f"row-aligned with sample.csv | {sizes_mb['embeddings.npy']:.2f} |",
        f"| sample_ids.npy | sample_id of each embedding row | "
        f"{sizes_mb['sample_ids.npy']:.2f} |",
        "",
        "## Source and license",
        f"- RouterBench 0-shot, Hugging Face dataset `{config.DATASET_REPO}`, pinned "
        "revision in `config.py`.",
        "- The dataset page lists no license. These rows are shared in this repository "
        "only with the course staff's permission. [Update with the TA's reply and date.]",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    from src.data import load_data  # here, so importing this module stays cheap for tests

    df = load_data(verbose=False)
    sample = draw_sample(df, config.SAMPLE_N_ROWS, config.SAMPLE_MIN_PER_GROUP,
                         config.SAMPLE_SEED)
    ids = sample[ID_COL].to_numpy(dtype=str)  # fixed-width strings, saved without pickle
    emb = lookup_embeddings(ids, config.EMB_CACHE, config.EMB_IDS)

    out = config.SAMPLE_DIR
    out.mkdir(parents=True, exist_ok=True)
    sample[sample_columns(sample)].to_csv(config.SAMPLE_CSV, index=False)
    np.save(config.SAMPLE_EMBEDDINGS, emb)
    np.save(config.SAMPLE_IDS, ids)

    files = {"sample.csv": config.SAMPLE_CSV, "embeddings.npy": config.SAMPLE_EMBEDDINGS,
             "sample_ids.npy": config.SAMPLE_IDS}
    sizes_mb = {name: p.stat().st_size / 1e6 for name, p in files.items()}
    write_info(sample, sizes_mb, config.SAMPLE_INFO, len(df))

    print(f"Sample: {len(sample):,} rows from {len(df):,}, seed {config.SAMPLE_SEED}")
    print(sample["group"].value_counts().sort_index().to_string())
    for name, mb in sizes_mb.items():
        print(f"  {name:15s} {mb:6.2f} MB")
    print(f"  total           {sum(sizes_mb.values()):6.2f} MB")
    too_big = [n for n, mb in sizes_mb.items() if mb >= MAX_FILE_MB]
    if too_big:
        sys.exit(f"Over {MAX_FILE_MB} MB, can't be committed: {too_big}")
    print("Don't stage data/sample/ until the TA confirms we may share it.")


if __name__ == "__main__":
    main()
