"""Train/validation/test splits (PRD 7.3, decision D4).

    from src import splits
    s = splits.random_split(df, seed)          # {"train": idx, "val": idx, "test": idx}
    s = splits.lobo_split(df, "mbpp", seed)    # mbpp is the whole test set
    for g in splits.lobo_groups(df): ...       # groups eligible to be held out (D3)

Indices are positions into `df` (the frame from src/data.py, which has a
fresh 0..n-1 index). Every split is checked before it's returned: no
sample_id in two parts, nothing lost, and a held-out group never in
train or validation.

Splits are redrawn per seed, so the spread of a metric over seeds includes
the luck of the split, not just of the model.
"""

import numpy as np
from sklearn.model_selection import train_test_split

import config
from src import data


# train_test_split can only stratify on labels that occur at least twice.
# In the small sample a group can have a single row in the 30% remainder,
# so groups with one row share a "_rare" stratum instead of crashing the
# split. A lone rare row joins the largest group's stratum.
def _strata(groups):
    """Group labels for stratification, with one-row groups merged."""
    groups = np.asarray(groups, dtype=object)
    values, counts = np.unique(groups, return_counts=True)
    rare = values[counts < 2]
    strata = np.where(np.isin(groups, rare), "_rare", groups).astype(object)
    if len(rare) == 1:
        strata[strata == "_rare"] = values[np.argmax(counts)]
    return strata


def _split(idx, groups, test_frac, seed):
    """One stratified split of positions `idx` into (rest, test)."""
    return train_test_split(idx, test_size=test_frac, random_state=seed,
                            stratify=_strata(groups[idx]))


def random_split(df, seed):
    """70/15/15 train/val/test, stratified by group (D4), two stages.

    Stage 1 takes 30% out for val+test; stage 2 halves it. Stratifying both
    stages keeps every group's share the same in all three parts.
    """
    groups = df["group"].to_numpy()
    idx = np.arange(len(df))
    rest_frac = config.VAL_FRAC + config.TEST_FRAC
    train, rest = _split(idx, groups, rest_frac, seed)
    val, test = _split(rest, groups, config.TEST_FRAC / rest_frac, seed)
    out = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    check_split(df, out)
    return out


def lobo_split(df, heldout, seed):
    """Leave-one-benchmark-out: all of `heldout` is test; the other groups
    are split 85/15 train/val, stratified by group, so validation (and so
    hyperparameter choice) never sees the held-out benchmark."""
    groups = df["group"].to_numpy()
    is_out = groups == heldout
    if not is_out.any():
        raise ValueError(f"group {heldout!r} not in the data")
    rest = np.flatnonzero(~is_out)
    train, val = _split(rest, groups, config.LOBO_VAL_FRAC, seed)
    out = {"train": np.sort(train), "val": np.sort(val), "test": np.flatnonzero(is_out)}
    check_split(df, out, heldout=heldout)
    return out


def lobo_groups(df, verbose=False):
    """Groups that may be held out on their own (D3 rule in config.py)."""
    table = data.heldout_table(df)
    if verbose:
        skipped = table.loc[~table["heldout_eligible"]]
        for _, r in skipped.iterrows():
            print(f"Not held out (too small, D3): {r['group']} "
                  f"({r['n_rows']} rows, cheap right {r['cheap_right']}, wrong {r['cheap_wrong']})")
    return table.loc[table["heldout_eligible"], "group"].tolist()


def check_split(df, split, heldout=None):
    """Raise if parts overlap, rows are lost, or a held-out group leaks."""
    ids = df["sample_id"].to_numpy()
    parts = {k: set(ids[v]) for k, v in split.items()}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = parts[a] & parts[b]
            if both:
                raise AssertionError(f"{len(both)} sample_ids in both {a} and {b}")
    n = sum(len(v) for v in split.values())
    if n != len(df):
        raise AssertionError(f"split covers {n} rows, data has {len(df)}")
    if heldout is not None:
        groups = df["group"].to_numpy()
        for k in ("train", "val"):
            if (groups[split[k]] == heldout).any():
                raise AssertionError(f"held-out group {heldout!r} appears in {k}")
        if not (groups[split["test"]] == heldout).all():
            raise AssertionError("test set holds rows from other groups")
