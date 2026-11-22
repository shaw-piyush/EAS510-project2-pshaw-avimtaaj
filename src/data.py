"""Load RouterBench 0-shot and apply the team's data decisions (PRD 6.3, 9 Wed).

    from src import data
    df = data.load_data()        # full data, cleaned; prints row counts per step
    df = data.load_sample()      # data/sample/sample.csv, same columns

Every decision value comes from config.py; this file only applies them, in
this order (each step prints how many rows it removed, for the report's Data
section):

1. D1:  drop the subtasks whose scores aren't 0/1 (config.EXCLUDED_EVAL_NAMES).
2. D3:  map subtasks to benchmark groups and drop config.EXCLUDED_GROUPS.
3. Join each prompt's turns into one text (config.PROMPT_TURN_ORDER).
4. D11: drop duplicates of the joined text within a group, before any split.
5. Labels: y = 1 if the cheap model (D2) scored 1.

The returned frame has, per row: sample_id, eval_name, group, prompt (the raw
stored string), text (the joined text every feature set reads), y, q_cheap,
q_strong, c_cheap, c_strong, and every model's score and cost column (the
multi-model extension needs them all). Model responses and the dataset's
oracle column are dropped: they're never features.

eval_name, group and sample_id are for grouping and splitting only, never
features (PRD 0.4): sample_id contains the benchmark name.
"""

import ast

import pandas as pd

import config

COST_SUFFIX = "|total_cost"
ID_COL = "sample_id"
EVAL_COL = "eval_name"
PROMPT_COL = "prompt"


# --------------------------------------------------------------------------
# Small pieces (each tested in tests/test_data.py)
# --------------------------------------------------------------------------
def model_names(columns):
    """Model names, taken from the cost columns so nothing is hard-coded."""
    return [c[: -len(COST_SUFFIX)] for c in columns if c.endswith(COST_SUFFIX)]


def group_of(eval_name):
    """D3: the parent benchmark of a subtask, by prefix (config.GROUP_PREFIXES)."""
    low = eval_name.lower()
    for prefix, group in config.GROUP_PREFIXES:
        if low.startswith(prefix.lower()):
            return group
    return eval_name


def parse_turns(raw):
    """Prompts are stored as str(list_of_turns). Return the list of turns.

    ast.literal_eval only evaluates Python literals, so it's safe on dataset
    text, and it turns the stored "\\n" escapes back into real newlines.
    Raises ValueError instead of guessing: the data check found no prompt
    that fails to parse, so a failure means the data changed.
    """
    try:
        turns = ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as e:
        raise ValueError(f"prompt is not a Python list literal: {raw[:80]!r}") from e
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"prompt is not a non-empty list: {raw[:80]!r}")
    return [str(t) for t in turns]


def join_turns(turns, order=None):
    """Join turns into one text. "last_first" puts the question before the
    instruction, so 256-token truncation cuts the instruction (config.py)."""
    order = order or config.PROMPT_TURN_ORDER
    if order == "last_first":
        turns = [turns[-1]] + turns[:-1]
    elif order != "in_order":
        raise ValueError(f"unknown PROMPT_TURN_ORDER: {order!r}")
    return config.PROMPT_TURN_SEPARATOR.join(turns)


def flatten_prompt(raw, order=None):
    """The text every feature set reads. Avimta's src/embed.py should call this."""
    return join_turns(parse_turns(raw), order)


def check_binary(df, models):
    """Raise unless every model's score is 0 or 1 (what D1 guarantees)."""
    bad = {m: int((~df[m].isin([0.0, 1.0])).sum()) for m in models}
    bad = {m: n for m, n in bad.items() if n}
    if bad:
        raise ValueError(f"scores other than 0/1 after D1 (rows per model): {bad}")


def drop_duplicates(df, models):
    """D11 on the joined text within each group.

    Returns (kept frame, number of extra copies dropped, number of rows in
    conflicting sets dropped). A set conflicts if any model's score differs
    between its copies; then every copy goes, because keeping one would pick
    the label arbitrarily.
    """
    key = ["group", "text"]
    in_set = df.duplicated(key, keep=False)
    n_conflict_rows = 0
    drop = pd.Series(False, index=df.index)
    if config.DROP_CONFLICTING_DUPLICATES and in_set.any():
        sets = df[in_set].groupby(key, sort=False)[models]
        conflicts = sets.transform("nunique").gt(1).any(axis=1)
        conflict_idx = conflicts[conflicts].index
        n_conflict_rows = len(conflict_idx)
        drop.loc[conflict_idx] = True
    extra = df.duplicated(key, keep="first") & ~drop
    drop |= extra
    return df[~drop], int(extra.sum()), n_conflict_rows


def add_labels(df):
    """D1/D2: y = 1 if the cheap model got it right; quality and cost of the pair."""
    cheap, strong = config.CHEAP_MODEL, config.STRONG_MODEL
    out = df.copy()
    out["y"] = (out[cheap] == 1.0).astype(int)
    out["q_cheap"] = out[cheap].astype(float)
    out["q_strong"] = out[strong].astype(float)
    out["c_cheap"] = out[cheap + COST_SUFFIX].astype(float)
    out["c_strong"] = out[strong + COST_SUFFIX].astype(float)
    return out


def _as_float(df, models):
    """Score columns are stored as object dtype holding floats (schema.csv)."""
    out = df.copy()
    for m in models:
        out[m] = out[m].astype(float)
        out[m + COST_SUFFIX] = out[m + COST_SUFFIX].astype(float)
    return out


def _check_pair(models):
    for name in ("CHEAP_MODEL", "STRONG_MODEL"):
        m = getattr(config, name)
        if m not in models:
            raise ValueError(f"config.{name} = {m!r} is not a model in the data: {models}")


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
def clean(raw, verbose=True):
    """Apply D1, D3, the prompt format, D11 and the labels to a raw frame.

    `raw` has the pickle's columns (sample_id, prompt, eval_name, one score
    column per model and one "<model>|total_cost" column per model). The
    step counts are also stored in result.attrs["steps"].
    """
    models = model_names(raw.columns)
    _check_pair(models)
    steps = []

    def step(name, before, after, note=""):
        steps.append({"step": name, "rows_before": before, "rows_after": after,
                      "rows_removed": before - after, "note": note})

    keep = [ID_COL, EVAL_COL, PROMPT_COL] + models + [m + COST_SUFFIX for m in models]
    df = _as_float(raw[keep], models)
    if not df[ID_COL].is_unique:
        raise ValueError("sample_id is not unique")
    n0 = len(df)
    step("loaded", n0, n0)

    # D1
    n = len(df)
    df = df[~df[EVAL_COL].isin(config.EXCLUDED_EVAL_NAMES)]
    check_binary(df, models)
    step("D1: drop subtasks with scores not 0/1", n, len(df),
         ", ".join(config.EXCLUDED_EVAL_NAMES))

    # D3
    n = len(df)
    df = df.assign(group=df[EVAL_COL].map(group_of))
    df = df[~df["group"].isin(config.EXCLUDED_GROUPS)]
    step("D3: drop excluded groups", n, len(df), ", ".join(config.EXCLUDED_GROUPS))

    # Prompt format, then D11 on the joined text
    df = df.assign(text=[flatten_prompt(p) for p in df[PROMPT_COL]])
    if config.DEDUPE_WITHIN_GROUP:
        n = len(df)
        df, n_extra, n_conflict = drop_duplicates(df, models)
        step("D11: drop duplicate text within a group", n, len(df),
             f"{n_extra} extra copies, {n_conflict} rows in sets with conflicting scores")

    df = add_labels(df).reset_index(drop=True)
    front = [ID_COL, EVAL_COL, "group", PROMPT_COL, "text", "y",
             "q_cheap", "q_strong", "c_cheap", "c_strong"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.attrs["steps"] = steps

    if verbose:
        print_steps(steps)
        print(f"Groups: {df['group'].nunique()}, "
              f"cheap model right on {df['y'].mean():.1%} of rows")
    return df


def print_steps(steps):
    t = pd.DataFrame(steps)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(t.to_string(index=False))


def load_data(verbose=True):
    """The full dataset, cleaned. Never uses the network."""
    if not config.RAW_PICKLE.exists():
        raise FileNotFoundError(
            f"{config.RAW_PICKLE.relative_to(config.ROOT)} not found. "
            "Run `python data/download.py` first."
        )
    return clean(pd.read_pickle(config.RAW_PICKLE), verbose=verbose)


def load_sample(path=None):
    """data/sample/sample.csv (Avimta's data/make_sample.py), same columns.

    The sample is drawn from load_data()'s rows, so D1/D3/D11 are already
    applied; this re-joins the turns from the raw `prompt` column, rebuilds
    the labels, and checks the D1/D3 guarantees still hold.
    """
    path = path or config.SAMPLE_CSV
    df = pd.read_csv(path)
    models = model_names(df.columns)
    _check_pair(models)
    df = _as_float(df, models)
    if not df[ID_COL].is_unique:
        raise ValueError("sample_id is not unique in the sample")
    check_binary(df, models)
    bad = set(df["group"]) & set(config.EXCLUDED_GROUPS)
    if bad:
        raise ValueError(f"sample contains excluded groups: {sorted(bad)}")
    df["text"] = [flatten_prompt(p) for p in df[PROMPT_COL]]
    return add_labels(df)


# --------------------------------------------------------------------------
# D3: which groups may be held out on their own
# --------------------------------------------------------------------------
def heldout_table(df):
    """Per group: rows, cheap right/wrong, and whether it may be held out.

    Uses the whole group, not a training split: in leave-one-benchmark-out
    the held-out group IS the test set, and this only decides which test
    sets are big enough to report, not anything a model is tuned on.
    """
    t = df.groupby("group").agg(n_rows=("y", "size"), cheap_right=("y", "sum"))
    t["cheap_wrong"] = t["n_rows"] - t["cheap_right"]
    t["heldout_eligible"] = (
        (t["n_rows"] >= config.HELDOUT_MIN_ROWS)
        & (t["cheap_right"] >= config.HELDOUT_MIN_CHEAP_RIGHT)
        & (t["cheap_wrong"] >= config.HELDOUT_MIN_CHEAP_WRONG)
    )
    return t.sort_values("n_rows", ascending=False).reset_index()


def heldout_groups(df):
    t = heldout_table(df)
    return t.loc[t["heldout_eligible"], "group"].tolist()


if __name__ == "__main__":
    d = load_data()
    print()
    print(heldout_table(d).to_string(index=False))
