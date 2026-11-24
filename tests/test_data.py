"""Tests for src/data.py. Synthetic frames, plus the committed data check
tables (results/full/data_check/), so no test needs the 100 MB pickle."""

import pandas as pd
import pytest

import config
from src import data

CHECK = config.DATA_CHECK_DIR
CHEAP, STRONG = config.CHEAP_MODEL, config.STRONG_MODEL


def raw_frame(rows):
    """A frame shaped like the pickle, with the two configured models only.

    rows: (sample_id, eval_name, prompt, cheap_score, strong_score)
    """
    df = pd.DataFrame(rows, columns=["sample_id", "eval_name", "prompt", CHEAP, STRONG])
    # Stored as object dtype holding floats, like the real pickle (schema.csv).
    df[CHEAP] = df[CHEAP].astype(object)
    df[STRONG] = df[STRONG].astype(object)
    df[CHEAP + data.COST_SUFFIX] = 0.0001
    df[STRONG + data.COST_SUFFIX] = 0.003
    return df


# --- prompt format -----------------------------------------------------------
def test_last_turn_comes_first():
    raw = "['Fixed instruction.', 'The question?']"
    assert data.flatten_prompt(raw, "last_first") == "The question?\n\nFixed instruction."
    assert data.flatten_prompt(raw, "in_order") == "Fixed instruction.\n\nThe question?"


def test_single_turn_is_unchanged_by_order():
    raw = "['Only turn']"
    assert data.flatten_prompt(raw, "last_first") == data.flatten_prompt(raw, "in_order") == "Only turn"


def test_stored_escapes_become_newlines():
    raw = "['line one\\nline two']"  # the pickle stores a backslash-n
    assert data.flatten_prompt(raw) == "line one\nline two"


@pytest.mark.parametrize("raw", ["not a list", "[]", "{'a': 1}", "['unclosed"])
def test_bad_prompts_raise(raw):
    with pytest.raises(ValueError):
        data.flatten_prompt(raw)


def test_unknown_order_raises():
    with pytest.raises(ValueError):
        data.join_turns(["a"], "sideways")


# --- D3 grouping ---------------------------------------------------------------
def test_grouping_matches_the_data_check_mapping():
    # The decision was "accept the data check's mapping", so every one of the
    # 86 subtasks must land in the same group as in eval_name_counts.csv.
    t = pd.read_csv(CHECK / "eval_name_counts.csv")
    assert len(t) == 86
    got = t["eval_name"].map(data.group_of)
    assert (got == t["candidate_group"]).all()
    assert got.nunique() == 13


# --- D1 matches the evidence -----------------------------------------------------
def test_excluded_subtasks_are_exactly_the_non_binary_ones():
    t = pd.read_csv(CHECK / "score_binary_by_eval.csv")
    non_binary = set(t.loc[~t["all_models_binary"], "eval_name"])
    assert set(config.EXCLUDED_EVAL_NAMES) == non_binary


def test_non_binary_score_after_d1_raises():
    df = raw_frame([("a", "mmlu-x", "['q1']", 0.5, 1.0), ("b", "mmlu-x", "['q2']", 1.0, 1.0)])
    with pytest.raises(ValueError, match="0/1"):
        data.clean(df, verbose=False)


# --- the pipeline ----------------------------------------------------------------
def test_clean_applies_d1_d3_and_labels():
    df = raw_frame([
        ("a", "mmlu-x", "['q1']", 1.0, 1.0),
        ("b", "mmlu-y", "['q2']", 0.0, 1.0),
        ("c", "grade-school-math", "['inst', 'q3']", 0.25, 0.75),  # D1
        ("d", "test-match", "['q4']", 1.0, 0.0),                   # D3
        ("e", "Chinese_riddles", "['inst', 'q5']", 0.0, 0.0),
    ])
    out = data.clean(df, verbose=False)
    assert list(out["sample_id"]) == ["a", "b", "e"]
    assert list(out["group"]) == ["mmlu", "mmlu", "chinese"]
    assert list(out["y"]) == [1, 0, 0]
    assert list(out["text"])[2] == "q5\n\ninst"
    assert out["q_cheap"].dtype == float and out["c_strong"].iloc[0] == 0.003
    steps = {s["step"].split(":")[0]: s["rows_removed"] for s in out.attrs["steps"]}
    assert steps["D1"] == 1 and steps["D3"] == 1 and steps["D11"] == 0


def test_dedupe_keeps_first_copy_within_a_group_only():
    df = raw_frame([
        ("a", "mmlu-x", "['same']", 1.0, 1.0),
        ("b", "mmlu-y", "['same']", 1.0, 1.0),       # same group: dropped
        ("c", "hellaswag", "['same']", 0.0, 1.0),    # other group: kept
    ])
    out = data.clean(df, verbose=False)
    assert list(out["sample_id"]) == ["a", "c"]


def test_dedupe_drops_every_copy_when_scores_conflict():
    df = raw_frame([
        ("a", "mbpp", "['p']", 1.0, 1.0),
        ("b", "mbpp", "['p']", 0.0, 1.0),   # cheap score disagrees: drop both
        ("c", "mbpp", "['r']", 1.0, 0.0),
        ("d", "mbpp", "['r']", 1.0, 0.0),   # agrees: keep the first
        ("e", "mbpp", "['r']", 1.0, 0.0),
    ])
    out = data.clean(df, verbose=False)
    assert list(out["sample_id"]) == ["c"]
    d11 = [s for s in out.attrs["steps"] if s["step"].startswith("D11")][0]
    assert d11["rows_removed"] == 4 and "2 extra copies, 2 rows" in d11["note"]


def test_dedupe_uses_the_joined_text_not_the_raw_string():
    # Two raw strings with different quoting that join to the same text.
    df = raw_frame([
        ("a", "mmlu-x", "['inst', 'q']", 1.0, 1.0),
        ("b", "mmlu-x", '["inst", "q"]', 1.0, 1.0),
    ])
    out = data.clean(df, verbose=False)
    assert list(out["sample_id"]) == ["a"]


def test_unknown_model_in_config_raises(monkeypatch):
    monkeypatch.setattr(config, "CHEAP_MODEL", "no-such-model")
    df = raw_frame([("a", "mmlu-x", "['q']", 1.0, 1.0)])
    with pytest.raises(ValueError, match="CHEAP_MODEL"):
        data.clean(df, verbose=False)


# --- D3 held-out rule --------------------------------------------------------------
def test_heldout_rule_needs_rows_and_both_outcomes():
    n = config.HELDOUT_MIN_ROWS
    right, wrong = config.HELDOUT_MIN_CHEAP_RIGHT, config.HELDOUT_MIN_CHEAP_WRONG
    df = pd.DataFrame({
        "group": ["big"] * n + ["few_wrong"] * n + ["small"] * (n - 1),
        "y": [1] * right + [0] * (n - right)              # big: exactly at both limits
             + [1] * (n - wrong + 1) + [0] * (wrong - 1)  # few_wrong: one wrong too few
             + [1] * right + [0] * (n - 1 - right),       # small: one row too few
    })
    assert data.heldout_groups(df) == ["big"]


# --- group summary (scripts/data_summary.py) ---------------------------------------
def test_group_summary_counts_and_all_row():
    df = pd.DataFrame({
        "group": ["b"] * 2 + ["a"] * 2 + ["c"] * 4,   # a and b tie on size
        "y":        [1, 0, 1, 1, 0, 0, 0, 1],
        "q_strong": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    })
    s = data.group_summary(df)
    # Largest group first, ties broken by name, ALL last.
    assert s["group"].tolist() == ["c", "a", "b", "ALL"]
    c, total = s.iloc[0], s.iloc[-1]
    assert (c["n_rows"], c["cheap_right"], c["cheap_wrong"]) == (4, 1, 3)
    assert c["cheap_accuracy"] == pytest.approx(0.25)
    assert c["strong_accuracy"] == pytest.approx(0.75)
    assert s["share_of_rows"].iloc[:-1].sum() == pytest.approx(1.0)
    assert (total["n_rows"], total["cheap_right"]) == (8, 4)
    assert total["cheap_accuracy"] == pytest.approx(0.5)
    assert total["strong_accuracy"] == pytest.approx(6 / 8)
    # Groups this small are never eligible to be held out (HELDOUT_MIN_ROWS).
    assert s["heldout_eligible"].iloc[:-1].tolist() == [False, False, False]
