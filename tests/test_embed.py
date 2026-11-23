"""Tests for src/embed.py that need no model download (no network)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on the import path (Piyush's pytest.ini also does this once merged).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import embed  # noqa: E402
from src.data import ID_COL, PROMPT_COL  # noqa: E402


def test_saved_ids_load_without_pickle(tmp_path, monkeypatch):
    # A tiny stand-in for the raw pickle, read through the script's own loader.
    raw = pd.DataFrame({
        ID_COL: ["mmlu.a.0", "mbpp.1", "hellaswag.22"],
        PROMPT_COL: ["['q one']", "['instr', 'q two']", "['q three']"],
    })
    fake = tmp_path / "raw.pkl"
    raw.to_pickle(fake)
    monkeypatch.setattr(config, "RAW_PICKLE", fake)

    ids, texts = embed.load_prompts()
    path = tmp_path / "emb_ids.npy"
    np.save(path, ids)  # same call run() uses

    arr = np.load(path)  # default allow_pickle=False: must not raise
    assert arr.dtype.kind == "U"
    assert arr.tolist() == raw[ID_COL].tolist()
    assert len(texts) == len(raw)
