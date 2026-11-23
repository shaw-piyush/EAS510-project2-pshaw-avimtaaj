"""Tests for src/features.py: hand-built values, TF-IDF fit on train only,
and embedding lookup by id."""

import numpy as np
import pandas as pd
import pytest

import config
from src import features


def test_handbuilt_values_on_a_known_text():
    text = "What is 2+2?\nA) 3\nB) 4"
    row = features.handbuilt_row(text)
    assert list(row) == config.HANDBUILT_FEATURES
    assert row["n_chars"] == len(text)
    assert row["n_words"] == 7
    assert row["n_lines"] == 3
    assert row["n_digits"] == 4
    assert row["n_question_marks"] == 1
    assert row["has_answer_choices"] == 1
    assert row["n_math_symbols"] == 1          # the "+"
    assert row["n_code_symbols"] == 2          # the two ")"
    assert row["uppercase_ratio"] == pytest.approx(3 / 8)  # W, A, B of 8 letters
    assert row["has_code_keyword"] == 0


@pytest.mark.parametrize("text,flag", [
    ("(A) yes\n(B) no", 1), ("A. yes", 1), ("  A) indented", 1),
    ("A cat sat.", 0), ("Answer: B", 0),
])
def test_answer_choice_flag(text, flag):
    assert features.handbuilt_row(text)["has_answer_choices"] == flag


@pytest.mark.parametrize("text,flag", [
    ("def f(x):\n    return x", 1), ("import os", 1), ("defined terms", 0), ("classy", 0),
])
def test_code_keyword_is_whole_word(text, flag):
    assert features.handbuilt_row(text)["has_code_keyword"] == flag


def test_empty_text_gives_zeros_not_nan():
    row = features.handbuilt_row("")
    assert not any(np.isnan(v) for v in row.values())


def test_tfidf_is_fit_on_train_only():
    train = pd.DataFrame({"text": ["alpha beta", "beta gamma", "alpha gamma",
                                   "alpha beta gamma", "beta beta alpha"]})
    test = pd.DataFrame({"text": ["zeta omega", "alpha beta"]})
    f = features.TfidfFeatures(svd_components=2, seed=0)
    f.fit_transform(train)
    vocab = set(f.pipe.named_steps["tfidf"].vocabulary_)
    assert "zeta" not in vocab and "alpha" in vocab
    X = f.transform(test)
    assert X.shape == (2, 2)
    # Unit length after Normalizer, except an all-unknown row, which stays 0.
    norms = np.linalg.norm(X, axis=1)
    assert norms[0] == pytest.approx(0.0) and norms[1] == pytest.approx(1.0)


def test_embeddings_are_looked_up_by_id(tmp_path):
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.save(tmp_path / "e.npy", emb)
    np.save(tmp_path / "i.npy", np.array(["a", "b", "c", "d"]))
    f = features.EmbeddingFeatures(tmp_path / "e.npy", tmp_path / "i.npy")
    X = f.transform(pd.DataFrame({"sample_id": ["c", "a"]}))
    assert np.array_equal(X, emb[[2, 0]])
    with pytest.raises(KeyError):
        f.transform(pd.DataFrame({"sample_id": ["zz"]}))


def test_object_array_ids_are_refused(tmp_path, monkeypatch):
    np.save(tmp_path / "e.npy", np.zeros((1, 3), dtype=np.float32))
    np.save(tmp_path / "i.npy", np.array(["a"], dtype=object), allow_pickle=True)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="dtype=str"):
        features.EmbeddingFeatures(tmp_path / "e.npy", tmp_path / "i.npy")


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError):
        features.make("bag_of_vibes", config.SAMPLE, seed=0)
