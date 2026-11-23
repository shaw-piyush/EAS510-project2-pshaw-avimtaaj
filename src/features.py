"""The three feature sets (PRD 7.4, decisions D5-D7).

    from src import features
    f = features.make("tfidf", profile=config.FULL, seed=0)
    X_train = f.fit_transform(df.iloc[train])    # fit on TRAIN only
    X_val = f.transform(df.iloc[val])

Every feature set reads only the `text` column (and `sample_id`, to look up
embeddings). eval_name, group and sample_id are never features (PRD 0.4).

- tfidf:     TF-IDF -> TruncatedSVD -> Normalizer, one Pipeline fit on train.
- handbuilt: 13 counts and flags computed from the text alone (stateless,
             so there's nothing to fit). Scaling happens inside the routers.
- embedding: frozen MiniLM vectors made by Avimta's src/embed.py, looked up
             by sample_id. Never computed here (the sample run must not
             load the model).
"""

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

import config

# "Lines starting A., A) or (A)" (PRD 7.4). Only the letter A: any prompt
# with answer choices has an A line, and matching B-E as well would only add
# false positives such as "B. Smith" at the start of a line.
_CHOICE_RE = re.compile(r"^\s*(?:\(A\)|A[.)])", re.MULTILINE)
# Whole words only. "for" and "while" are also ordinary English words, so
# this flag fires on much non-code text; the report says the hand-built
# features are crude on purpose.
_KEYWORD_RE = re.compile(r"\b(?:" + "|".join(config.CODE_KEYWORDS) + r")\b")


# --------------------------------------------------------------------------
# Hand-built features
# --------------------------------------------------------------------------
def handbuilt_row(text):
    """The 13 features of one prompt, in config.HANDBUILT_FEATURES order."""
    n_chars = len(text)
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    n_digits = sum(c.isdigit() for c in text)
    n_code = sum(text.count(c) for c in config.CODE_SYMBOLS)
    per_char = max(n_chars, 1)  # ratios of an empty text are 0, not NaN
    return {
        "n_chars": n_chars,
        "n_words": len(words),
        "n_lines": text.count("\n") + 1,
        "n_digits": n_digits,
        "digit_ratio": n_digits / per_char,
        "n_code_symbols": n_code,
        "code_symbol_ratio": n_code / per_char,
        "n_question_marks": text.count("?"),
        "has_answer_choices": int(bool(_CHOICE_RE.search(text))),
        "n_math_symbols": sum(text.count(c) for c in config.MATH_SYMBOLS),
        # Share of letters that are capitals (not of all characters, so
        # digits and spaces don't dilute it).
        "uppercase_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "has_code_keyword": int(bool(_KEYWORD_RE.search(text))),
    }


def handbuilt(texts):
    rows = [handbuilt_row(t) for t in texts]
    return pd.DataFrame(rows, columns=config.HANDBUILT_FEATURES).to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Feature sets with one fit/transform interface
# --------------------------------------------------------------------------
class TfidfFeatures:
    """TF-IDF + SVD + L2 normalization. Normalizing after SVD makes the
    vectors unit length, so cosine distance (kNN) and dot products agree."""

    def __init__(self, svd_components, seed):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**config.TFIDF_PARAMS)),
            ("svd", TruncatedSVD(n_components=svd_components, random_state=seed)),
            ("norm", Normalizer()),
        ])

    def fit_transform(self, df):
        return self.pipe.fit_transform(df["text"])

    def transform(self, df):
        return self.pipe.transform(df["text"])


class HandbuiltFeatures:
    """Stateless: computed from each text alone, so fitting learns nothing
    and can't leak anything from validation or test."""

    def fit_transform(self, df):
        return handbuilt(df["text"])

    def transform(self, df):
        return handbuilt(df["text"])


class EmbeddingFeatures:
    """Precomputed MiniLM vectors, looked up by sample_id.

    Full runs read data/cache/ (written by src/embed.py); the sample run
    reads data/sample/ (written by data/make_sample.py).
    """

    def __init__(self, emb_path, ids_path):
        if not emb_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"{emb_path.relative_to(config.ROOT)} or {ids_path.name} is missing. "
                "Full data: run `python -m src.embed` (Avimta's script) first.")
        self.emb = np.load(emb_path)
        try:
            ids = np.load(ids_path)  # no allow_pickle: ids must be plain strings
        except ValueError as e:
            raise ValueError(
                f"{ids_path.name} holds Python objects, not strings, so it can't be "
                "loaded safely. src/embed.py should save the ids with "
                "to_numpy(dtype=str) (reported to Avimta).") from e
        if len(ids) != len(self.emb):
            raise ValueError(f"{len(ids)} ids for {len(self.emb)} embeddings")
        self.row = pd.Series(np.arange(len(ids)), index=pd.Index(ids.astype(str)))
        if not self.row.index.is_unique:
            raise ValueError(f"duplicate ids in {ids_path.name}")

    def fit_transform(self, df):
        return self.transform(df)

    def transform(self, df):
        ids = df["sample_id"].astype(str)
        missing = ~ids.isin(self.row.index)
        if missing.any():
            raise KeyError(f"{int(missing.sum())} prompts have no embedding, "
                           f"e.g. {ids[missing].iloc[0]!r}")
        return self.emb[self.row.loc[ids].to_numpy()]


def make(name, profile, seed):
    """A fresh feature set for one run (one seed, one split)."""
    if name == "tfidf":
        return TfidfFeatures(profile["svd_components"], seed)
    if name == "handbuilt":
        return HandbuiltFeatures()
    if name == "embedding":
        if profile["name"] == "sample":
            return EmbeddingFeatures(config.SAMPLE_EMBEDDINGS, config.SAMPLE_IDS)
        return EmbeddingFeatures(config.EMB_CACHE, config.EMB_IDS)
    raise ValueError(f"unknown feature set {name!r}; expected one of {config.FEATURE_SETS}")
