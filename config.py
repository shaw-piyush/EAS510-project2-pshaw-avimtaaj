"""Every seed, path, decision value and hyperparameter for the project.

Nothing else in the repo should hard-code a seed, a path or a grid; import it
from here. That way a run is fully described by this file plus the code, and
`run_info.json` can snapshot it.

Decisions D1-D11 (PRD Section 6.3) were decided on 2026-11-22: D1-D3, D11
and the prompt format from the data check evidence, D4-D9 as the PRD
defaults. D10 (the sample) waits on the TA. A `TODO(D#)` marks anything
still to be confirmed.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------
# Five seeds so every metric can be reported as mean and std over redrawn
# splits; the spread across seeds is the yardstick for "is this difference
# real?" in the features-vs-models analysis.
SEEDS = [0, 1, 2, 3, 4]

# --------------------------------------------------------------------------
# Dataset (PRD 6.1, 7.2)
# --------------------------------------------------------------------------
DATASET_REPO = "withmartian/routerbench"
DATASET_FILE = "routerbench_0shot.pkl"  # 0-shot only; the 5-shot file is out of scope
# Hugging Face commit sha of the dataset repo. Pinning it means every later
# download fetches the exact same file. Taken from data/raw/DOWNLOAD_INFO.json
# after the first real download on 2026-11-17 (the pickle's SHA-256 is there).
DATASET_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # gitignored: the official pickle
CACHE_DIR = DATA_DIR / "cache"      # gitignored: full-data embeddings
SAMPLE_DIR = DATA_DIR / "sample"    # built by Avimta's data/make_sample.py

RAW_PICKLE = RAW_DIR / DATASET_FILE
DOWNLOAD_INFO = RAW_DIR / "DOWNLOAD_INFO.json"

SAMPLE_CSV = SAMPLE_DIR / "sample.csv"
SAMPLE_EMBEDDINGS = SAMPLE_DIR / "embeddings.npy"
SAMPLE_IDS = SAMPLE_DIR / "sample_ids.npy"

EMB_CACHE = CACHE_DIR / "emb_minilm.npy"
EMB_IDS = CACHE_DIR / "emb_ids.npy"

RESULTS_DIR = ROOT / "results"
RESULTS_SAMPLE = RESULTS_DIR / "sample"
RESULTS_FULL = RESULTS_DIR / "full"
RESULTS_INTERIM = RESULTS_FULL / "interim"      # preliminary results only
DATA_CHECK_DIR = RESULTS_FULL / "data_check"
# scripts/truncation_check.py. Kept out of data_check/, which holds only what
# scripts/data_check.py writes, so that folder can be rebuilt and compared
# file for file.
TRUNCATION_DIR = RESULTS_FULL / "truncation"

# --------------------------------------------------------------------------
# D1-D3, D11 and prompt format: data decisions (decided 2026-11-22 from the
# data check evidence in results/full/data_check/). src/data.py applies them.
# --------------------------------------------------------------------------
# D2: the cheap/strong pair. GPT-4 has the highest mean score; Mixtral is a
# low-cost model with a large gap (model_summary.csv, 0/1 benchmarks:
# 0.816 vs 0.552, Mixtral costs 4.1% of GPT-4), and it's the strong/weak
# pair RouteLLM used (Ong et al., arXiv:2406.18665). Those means are over the whole dataset, so the ranking must be
# re-checked on the training split only (PRD 0.4) once src/splits.py exists.
STRONG_MODEL = "gpt-4-1106-preview"
CHEAP_MODEL = "mistralai/mixtral-8x7b-chat"  # TODO(D2): confirm ranking on train split

# D3: each raw `eval_name` subtask maps to a parent benchmark ("group") by
# prefix, matched case-insensitively (e.g. "Chinese_character_riddles"); a
# subtask that matches no prefix is its own group. This is the mapping in
# candidate_group_counts.csv: 86 subtasks -> 13 groups. Groups drive
# stratification and held-out-benchmark splits; never a feature (PRD 0.4).
GROUP_PREFIXES = [
    ("mmlu-", "mmlu"),
    ("mtbench", "mtbench"),
    ("chinese", "chinese"),
]

# D1: subtasks dropped because their scores aren't 0/1, so "the cheap model
# is correct" is undefined for them (score_binary_by_eval.csv: every model
# has fractional scores on exactly these six). 7,907 of 36,497 rows. Losing
# GSM8K is flagged to the instructor; MBPP stays as the only free-form
# benchmark.
EXCLUDED_EVAL_NAMES = [
    "chinese_idioms",
    "consensus_summary",
    "grade-school-math",
    "mtbench",
    "mtbench-math",
    "mtbench-reference",
]

# D3: whole groups dropped. test-match has 3 rows: too few to split 70/15/15
# and too few to mean anything as a benchmark.
EXCLUDED_GROUPS = ["test-match"]

# D3: a group is held out on its own (leave-one-benchmark-out) only if it is
# big enough for the test metrics to mean something: at least 250 rows, and
# the cheap model both right and wrong at least 20 times, so ROC AUC and the
# routing curve aren't decided by a handful of prompts. Smaller groups stay
# in training for every split; they're just never the held-out test set.
HELDOUT_MIN_ROWS = 250
HELDOUT_MIN_CHEAP_RIGHT = 20
HELDOUT_MIN_CHEAP_WRONG = 20

# Prompt format. Prompts are stored as str(list_of_turns). We join the turns
# into one text, LAST turn first. In two-turn prompts the first turn is
# often a fixed instruction shared by every row of the benchmark
# (prompt_format.csv: 12 subtasks) and the last turn is the actual question,
# so putting the question first means MiniLM's 256-token truncation (D7)
# cuts the shared instruction, not the question. One rule for every
# feature set (TF-IDF, hand-built, embeddings).
PROMPT_TURN_ORDER = "last_first"   # or "in_order"
PROMPT_TURN_SEPARATOR = "\n\n"     # a blank line, so turns read as paragraphs

# D11: duplicates are found on the final joined text within a group, before
# splitting, so the same prompt can't land in both train and test. Keep the
# first copy (pickle order). If the copies' scores disagree for any model,
# drop every copy: keeping one would be an arbitrary choice of label.
DEDUPE_WITHIN_GROUP = True
DROP_CONFLICTING_DUPLICATES = True

# --------------------------------------------------------------------------
# D4: splits (PRD default, decided 2026-11-22)
# --------------------------------------------------------------------------
# Random setting: 70/15/15 train/val/test, stratified by group, redrawn per seed.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
# Leave-one-benchmark-out: the held-out group is the whole test set; the
# remaining groups are split 85/15 train/val, so validation never sees the
# held-out benchmark.
LOBO_TRAIN_FRAC = 0.85
LOBO_VAL_FRAC = 0.15

SETTINGS = ["random", "lobo"]

# --------------------------------------------------------------------------
# D5-D7: features
# --------------------------------------------------------------------------
FEATURE_SETS = ["tfidf", "handbuilt", "embedding"]

# D5 (PRD default, decided 2026-11-22). Word 1-2 grams catch short
# phrases like "which of"; min_df=2 drops one-off tokens that can't
# generalize; sublinear_tf damps very long prompts.
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "max_features": 50000,
}
# SVD size differs by profile (see SAMPLE / FULL below): the sample has too
# few rows to support 256 components.

# D6 (PRD default, decided 2026-11-22): the 13 hand-built features.
# Several of these reveal a benchmark's format (e.g. answer choices for MMLU,
# code keywords for MBPP); the report must say so.
HANDBUILT_FEATURES = [
    "n_chars",
    "n_words",
    "n_lines",
    "n_digits",
    "digit_ratio",
    "n_code_symbols",
    "code_symbol_ratio",
    "n_question_marks",
    "has_answer_choices",   # a line starting "A.", "A)" or "(A)"
    "n_math_symbols",
    "uppercase_ratio",
    "mean_word_len",
    "has_code_keyword",     # whole-word match on CODE_KEYWORDS
]
CODE_SYMBOLS = "{}[]();=<>_#"
MATH_SYMBOLS = "+-*/^%$"
CODE_KEYWORDS = ("def", "return", "import", "class", "for", "while")

# D7 (PRD default, decided 2026-11-22). Frozen embeddings only, no
# fine-tuning (scope). Normalized so cosine similarity is a dot product.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_NORMALIZE = True
# MiniLM reads at most 256 tokens (its max_seq_length) and drops the rest.
# scripts/truncation_check.py checks this against the downloaded model.
EMBEDDING_MAX_TOKENS = 256

# --------------------------------------------------------------------------
# D8: operating point (PRD default, decided 2026-11-22)
# --------------------------------------------------------------------------
# Cost needed to reach 95% of always-strong quality, read off the test curve
# and also at the threshold chosen on validation.
TARGET_QUALITY_FRAC = 0.95

# --------------------------------------------------------------------------
# D9: hyperparameter grids, selected by validation curve AUC (PRD default,
# decided 2026-11-22)
# --------------------------------------------------------------------------
# FULL grids for Piyush's routers are the PRD 7.6 starting values. Avimta
# fills in hgb / rf / mlp.
GRIDS_FULL = {
    "lr": {"C": [0.01, 0.1, 1, 10], "max_iter": [2000]},
    "knn": {"n_neighbors": [5, 15, 50, 100], "weights": ["uniform", "distance"]},
    "kmeans": {"n_clusters": [10, 25, 50, 100], "n_init": [10]},
    "hgb": None,  # Avimta
    "rf": None,   # Avimta
    "mlp": None,  # Avimta
}

# SAMPLE grids: one or two values per setting so `python main.py` stays well
# under 5 minutes. Proposed by Claude Code; Piyush to confirm.
GRIDS_SAMPLE = {
    "lr": {"C": [0.1, 1], "max_iter": [2000]},
    "knn": {"n_neighbors": [15, 50], "weights": ["distance"]},
    "kmeans": {"n_clusters": [10, 25], "n_init": [10]},
    "hgb": None,  # Avimta
    "rf": None,   # Avimta
    "mlp": None,  # Avimta
}

SELECTION_METRIC = "val_curve_auc"

# --------------------------------------------------------------------------
# Run profiles
# --------------------------------------------------------------------------
SAMPLE_TIME_LIMIT_S = 300  # the brief's 5-minute limit for `python main.py`

SAMPLE = {
    "name": "sample",
    "seeds": SEEDS[:1],
    "svd_components": 128,
    "grids": GRIDS_SAMPLE,
    "results_dir": RESULTS_SAMPLE,
}

FULL = {
    "name": "full",
    "seeds": SEEDS,
    "svd_components": 256,
    "grids": GRIDS_FULL,
    "results_dir": RESULTS_FULL,
}

# === Avimta: settings ===
# Embedding run (src/embed.py, PRD 8.2).
# Batch size 64 is the PRD's starting value: large enough to keep the CPU
# busy, small enough to stay well inside 8 GB RAM with MiniLM.
EMBED_BATCH_SIZE = 64
# Prompts per saved chunk. Each chunk is written to disk as soon as it is
# done, so an interrupted overnight run resumes from the last finished chunk
# instead of starting again.
EMBED_CHUNK_SIZE = 2000
EMB_CHUNK_DIR = CACHE_DIR / "emb_chunks"
# === end Avimta ===
