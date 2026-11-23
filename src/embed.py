"""Frozen sentence embeddings for every prompt, cached to data/cache/ (PRD 8.2).

Usage:
    python -m src.embed --limit 50     # quick test on the first 50 prompts
    python -m src.embed                # full run (resumable; rerun if interrupted)

Writes (full run):
    data/cache/emb_minilm.npy   float32, one row per prompt, L2-normalized
    data/cache/emb_ids.npy      the sample_id of each row, same order

Design choices:
- Frozen model only (scope: no fine-tuning). sentence-transformers/all-MiniLM-L6-v2
  with normalize_embeddings=True (D7), so cosine similarity is a plain dot
  product for kNN.
- sentence_transformers is imported INSIDE the functions that need it, never
  at module top, so `python main.py` (which must not use the network and
  must not need this package) never loads it.
- The run is saved in chunks (config.EMBED_CHUNK_SIZE). If it is interrupted,
  rerunning the same command skips finished chunks. A manifest records the
  model and the flattening version, so chunks made with different settings
  are never mixed together.
- This is one of the three entry points allowed to use the network
  (PRD 0.10): it downloads the model from Hugging Face on first use.

Prompt flattening (agreed 2026-11-22):
Prompts are stored as Python list literals in a string, e.g. "['turn 1', 'turn 2']".
We call src.data.flatten_prompt, the same function every other feature set
uses, so TF-IDF, hand-built features and embeddings all read identical text.
It puts the LAST turn first (config.PROMPT_TURN_ORDER): in two-turn prompts
the first turn is often an instruction shared by the whole benchmark, and
MiniLM only reads the first config.EMBEDDING_MAX_TOKENS tokens, so the
question must come first to survive truncation.

Which rows:
Every row of the raw pickle is embedded, not only the rows load_data() keeps
after D1/D3/D11. Consumers select rows by sample_id via emb_ids.npy. This
costs some extra CPU time once, but if a data decision changes later the
overnight run does not have to be redone.
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

import config
from src.data import ID_COL, PROMPT_COL, flatten_prompt

# The full run is allowed once flattening is agreed (it was, on 2026-11-22,
# and src.data.flatten_prompt implements it). The version string goes into
# the chunk manifest, so changing the turn order or separator in config.py
# makes old cached chunks refuse to mix with new ones.
FLATTENING_AGREED = True
FLATTEN_VERSION = (
    f"data.flatten_prompt/{config.PROMPT_TURN_ORDER}/"
    f"{config.PROMPT_TURN_SEPARATOR!r}"
)


def load_prompts(limit=None):
    """Return (ids, texts) in the pickle's row order.

    Only the id and prompt columns are used; no score, cost or eval_name
    column is touched, so nothing here can leak labels into features.
    """
    if not config.RAW_PICKLE.exists():
        sys.exit(f"{config.RAW_PICKLE} not found. Run `python data/download.py` first.")
    df = pd.read_pickle(config.RAW_PICKLE)[[ID_COL, PROMPT_COL]]
    if limit is not None:
        df = df.head(limit)
    ids = df[ID_COL].astype(str).to_numpy()
    texts = [flatten_prompt(p) for p in df[PROMPT_COL]]
    return ids, texts


def load_model():
    # Imported here, not at the top: see the module docstring.
    from sentence_transformers import SentenceTransformer

    # device="cpu": the brief targets CPU-only machines, and fixing the
    # device keeps runs comparable between Avimta's and Piyush's laptops.
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    # Set the truncation length explicitly from config (it is MiniLM's
    # default, 256) so the window Piyush's truncation check measured is the
    # one actually used, even if a library update changes the default.
    model.max_seq_length = config.EMBEDDING_MAX_TOKENS
    return model


def encode(model, texts):
    emb = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=config.EMBEDDING_NORMALIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def check_manifest(chunk_dir, n_prompts):
    """Refuse to resume from chunks made with different settings."""
    manifest = {
        "model": config.EMBEDDING_MODEL,
        "normalize": config.EMBEDDING_NORMALIZE,
        "flatten_version": FLATTEN_VERSION,
        "max_tokens": config.EMBEDDING_MAX_TOKENS,
        "chunk_size": config.EMBED_CHUNK_SIZE,
        "n_prompts": n_prompts,
    }
    path = chunk_dir / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            sys.exit(
                f"{chunk_dir} holds chunks made with different settings:\n"
                f"  old: {old}\n  new: {manifest}\n"
                "Move or delete that folder yourself, then rerun."
            )
    else:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def run(limit=None):
    if limit is None and not FLATTENING_AGREED:
        sys.exit(
            "Full run refused: prompt flattening is still PROVISIONAL.\n"
            "Agree the rule with Piyush, update flatten_prompt (or call src/data.py),\n"
            "set FLATTENING_AGREED = True, then rerun. For a test, use --limit N."
        )

    # A --limit test writes to its own folder so it can never be mistaken
    # for (or mixed into) the real cache.
    if limit is None:
        chunk_dir = config.EMB_CHUNK_DIR
        out_emb, out_ids = config.EMB_CACHE, config.EMB_IDS
    else:
        test_dir = config.CACHE_DIR / f"test_limit{limit}"
        chunk_dir = test_dir / "chunks"
        out_emb, out_ids = test_dir / "emb_minilm.npy", test_dir / "emb_ids.npy"

    t0 = time.perf_counter()
    ids, texts = load_prompts(limit)
    n = len(ids)
    print(f"Prompts: {n:,} (flattening: {FLATTEN_VERSION})")
    check_manifest(chunk_dir, n)

    model = None  # loaded lazily, so a fully cached rerun needs no download
    size = config.EMBED_CHUNK_SIZE
    starts = range(0, n, size)
    for i, start in enumerate(starts):
        path = chunk_dir / f"chunk_{i:05d}.npy"
        if path.exists():
            print(f"chunk {i + 1}/{len(starts)}: already done, skipping")
            continue
        if model is None:
            model = load_model()
        emb = encode(model, texts[start:start + size])
        # Write to a temp name, then rename: an interrupted write never
        # leaves a half-written chunk that a resume would trust.
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, emb)
        tmp.replace(path)
        print(f"chunk {i + 1}/{len(starts)}: saved {path.name}")

    emb = np.concatenate([np.load(chunk_dir / f"chunk_{i:05d}.npy") for i in range(len(starts))])
    if emb.shape[0] != n:
        sys.exit(f"Row count mismatch: {emb.shape[0]} embeddings for {n} prompts")
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, emb)
    np.save(out_ids, ids)

    elapsed = time.perf_counter() - t0
    print(f"Embeddings: {emb.shape} {emb.dtype}")
    print(f"Time: {elapsed:.1f} s")
    print(f"Wrote {out_emb.relative_to(config.ROOT)} and {out_ids.relative_to(config.ROOT)}")
    return emb, ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="embed only the first N prompts (test run)")
    args = parser.parse_args(argv)
    run(args.limit)


if __name__ == "__main__":
    main()
