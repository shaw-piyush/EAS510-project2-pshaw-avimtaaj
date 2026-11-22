"""How much of each prompt fits in MiniLM's 256-token window, per benchmark.

    python scripts/truncation_check.py

Evidence for the prompt format decision (config.PROMPT_TURN_ORDER). Runs on
the rows src/data.py keeps (D1, D3 and D11 applied) and compares the two
turn orders. The total token count is the same either way; what differs is
WHICH turn the truncation cuts:

- in_order:   instruction first, question last, so the question is cut;
- last_first: question first, instruction last, so the instruction is cut.

Writes results/full/truncation/truncation_by_benchmark.csv (counts only, no
prompt text) and truncation_info.json (model snapshot, window, versions).

Needs the embedding packages (embed_pins_tested.txt) and the MiniLM model in
the local Hugging Face cache. It never downloads: if the model isn't cached,
run `python -m src.embed --limit 5` once (PRD 0.10 lets src/embed.py
download), then rerun this.
"""

import json
import os
import sys
from pathlib import Path

# Offline before any Hugging Face import: this script must not download.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data  # noqa: E402

OUT_CSV = config.TRUNCATION_DIR / "truncation_by_benchmark.csv"
OUT_INFO = config.TRUNCATION_DIR / "truncation_info.json"


def load_tokenizer():
    try:
        import huggingface_hub
        import transformers
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("Needs transformers: pip install -r embed_pins_tested.txt (see the setup guide).")
    try:
        snap = Path(huggingface_hub.snapshot_download(config.EMBEDDING_MODEL, local_files_only=True))
    except Exception:
        sys.exit(f"{config.EMBEDDING_MODEL} is not in the local cache. "
                 "Run `python -m src.embed --limit 5` once, then rerun this.")
    # The window the model really uses (sentence-transformers' max_seq_length),
    # checked against config so the table and the embeddings agree.
    window = json.loads((snap / "sentence_bert_config.json").read_text())["max_seq_length"]
    if window != config.EMBEDDING_MAX_TOKENS:
        sys.exit(f"Model window is {window}, config.EMBEDDING_MAX_TOKENS is "
                 f"{config.EMBEDDING_MAX_TOKENS}. Fix config first.")
    tok = AutoTokenizer.from_pretrained(snap)
    versions = {"transformers": transformers.__version__,
                "huggingface_hub": huggingface_hub.__version__}
    return tok, window, snap.name, versions


def n_tokens(tok, texts, special):
    ids = tok(list(texts), add_special_tokens=special, truncation=False, verbose=False)["input_ids"]
    return np.array([len(i) for i in ids])


def main():
    tok, window, revision, versions = load_tokenizer()
    df = data.load_data(verbose=False)
    turns = [data.parse_turns(p) for p in df["prompt"]]
    two = np.array([len(t) == 2 for t in turns])
    if any(len(t) > 2 for t in turns):
        sys.exit("A prompt has more than two turns; this table assumes one or two.")

    # Exact totals: tokenize the real joined text, special tokens included,
    # exactly as the model sees it.
    total = {o: n_tokens(tok, (data.join_turns(t, o) for t in turns), True)
             for o in ("in_order", "last_first")}
    # Per-turn lengths, to say which turn gets cut. For WordPiece the joined
    # text's length is the turns' lengths plus the 2 special tokens (the
    # blank-line separator is only whitespace); checked here, not assumed.
    first = n_tokens(tok, (t[0] for t in turns), False)
    last = n_tokens(tok, (t[-1] for t in turns), False)
    pieces = np.where(two, first + last, last) + 2
    mismatch = int(sum((total[o] != pieces).sum() for o in total))
    if mismatch:
        sys.exit(f"{mismatch} rows where joined length != sum of turns; per-turn columns would be wrong.")

    room = window - 2  # tokens left for text after [CLS] and [SEP]
    over = total["last_first"] > window
    f = pd.DataFrame({
        "group": df["group"].to_numpy(),
        "two_turn": two,
        "tokens": total["last_first"],
        "over_window": over,
        # in_order: the question (last turn) is cut whenever the row is too
        # long, and lost entirely if the instruction alone fills the window.
        "inorder_question_cut": over,
        "inorder_question_lost": two & (first >= room),
        "inorder_instruction_cut": two & (first > room),
        # last_first: the question is cut only if it alone is too long; the
        # instruction is cut whenever a two-turn row is too long.
        "lastfirst_question_cut": last > room,
        "lastfirst_instruction_cut": two & over,
        "lastfirst_instruction_lost": two & (last >= room),
    })

    def summarise(name, g):
        row = {"group": name, "n_rows": len(g), "n_two_turn": int(g["two_turn"].sum()),
               "median_tokens": int(np.median(g["tokens"])),
               "n_over_window": int(g["over_window"].sum()),
               "share_over_window": round(g["over_window"].mean(), 4)}
        for c in ["inorder_question_cut", "inorder_question_lost", "inorder_instruction_cut",
                  "lastfirst_question_cut", "lastfirst_instruction_cut",
                  "lastfirst_instruction_lost"]:
            row["n_" + c] = int(g[c].sum())
        return row

    # Built from plain dicts so the counts stay integers in the CSV.
    rows = [summarise(name, g) for name, g in f.groupby("group")]
    rows.sort(key=lambda r: (-r["n_rows"], r["group"]))
    table = pd.DataFrame(rows + [summarise("ALL", f)])

    config.TRUNCATION_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    info = {
        "model": config.EMBEDDING_MODEL,
        "model_snapshot": revision,
        "window_tokens": window,
        "rows": "src/data.py load_data(): D1, D3 and D11 applied",
        "n_rows": len(df),
        "separator": config.PROMPT_TURN_SEPARATOR,
        "joined_length_equals_sum_of_turns": mismatch == 0,
        **versions,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2) + "\n")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string(index=False))
    print(f"\nWrote {OUT_CSV.relative_to(config.ROOT)} and {OUT_INFO.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
