# Features or Models? LLM Routers on RouterBench

EAS 510 Project 2 (Fall 2026) · Piyush Ranjan Shaw and Avimta Ajit Shahane

## What the project does

An LLM router decides, prompt by prompt, whether a cheap model is good enough or the prompt should go to a strong, expensive one. We route between **Mixtral-8x7B** (cheap) and **GPT-4** (strong) on [RouterBench](https://huggingface.co/datasets/withmartian/routerbench) (0-shot). RouterBench stores every model's scored answer to every prompt, so a router can be evaluated offline: no LLM is ever called.

The question is **which matters more, the features or the model**. We cross three ways of representing a prompt:

- TF-IDF with SVD;
- 13 hand-built text features;
- frozen MiniLM sentence embeddings (no fine-tuning).

with these routers:

- a logistic regression baseline;
- kNN;
- k-means cluster routing;
- gradient boosting, with a random forest check;
- an MLP.

Every combination is tested on a random split and on leave-one-benchmark-out splits, which check whether a router generalizes to a benchmark it never saw.

Routers are compared by:

- the area under the quality-vs-cost curve and its lift over random routing;
- the cost needed to reach 95% of GPT-4's quality, at a threshold chosen on validation.

An extension routes among more than two models.

## Where we are (Mon Nov 23, 2026)

Interim report: [`docs/interim_report.md`](docs/interim_report.md). Final deadline: Mon Dec 7.

**Done**

- **Data:** the RouterBench pickle is downloaded at a pinned revision and cleaned from 36,497 to 28,572 prompts in 9 benchmark groups. Subtasks whose scores aren't 0/1 are dropped, including GSM8K, and so are duplicate prompts. Counts per step and per benchmark are in [`results/full/data_summary/`](results/full/data_summary/).
- **Splits:** random 70/15/15 stratified by benchmark, redrawn for 5 seeds, plus leave-one-benchmark-out for benchmarks large enough to test on.
- **Features:** TF-IDF + SVD (256 dimensions), 13 hand-built features, and MiniLM embeddings for every prompt. The embeddings are cached locally and are not in the repo.
- **Evaluation:** routing curves, AUC, lift over random, cost@95%, classifier metrics, reference policies (always-cheap, always-strong, random, oracle) and plots.
- **Baseline:** logistic regression on TF-IDF and on the hand-built features, run end to end on the full data with 5 seeds.
- 85 unit tests, which need no data download.

**Preliminary results** (random split, test set, mean ± std over 5 seeds; source: [`results/full/interim/summary.md`](results/full/interim/summary.md)):

| Router | Test AUC | Lift over random | Cost@95%, test curve (% of GPT-4) | Cost at validation threshold (% of GPT-4) |
|---|---|---|---|---|
| LR, TF-IDF | 0.6961 ± 0.0042 | 0.0135 ± 0.0036 | 76.9 ± 1.4 | 76.6 ± 1.3 |
| LR, hand-built | 0.6933 ± 0.0037 | 0.0107 ± 0.0044 | 81.0 ± 1.9 | 79.6 ± 1.8 |
| Random routing | 0.6826 ± 0.0041 | 0 | — | — |

The baseline beats random routing, but only by a small margin. Embeddings, the other routers and leave-one-benchmark-out have not been run yet.

**Next**

- kNN and k-means cluster routing (Piyush); gradient boosting, the random forest check and the MLP (Avimta).
- The full comparison on both split types, per-benchmark error analysis, and the multi-model extension.
- The bundled sample and a working `python main.py`, then the final `report.md`.

**Known gaps**

- `python main.py` is still a stub: it prints the chosen settings and exits.
- `data/sample/` is empty. We plan to bundle a small sample for the offline `main.py` run; RouterBench lists no license, so it is unclear whether we may redistribute part of it.
- `requirements.txt` doesn't list the embedding packages yet (see [Embeddings](#embeddings)).

## Setup

Tested with Python 3.14.6 on macOS 26.6.2 (Apple silicon), CPU only.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py           # stub for now; never uses the network
python -m pytest         # 85 tests, no data needed
```

## Reproducing the full-data results

```bash
python data/download.py              # RouterBench 0-shot pickle (~100 MB) into data/raw/
python scripts/data_check.py         # raw schema and score checks -> results/full/data_check/
python scripts/data_summary.py       # cleaning counts per step and benchmark -> results/full/data_summary/
python scripts/d2_check.py           # confirms GPT-4/Mixtral on training splits -> results/full/d2_check/
python scripts/interim_baseline.py   # preliminary LR baseline -> results/full/interim/
```

`data/download.py` is pinned to dataset revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`. It records the file's SHA-256 and retrieval time in `data/raw/DOWNLOAD_INFO.json`. Only `data/download.py` and `src/embed.py` use the network; everything else runs offline. The interim baseline took 33 s with 1,010 MB peak memory (`results/full/interim/run_info.json`).

### Embeddings

The embedding script and the truncation check need three packages that aren't in `requirements.txt` yet. These are the versions we used:

```bash
python -m pip install sentence-transformers==6.3.0 torch==2.15.0 transformers==5.23.0
python -m src.embed --limit 50       # quick test on the first 50 prompts
python -m src.embed                  # all prompts -> data/cache/ (resumable; downloads MiniLM on first use)
python scripts/truncation_check.py   # prompts longer than MiniLM's 256-token window -> results/full/truncation/
```

## Repository layout

```
config.py            every seed, path, data decision and hyperparameter
main.py              single entry point (sample run; full run with --full)
data/download.py     fetches the pinned RouterBench pickle
src/data.py          loading, cleaning, labels (cheap model right or wrong)
src/splits.py        random and leave-one-benchmark-out splits
src/features.py      TF-IDF + SVD, hand-built features, embedding lookup
src/embed.py         frozen MiniLM embeddings, cached to data/cache/
src/routers/         router interface and the logistic regression baseline
src/experiment.py    one run: tune on validation, evaluate on test once
src/metrics.py       routing curves, AUC, cost@95%, classifier metrics
src/policies.py      always-cheap, always-strong, random and oracle
src/plots.py         figures
scripts/             data checks and the interim baseline
results/full/        full-data outputs (results/sample/ for main.py)
tests/               unit tests
docs/                interim report
```

## Data

RouterBench (`withmartian/routerbench`, file `routerbench_0shot.pkl`). The dataset page lists no license; the RouterBench code repository is MIT. We don't redistribute the data: `data/raw/` and `data/cache/` are git-ignored, and `data/download.py` fetches the file from Hugging Face.
