"""Row counts of the cleaned data, for the report's Data section.

    python scripts/data_summary.py

src/data.py prints these when it cleans the data; this saves them, so every
number the reports quote about the cleaned data traces to a file
(CLAUDE.md rule 5):

    results/full/data_summary/cleaning_steps.csv   one row per cleaning step (D1, D3, D11)
    results/full/data_summary/group_summary.csv    per benchmark group, then ALL: rows,
                                                   share of rows, cheap and strong model
                                                   accuracy, held-out eligibility (D3)

Descriptive only, on the whole cleaned set (no split); nothing is tuned on
it. Offline; needs only the pickle.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data  # noqa: E402


def main():
    df = data.load_data(verbose=False)
    steps = pd.DataFrame(df.attrs["steps"])
    groups = data.group_summary(df)

    out = config.DATA_SUMMARY_DIR
    out.mkdir(parents=True, exist_ok=True)
    steps.to_csv(out / "cleaning_steps.csv", index=False)
    # Four decimals: enough for any percentage the report quotes, and the
    # file stays byte-identical across reruns.
    groups.to_csv(out / "group_summary.csv", index=False, float_format="%.4f")

    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(steps.to_string(index=False))
        print()
        print(groups.to_string(index=False))
    print(f"\nWrote {out.relative_to(config.ROOT)}/cleaning_steps.csv and group_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
