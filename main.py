"""Single entry point for the project.

    python main.py          # SAMPLE profile on data/sample/, offline, < 5 min on CPU
    python main.py --full   # FULL profile: the whole experiment matrix -> results/full/

Development flags narrow a run (PRD 7.8), e.g.
    python main.py --full --routers lr --settings lobo --seeds 0

The full run resumes: finished cells are skipped, so rerunning after an
interruption, or after Avimta's routers are added, only runs what's missing.

The sample run is not wired yet: it waits for data/sample/ (Avimta's
data/make_sample.py). This file never uses the network (CLAUDE.md rule 9).
"""

import argparse
import sys

import config


def _csv(cast=str):
    return lambda s: [cast(x) for x in s.split(",") if x]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RouterBench router experiments")
    parser.add_argument("--full", action="store_true",
                        help="run the FULL profile on data/raw/ instead of the sample")
    parser.add_argument("--routers", type=_csv(), default=None,
                        help="comma-separated routers (default: every registered router)")
    parser.add_argument("--settings", type=_csv(), default=None,
                        help="comma-separated settings: random, lobo (default: both)")
    parser.add_argument("--seeds", type=_csv(int), default=None,
                        help="comma-separated seeds (default: the profile's seeds)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    profile = config.FULL if args.full else config.SAMPLE
    print(f"Profile: {profile['name']}")
    print(f"  seeds:          {args.seeds or profile['seeds']}")
    print(f"  svd_components: {profile['svd_components']}")
    print(f"  results_dir:    {profile['results_dir'].relative_to(config.ROOT)}")

    if not args.full:
        print("Sample run not wired yet: it waits for data/sample/ "
              "(Avimta's data/make_sample.py). Exiting.")
        return 0

    if not config.RAW_PICKLE.exists():
        print(f"{config.RAW_PICKLE.relative_to(config.ROOT)} is missing. Download it with:\n"
              "    python data/download.py", file=sys.stderr)
        return 1

    # Imported here so the sample path (and --help) stays fast and light.
    from src import data, experiment

    df = data.load_data(verbose=True)
    counts = experiment.run_all(df, profile, profile["results_dir"], router_names=args.routers,
                                settings=args.settings, seeds=args.seeds)
    print(f"Done: {counts['ran']} cells run, {counts['skipped']} already finished, "
          f"{counts['total']} in total. Results in "
          f"{profile['results_dir'].relative_to(config.ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
