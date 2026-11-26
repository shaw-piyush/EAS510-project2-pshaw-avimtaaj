"""What a run records about itself: code version, machine, memory.

Kept separate so every script and the experiment runner describe their runs
the same way (PRD 7.10: run_info.json).
"""

import platform
import resource
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn

import config

# Code that decides results. A run with uncommitted changes here can't be
# traced to a commit, so it is marked "-dirty".
CODE_PATHS = ["src", "config.py", "scripts", "main.py"]


def git_commit():
    """HEAD's sha, with "-dirty" if result-relevant code has uncommitted
    changes; None outside a git checkout."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=config.ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_PATHS],
                               capture_output=True, text=True, cwd=config.ROOT,
                               check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


def peak_memory_mb():
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (2**20 if sys.platform == "darwin" else 2**10), 1)


def versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
    }
