"""Router registry: name -> class.

Each router lives in its own file, so the two of us never edit the same
module. To add one, add its import and one line in ROUTERS (Avimta: one
import and one registry line per router, nothing else; PRD 5.4).
"""

from src.routers.lr import LRRouter

ROUTERS = {
    "lr": LRRouter,
}


def make(name, seed, **params):
    if name not in ROUTERS:
        raise ValueError(f"unknown router {name!r}; registered: {sorted(ROUTERS)}")
    return ROUTERS[name](seed, **params)
