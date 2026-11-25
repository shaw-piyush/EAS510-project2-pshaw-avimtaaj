"""Router registry: name -> class.

Each router lives in its own file, so the two of us never edit the same
module. To add one, add its import and one line in ROUTERS (Avimta: one
import and one registry line per router, nothing else; PRD 5.4).
"""

from src.routers.hgb import HGBRouter
from src.routers.kmeans import KMeansRouter
from src.routers.knn import KNNRouter
from src.routers.lr import LRRouter
from src.routers.rf import RFRouter

ROUTERS = {
    "lr": LRRouter,
    "knn": KNNRouter,
    "kmeans": KMeansRouter,
    "hgb": HGBRouter,
    "rf": RFRouter,
}


def make(name, seed, **params):
    if name not in ROUTERS:
        raise ValueError(f"unknown router {name!r}; registered: {sorted(ROUTERS)}")
    return ROUTERS[name](seed, **params)
