from __future__ import annotations

import numpy as np

from src.constants import PAT_CODES, THREATS, THREAT_PAT_MAP


def node_list():
    return THREATS + PAT_CODES + ["ACA", "PH"]


def predefined_adjacency() -> np.ndarray:
    nodes = node_list()
    idx = {n: i for i, n in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for t, pats in THREAT_PAT_MAP.items():
        for p in pats:
            i, j = idx[t], idx[p]
            a[i, j] = 1.0
            a[j, i] = 1.0
    return a
