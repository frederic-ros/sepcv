"""Acquisition strategies used as robustness factors (not compared to each other).

* ``random``               -- uniform; labels do not depend on the learner.
* ``margin``               -- smallest |2p - 1| (reference rule), random tie-break.
* ``most_likely_positive`` -- greedy active search (highest p), the rule fraud
                              investigators effectively apply; rare-oriented.
* ``coreset``              -- greedy k-center in feature space; geometry only, so
                              labels do not depend on the learner.

For binary problems entropy and least-confidence induce the same ranking as
margin, so they are not separate factors here.
"""
from __future__ import annotations

import numpy as np

LEARNER_FREE = {"random", "coreset"}
STRATEGIES = ("random", "margin", "most_likely_positive", "coreset")


def select(name, p_pool, X_pool, X_lab, batch, rng):
    n = len(X_pool)
    batch = min(batch, n)
    jitter = rng.random(n)
    if name == "random":
        return rng.choice(n, batch, replace=False)
    if name == "margin":
        return np.lexsort((jitter, np.abs(2.0 * p_pool - 1.0)))[:batch]
    if name == "most_likely_positive":
        return np.lexsort((jitter, -p_pool))[:batch]
    if name == "coreset":
        d = _min_dist(X_pool, X_lab)
        picked = []
        for _ in range(batch):
            i = int(np.argmax(d + 1e-12 * jitter))
            picked.append(i)
            d = np.minimum(d, np.linalg.norm(X_pool - X_pool[i], axis=1))
            d[i] = -1.0
        return np.array(picked)
    raise ValueError(f"unknown strategy '{name}'")


def _min_dist(A, B, chunk=2048):
    out = np.full(len(A), np.inf)
    b2 = (B ** 2).sum(1)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        d2 = (a ** 2).sum(1)[:, None] + b2[None, :] - 2 * a @ B.T
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
    return out
