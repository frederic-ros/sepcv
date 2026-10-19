"""Small controlled generator: rarity and separability crossed orthogonally.

Used by the smoke test and by the controlled factorial check. Both classes have
identity covariance; ``sep`` scales the rare-class mean structure, so that rarity
(``ir``) and separability (``sep``) are crossed orthogonally. Geometries:
``gauss`` (one shifted mode, linear), ``multimodal`` (three pockets around the
majority, rare mean equal to the majority mean, invisible to a linear score) and
``moon`` (half circle, partly linear). A random rotation removes any privileged axis.
"""
from __future__ import annotations

import itertools

import numpy as np


def synthetic_specs(src):
    specs = []
    grid = itertools.product(src.get("geometries", ["gauss"]), src.get("irs", [10, 50]),
                             src.get("seps", [0.5, 1.5, 3.0]), src.get("dims", [10]))
    for geo, ir, sep, d in grid:
        name = f"{geo}_ir{ir}_sep{sep}_d{d}"
        specs.append(dict(kind="synthetic", family=src.get("family", "synthetic"),
                          task_id=f"synthetic|{name}", dataset=name, group=name, domain="synthetic",
                          geometry=geo, ir=ir, sep=sep, d=d, n=src.get("n", 3000),
                          ir_nominal=ir))
    return specs


def _rotation(d, rng):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def generate(spec, seed):
    rng = np.random.default_rng(seed)
    n, ir, sep, d = spec["n"], spec["ir"], spec["sep"], spec["d"]
    n_pos = max(20, int(round(n / (1 + ir))))
    n_neg = int(round(n_pos * ir))
    X0 = rng.normal(size=(n_neg, d))
    geo = spec["geometry"]
    # Both classes have identity covariance; only the rare-class mean structure,
    # scaled by `sep`, differs. sep -> 0 gives identical class distributions in
    # every geometry, so `sep` is the single separability lever.
    X1 = rng.normal(size=(n_pos, d))
    if geo == "gauss":          # one shifted mode: linearly separable
        X1[:, 0] += sep
    elif geo == "multimodal":   # three pockets around the majority: rare mean = majority mean,
        ang = 2 * np.pi * rng.integers(0, 3, n_pos) / 3   # invisible to a linear score
        X1[:, 0] += sep * np.cos(ang)
        X1[:, 1] += sep * np.sin(ang)
    elif geo == "moon":         # rare class on a half circle: partly linear
        t = rng.uniform(0, np.pi, n_pos)
        X1[:, 0] += sep * np.cos(t)
        X1[:, 1] += sep * np.sin(t)
    else:
        raise ValueError(geo)
    X = np.vstack([X0, X1]) @ _rotation(d, rng)
    y = np.r_[np.zeros(n_neg, int), np.ones(n_pos, int)]
    return X, y
