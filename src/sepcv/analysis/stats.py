"""Small statistics helpers: rank correlation, LOO/LOGO R^2, cluster bootstrap."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(spearmanr(x, y)[0])


def _ols_fit(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _ols_pred(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def r2(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else np.nan


def logo_predict(X, y, groups):
    """Leave-one-group-out OLS predictions (groups = datasets or source groups)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        if (~te).sum() < X.shape[1] + 2:
            continue
        pred[te] = _ols_pred(_ols_fit(X[~te], y[~te]), X[te])
    return pred


def cluster_bootstrap(groups, stat_fn, B=2000, seed=0):
    """Percentile CI of stat_fn(index array) resampling whole groups."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return np.nan, np.nan
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        v = stat_fn(idx)
        if v == v:
            vals.append(v)
    if len(vals) < B * 0.5:
        return np.nan, np.nan
    return tuple(np.percentile(vals, [2.5, 97.5]))
