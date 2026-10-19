"""Early estimators computed on the current labeled set L_t only.

SepCV (the proposed diagnostic): repeated stratified k-fold (k = min(3, n+))
rare-vs-rest discrimination of a class-weighted base learner, reported as
cross-validated AUROC (the declared estimator) and AP (variant). Undefined
(NaN) below ``min_positives`` discovered rare examples; the status is kept.

Comparators observable at the same probe: held-out AP (one stratified split),
labeled rare fraction, and two data-complexity measures (Fisher F1, N3).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

from sepcv.learners import fit, pos_proba


def sepcv(X, y, base, ecfg, seed):
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    out = {f"sepcv_{base}_{m}": np.nan for m in ecfg["metrics"]}
    if npos < ecfg["min_positives"] or nneg < ecfg["min_positives"]:
        return out
    k = min(ecfg["cv_folds_max"], npos)
    rep = ecfg["cv_repeats"].get(base, 3)
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=rep, random_state=seed % (2**31 - 1))
    acc = {m: [] for m in ecfg["metrics"]}
    for tr, te in cv.split(X, y):
        model, _ = fit(base, seed, X[tr], y[tr], rf_trees=ecfg["rf_trees"])
        s = pos_proba(model, X[te])
        if "auroc" in acc:
            acc["auroc"].append(roc_auc_score(y[te], s))
        if "ap" in acc:
            acc["ap"].append(average_precision_score(y[te], s))
    for m, v in acc.items():
        out[f"sepcv_{base}_{m}"] = float(np.mean(v))
    return out


def holdout_ap(X, y, ecfg, seed):
    if y.sum() < ecfg["min_positives"] + 1 or (len(y) - y.sum()) < ecfg["min_positives"] + 1:
        return np.nan
    tr, te = next(StratifiedShuffleSplit(1, test_size=1 / 3, random_state=seed % (2**31 - 1)).split(X, y))
    model, _ = fit("logreg", seed, X[tr], y[tr])
    return float(average_precision_score(y[te], pos_proba(model, X[te])))


def complexity(X, y, ecfg):
    """Fisher F1 (higher = more separable) and N3 / N3-rare (1-NN error; higher = harder)."""
    out = {"fisher_f1": np.nan, "n3": np.nan, "n3_rare": np.nan}
    if y.sum() < ecfg["min_positives"] or (len(y) - y.sum()) < ecfg["min_positives"]:
        return out
    X1, X0 = X[y == 1], X[y == 0]
    num = (X1.mean(0) - X0.mean(0)) ** 2
    den = X1.var(0) + X0.var(0) + 1e-12
    out["fisher_f1"] = float(np.max(num / den))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1]
    err = y[idx] != y
    out["n3"] = float(err.mean())
    out["n3_rare"] = float(err[y == 1].mean())
    return out


def probe(X, y, ecfg, seed):
    row = {"n_lab": int(len(y)), "n_pos": int(y.sum()), "rare_frac": float(y.mean())}
    for b in ecfg["bases"]:
        row.update(sepcv(X, y, b, ecfg, seed))
    if ecfg.get("holdout", True):
        row["holdout_ap"] = holdout_ap(X, y, ecfg, seed)
    if ecfg.get("complexity", True):
        row.update(complexity(X, y, ecfg))
    return row


def reference(Xtr, ytr, base, seed, max_n=5000):
    """Full-data separability S*: 3-fold CV-AUROC on the full training pool."""
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed)
    if len(ytr) > max_n:                                 # IR-preserving cap for cost
        pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
        k = max_n / len(ytr)
        idx = np.r_[rng.choice(pos, max(3, int(len(pos) * k)), replace=False),
                    rng.choice(neg, int(len(neg) * k), replace=False)]
        Xtr, ytr = Xtr[idx], ytr[idx]
    s = np.zeros(len(ytr))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed % (2**31 - 1)).split(Xtr, ytr):
        model, _ = fit(base, seed, Xtr[tr], ytr[tr], rf_trees=100)
        s[te] = pos_proba(model, Xtr[te])
    return float(roc_auc_score(ytr, s))
