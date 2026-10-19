"""Dispersion of early estimators on a fixed-separability problem (appendix figure).

For n+ discovered positives and 30 negatives drawn from a fixed binormal problem,
compare the standard deviation (over resampled labeled sets) of:
  CV-AUROC (SepCV), CV-AP, held-out AUROC, held-out AP (single 2/3-1/3 split).
metric effect     = sd(CV-AP) / sd(CV-AUROC)
aggregation effect = sd(held-out AUROC) / sd(CV-AUROC)
combined          = sd(held-out AP) / sd(CV-AUROC)
Usage: python scripts/fig_variance.py [out.pdf]
"""
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, "src")
from sepcv.learners import fit, pos_proba  # noqa: E402


def one(npos, rng, d=8, delta=1.5, nneg=30):
    X = np.vstack([rng.normal(size=(nneg, d)), rng.normal(size=(npos, d)) + np.r_[delta, np.zeros(d - 1)]])
    y = np.r_[np.zeros(nneg, int), np.ones(npos, int)]
    seed = int(rng.integers(2**31))
    au, ap = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=min(3, npos), n_repeats=5, random_state=seed).split(X, y):
        s = pos_proba(fit("logreg", seed, X[tr], y[tr])[0], X[te])
        au.append(roc_auc_score(y[te], s)); ap.append(average_precision_score(y[te], s))
    out = [np.mean(au), np.mean(ap)]
    if npos >= 4:
        tr, te = next(StratifiedShuffleSplit(1, test_size=1 / 3, random_state=seed).split(X, y))
        s = pos_proba(fit("logreg", seed, X[tr], y[tr])[0], X[te])
        out += [roc_auc_score(y[te], s), average_precision_score(y[te], s)]
    else:
        out += [np.nan, np.nan]
    return out


def main(out="fig_variance_decomposition.pdf", reps=400):
    rng = np.random.default_rng(0)
    rows = []
    for npos in (3, 4, 5, 6, 8, 12, 20):
        v = np.array([one(npos, rng) for _ in range(reps)])
        sd = np.nanstd(v, axis=0)
        rows.append((npos, sd[0], sd[1] / sd[0], sd[2] / sd[0], sd[3] / sd[0]))
        print(f"n+={npos:2d}  sd(CV-AUROC)={sd[0]:.3f}  metric={sd[1]/sd[0]:.2f}x  "
              f"aggregation={sd[2]/sd[0]:.2f}x  combined={sd[3]/sd[0]:.2f}x")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = np.array(rows)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.plot(r[:, 0], r[:, 2], marker="o", label="metric (CV-AP / CV-AUROC)")
        ax.plot(r[:, 0], r[:, 3], marker="s", label="aggregation (held-out / CV, AUROC)")
        ax.plot(r[:, 0], r[:, 4], marker="^", label="combined (held-out AP / CV-AUROC)")
        ax.axhline(1, color="grey", lw=0.6)
        ax.set_xlabel("discovered rare examples $n_+$")
        ax.set_ylabel("dispersion relative to CV-AUROC")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out)
        print("written", out)
    except Exception as e:                                     # noqa: BLE001
        print("no figure:", e)


if __name__ == "__main__":
    main(*sys.argv[1:])
