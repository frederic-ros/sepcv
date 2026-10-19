"""What SepCV+ adds, how sensitive it is to its one constant, and where it fails.

* ``sensitivity`` -- task-level Spearman with the final AUPRC for the components of
  SepCV+: LR base only vs mean of the LR, RF and k-NN bases, crossed with the
  shrinkage constant k (k = 0 means no shrinkage). k = 3 is the a-priori choice.
* ``failures`` -- tasks where the diagnostic misleads: largest gaps between the
  rank of SepCV+ and the rank of the final AUPRC, largest disagreement between the
  linear and the non-linear bases, and tasks where the diagnostic is undefined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sepcv.analysis.load import PRIMARY, sepcv_plus
from sepcv.analysis.stats import rho

KS = (0, 1, 3, 6, 10)
VARIANTS = {"LR": ("sepcv_logreg_auroc",),
            "mean3": ("sepcv_logreg_auroc", "sepcv_rf_auroc", "sepcv_knn_auroc")}   # mean of LR, RF, k-NN


def sensitivity(runs, round_, min_def=0.5):
    r = runs[runs["round"] == round_].copy()
    cols = []
    for vname, bases in VARIANTS.items():
        if not all(b in r for b in bases):
            continue
        for k in KS:
            c = f"{vname} k={k}"
            r[c] = sepcv_plus(r, k=k, bases=bases, shrink=k > 0)
            cols.append(c)
    keys = ["family", "start", "strategy", "learner"]
    out = []
    for cond, sub in r.groupby(keys):
        g = sub.groupby("task_id")
        t = g[cols + ["final_auprc"]].mean()
        ok = g[cols[0]].apply(lambda s: s.notna().mean()) >= min_def   # same availability for all
        t = t[ok]
        if len(t) < 8:
            continue
        row = dict(zip(keys, cond), n_tasks=len(t))
        for c in cols:
            row[c] = rho(t[c], t.final_auprc)
        out.append(row)
    return pd.DataFrame(out)


def failures(task, round_, start="warm", strategy="margin", top=4, min_def=0.5):
    out = []
    t = task[(task["round"] == round_) & (task.start == start) & (task.strategy == strategy)]
    for (fam, L), sub in t.groupby(["family", "learner"]):
        d = sub[sub[f"def_{PRIMARY}"] >= min_def].copy()
        if len(d) >= 8:
            d["rank_score"] = d[PRIMARY].rank(pct=True)
            d["rank_final"] = d.final_auprc.rank(pct=True)
            d["gap"] = d.rank_score - d.rank_final
            for kind, q in (("optimistic (high score, low final)", d.nlargest(top, "gap")),
                            ("pessimistic (low score, high final)", d.nsmallest(top, "gap"))):
                for _, x in q.iterrows():
                    out.append(_row(fam, L, kind, x))
            if {"sepcv_logreg_auroc", "sepcv_knn_auroc"} <= set(d.columns):
                d["base_gap"] = d.sepcv_knn_auroc - d.sepcv_logreg_auroc
                for _, x in d.reindex(d.base_gap.abs().sort_values(ascending=False).index).head(top).iterrows():
                    out.append(_row(fam, L, "linear vs k-NN base disagree", x))
        u = sub[sub[f"def_{PRIMARY}"] < min_def].sort_values("final_auprc")
        for _, x in u.iterrows():
            out.append(_row(fam, L, "undefined at probe", x))
    return pd.DataFrame(out)


def _row(fam, L, kind, x):
    g = lambda c: (round(float(x[c]), 3) if c in x and x[c] == x[c] else np.nan)   # noqa: E731
    return {"family": fam, "learner": L, "case": kind, "task_id": x.task_id,
            "ir": g("ir_eval"), "n_pos": g("n_pos"), "sepcv_plus": g(PRIMARY),
            "sepcv_lr": g("sepcv_logreg_auroc"), "sepcv_rf": g("sepcv_rf_auroc"),
            "sepcv_knn": g("sepcv_knn_auroc"), "holdout_ap": g("holdout_ap"),
            "defined_share": g(f"def_{PRIMARY}"), "final_auprc": g("final_auprc")}
