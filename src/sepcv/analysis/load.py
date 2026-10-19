"""Read unit files into tidy tables.

``runs``  -- one row per (task, seed, start, strategy, target learner, probe round):
             estimators on L_t, test AUPRC of the target learner at the probe,
             final AUPRC. Learner-free strategies (random, coreset) are expanded
             to every learner (same labels, learner-specific AUPRC).
``tasks`` -- one row per task: metadata, realized IR, full-data references S*.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ID_COLS = ["task_id", "family", "group", "dataset", "domain", "backbone", "positive_class",
           "ir_target"]


def _obs_lce(probes, n_final, key="sepcv_logreg_ap"):
    """Observable learning-curve extrapolation: a + b log n fitted on the
    estimator values at the probes seen so far, extrapolated to the final budget."""
    out, xs, ys = {}, [], []
    for p in probes:
        v = p.get(key)
        if v is not None and v == v:
            xs.append(np.log(p["n_lab"])); ys.append(v)
        if len(xs) >= 2 and len(set(xs)) >= 2:
            b, a = np.polyfit(xs, ys, 1)
            out[p["round"]] = float(np.clip(a + b * np.log(n_final), 0, 1))
        else:
            out[p["round"]] = np.nan
    return out


# Primary diagnostic (SepCV+) -------------------------------------------------
# SepCV+ = 0.5 + n+/(n+ + k) * (mean_b SepCV_b - 0.5)
#   * mean over the CV-AUROC bases (LR, RF, k-NN) that are defined on L_t:
#     a learner-agnostic reading of separability;
#   * shrinkage towards chance (0.5) with a prior worth k pseudo-positives: with
#     few discovered rare cases the observed separability is weak evidence.
# k = 3 (the definition threshold) is fixed a priori; ``sensitivity`` reports k.
# SepCV+ is defined exactly when SepCV-LR is (n+ >= 3 and n- >= 3).
PRIMARY = "sepcv_plus"
SHRINK_K = 3
BASES = ("sepcv_logreg_auroc", "sepcv_rf_auroc", "sepcv_knn_auroc")


def sepcv_plus(runs, k=SHRINK_K, bases=BASES, shrink=True):
    b = [c for c in bases if c in runs]
    m = runs[b].mean(axis=1, skipna=True)
    m[runs["sepcv_logreg_auroc"].isna()] = np.nan
    if not shrink or k == 0:
        return m
    w = runs["n_pos"] / (runs["n_pos"] + k)
    return 0.5 + w * (m - 0.5)


def add_sepcv_plus(runs, k=SHRINK_K):
    if "sepcv_logreg_auroc" not in runs or "n_pos" not in runs:
        return runs
    runs["sepcv_ens"] = sepcv_plus(runs, shrink=False)                       # bases only
    runs["sepcv_shr_lr"] = sepcv_plus(runs, k=k, bases=("sepcv_logreg_auroc",))  # shrinkage only
    runs[PRIMARY] = sepcv_plus(runs, k=k)
    return runs


def load(result_dirs):
    rows, metas = [], []
    for d in result_dirs:
        files = glob.glob(os.path.join(str(d), "units", "*.json"))
        for f in files:
            with open(f) as fh:
                u = json.load(fh)
            fx = os.path.join(os.path.dirname(os.path.dirname(f)), "units_extra", os.path.basename(f))
            if os.path.exists(fx):                      # post-hoc probes (sepcv reprobe)
                with open(fx) as fh:
                    extra = {(t["strategy"], t["learner"]): t["probes"] for t in json.load(fh)["trajectories"]}
                for tr in u["trajectories"]:
                    more = extra.get((tr["strategy"], tr["learner"]), [])
                    have = {p["round"] for p in tr["probes"]}
                    tr["probes"] = sorted(tr["probes"] + [{**p, "extra": True} for p in more
                                                          if p["round"] not in have],
                                          key=lambda p: p["round"])
            m = u["meta"]
            metas.append(m)
            ident = {k: m.get(k) for k in ID_COLS}
            for tr in u["trajectories"]:
                lce = _obs_lce(tr["probes"], tr["n_lab_final"])
                for L, fin in tr["final"].items():
                    curve = tr["curve"][L]
                    for p in tr["probes"]:
                        rows.append({**ident, "seed": m["seed"], "start": m["start"],
                                     "ir_eval": m["ir_eval"], "strategy": tr["strategy"],
                                     "learner": L, **p, "obs_lce": lce.get(p["round"], np.nan),
                                     "auprc_probe": curve.get(str(p["round"]), np.nan),
                                     "auprc_next": curve.get(str(p["round"] + 1), np.nan),
                                     "final_auprc": fin, "n_pos_final": tr["n_pos_final"],
                                     "n_lab_final": tr["n_lab_final"]})
    if not rows:
        raise RuntimeError(f"no unit files under {result_dirs}")
    runs = pd.DataFrame(rows)
    runs["extra"] = runs["extra"].eq(True) if "extra" in runs else False
    meta = pd.DataFrame(metas)
    agg = {"ir_eval": "mean", "n_used": "mean", "n_rare": "mean", "d": "first"}
    sstar = [c for c in meta.columns if c.startswith("sstar_")]
    tasks = meta.groupby("task_id").agg({**agg, **{c: "max" for c in sstar}})
    tasks = tasks.join(meta.groupby("task_id")[["family", "group", "dataset", "domain"]].first())
    if "fisher_f1" in runs:                       # F1 is unbounded: use log(1 + F1)
        runs["log_f1"] = np.log1p(runs["fisher_f1"])
    # oracle reference (NOT observable): test AUPRC of the target learner at the probe
    runs["probe_oracle"] = runs["auprc_probe"]
    add_sepcv_plus(runs)
    # two-part forecaster input: the primary diagnostic with an explicit "undefined" indicator
    if PRIMARY in runs:
        runs["sepcv_undef"] = runs[PRIMARY].isna().astype(float)
        runs["sepcv_fill"] = runs[PRIMARY].fillna(0.5)
    runs["log_ir"] = np.log(runs["ir_eval"])
    runs["neg_log_ir"] = -runs["log_ir"]
    return runs, tasks.reset_index()


ESTIMATORS_ORDER = ["sepcv_plus", "sepcv_ens", "sepcv_shr_lr", "sepcv_logreg_auroc", "sepcv_logreg_ap", "sepcv_rf_auroc", "sepcv_rf_ap",
                    "sepcv_knn_auroc", "sepcv_knn_ap", "holdout_ap", "rare_frac", "obs_lce",
                    "log_f1", "n3", "n3_rare", "neg_log_ir", "probe_oracle"]


def predictors(runs):
    return [c for c in ESTIMATORS_ORDER if c in runs.columns]
