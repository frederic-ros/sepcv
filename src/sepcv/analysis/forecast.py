"""Prospective forecasting with grouped leave-one-group-out (LOGO).

All trajectories of a held-out group (a dataset, a source such as the five
Polish-bankruptcy years, or an image dataset with all its classes and both
backbones) are excluded when fitting the forecaster. Two read-outs:

* run level  -- one trajectory, one prediction (what a practitioner deploys);
* task level -- run predictions averaged per task, against the task-mean final AUPRC.

Undefined estimators are not imputed: ``coverage`` reports the share of runs
(and tasks) for which a prediction exists; predictor sets are compared on the
matched runs where SepCV+ and the held-out AP are both defined
(plus whatever extra columns a set needs, reported in ``n_runs``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sepcv.analysis.stats import cluster_bootstrap, logo_predict, r2, rho

COND = ["family", "start", "strategy", "learner", "round"]

PRIMARY_SET = "SepCV+"
TWO_PART = "SepCV+ + undefined flag (all runs)"
COMPETITORS = "Competitors (held-out AP, rare fraction, N3-rare)"
DEFAULT_SETS = {
    PRIMARY_SET: ["sepcv_plus"],
    "SepCV-LR (AUROC)": ["sepcv_logreg_auroc"],
    "SepCV-LR (AP)": ["sepcv_logreg_ap"],
    "SepCV bases mean (no shrinkage)": ["sepcv_ens"],
    "SepCV-LR shrunk (no base mean)": ["sepcv_shr_lr"],
    "Held-out AP": ["holdout_ap"],
    "Labeled rare fraction": ["rare_frac"],
    "Rarity (log IR)": ["neg_log_ir"],
    "Observable LCE": ["obs_lce"],
    "Complexity (log F1, N3-rare)": ["log_f1", "n3_rare"],
    "SepCV+ + rare fraction": ["sepcv_plus", "rare_frac"],
    "SepCV+ + log IR": ["sepcv_plus", "neg_log_ir"],
    "SepCV+ + observable LCE": ["sepcv_plus", "obs_lce"],
    "Held-out AP + rare fraction": ["holdout_ap", "rare_frac"],
    COMPETITORS: ["holdout_ap", "rare_frac", "n3_rare"],
    "Competitors + SepCV+": ["holdout_ap", "rare_frac", "n3_rare", "sepcv_plus"],
    TWO_PART: ["sepcv_fill", "sepcv_undef"],
    "Undefined flag alone (all runs)": ["sepcv_undef"],
    "Rarity, all runs": ["neg_log_ir"],
    "Probe test AUPRC (oracle, not observable)": ["probe_oracle"],
}
# (a, b, population): paired Delta R^2 = R^2(a) - R^2(b) on the same tasks
PAIRED = [(PRIMARY_SET, o, "common") for o in
          ["SepCV-LR (AUROC)", "SepCV-LR (AP)", "SepCV bases mean (no shrinkage)",
           "SepCV-LR shrunk (no base mean)", "Held-out AP", "Labeled rare fraction",
           "Rarity (log IR)", "Observable LCE", "Complexity (log F1, N3-rare)",
           "Probe test AUPRC (oracle, not observable)"]]
PAIRED += [("Competitors + SepCV+", COMPETITORS, "common"),          # information added by SepCV+
           (TWO_PART, "Undefined flag alone (all runs)", "own"),     # value beyond the flag
           (TWO_PART, "Rarity, all runs", "own")]
ALL_RUNS = {TWO_PART, "Undefined flag alone (all runs)", "Rarity, all runs"}


def forecast(runs, sets=None, boot=0, primary_round=None, seed=0):
    sets = sets or {k: v for k, v in DEFAULT_SETS.items() if all(c in runs for c in v)}
    core = [c for c in ("sepcv_plus", "holdout_ap") if c in runs]
    out, paired = [], []
    for cond, sub in runs.groupby(COND):
        c = dict(zip(COND, cond))
        tl_pop = {"common": {}, "own": {}}
        common = sub.dropna(subset=core)                       # matched runs
        cov_common = len(common) / max(len(sub), 1)
        for name, cols in sets.items():
            pops = (("own", sub.dropna(subset=cols)),) if name in ALL_RUNS else \
                (("own", sub.dropna(subset=cols)), ("common", common.dropna(subset=cols)))
            for pop, s in pops:
                if s["group"].nunique() < 5:
                    continue
                pred = logo_predict(s[cols].values, s.final_auprc.values, s.group.values)
                ok = ~np.isnan(pred)
                s = s[ok].assign(pred=pred[ok])
                tl = s.groupby("task_id").agg(pred=("pred", "mean"), y=("final_auprc", "mean"),
                                              group=("group", "first"))
                if len(tl) < 3:
                    continue
                tl_pop[pop][name] = tl
                row = {**c, "set": name, "population": pop,
                       "coverage_runs": len(s) / max(len(sub), 1),
                       "coverage_common": cov_common, "n_runs": len(s), "n_tasks": len(tl),
                       "r2_run": r2(s.final_auprc, s.pred), "r2_task": r2(tl.y, tl.pred),
                       "rho_task": rho(tl.pred, tl.y),
                       "mae_task": float(np.mean(np.abs(tl.y - tl.pred)))}
                if boot and (pop == "common" or name in ALL_RUNS) \
                        and (primary_round is None or c["round"] == primary_round):
                    yv, pv, gv = tl.y.values, tl.pred.values, tl.group.values
                    row["r2_task_lo"], row["r2_task_hi"] = cluster_bootstrap(
                        gv, lambda i: r2(yv[i], pv[i]), B=boot, seed=seed)
                out.append(row)
        if boot and (primary_round is None or c["round"] == primary_round):
            for a, b, pop in PAIRED:
                ta, tb = tl_pop[pop].get(a), tl_pop[pop].get(b)
                if ta is None or tb is None:
                    continue
                j = ta.join(tb[["pred"]], rsuffix="_o", how="inner")
                if j.group.nunique() < 5:
                    continue
                y, pa, pb, g = j.y.values, j.pred.values, j.pred_o.values, j.group.values
                lo, hi = cluster_bootstrap(g, lambda i: r2(y[i], pa[i]) - r2(y[i], pb[i]), B=boot, seed=seed)
                paired.append({**c, "base": a, "other": b, "population": pop, "n_tasks": len(j),
                               "delta_r2": r2(y, pa) - r2(y, pb), "lo": lo, "hi": hi})
    return pd.DataFrame(out), pd.DataFrame(paired)
