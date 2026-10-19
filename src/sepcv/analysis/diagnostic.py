"""Diagnostic association: does an early estimator rank tasks by attainable level?

Task level = mean over seeds of each estimator and of the final AUPRC. An
estimator enters a task's mean only for the seeds where it is defined; a task is
*eligible* for an estimator when it is defined in at least ``min_def`` of seeds.
Every predictor is reported on (a) its own eligible population and (b) the
matched population where the primary diagnostic (SepCV+) and the held-out AP are both eligible,
so that comparisons use identical tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sepcv.analysis.load import PRIMARY
from sepcv.analysis.stats import cluster_bootstrap, rho

COND = ["family", "start", "strategy", "learner", "round"]
LOWER_IS_EASIER = {"n3", "n3_rare"}


def task_level(runs, preds):
    g = runs.groupby(COND + ["task_id"])
    t = g[preds + ["final_auprc", "auprc_probe", "n_pos"]].mean()
    for p in preds:
        t[f"def_{p}"] = g[p].apply(lambda s: s.notna().mean())
    t["group"] = g["group"].first()
    t["ir_eval"] = g["ir_eval"].mean()
    return t.reset_index()


def association(task, preds, min_def=0.5, boot=0, primary_round=None, ci_preds=None, seed=0):
    out = []
    for cond, sub in task.groupby(COND):
        c = dict(zip(COND, cond))
        matched = sub[(sub[f"def_{PRIMARY}"] >= min_def) & (sub["def_holdout_ap"] >= min_def)] \
            if "def_holdout_ap" in sub else sub[sub[f"def_{PRIMARY}"] >= min_def]
        for p in preds:
            for pop, s in (("own", sub[sub[f"def_{p}"] >= min_def]), ("matched", matched)):
                s = s.dropna(subset=[p])
                row = {**c, "predictor": p, "population": pop, "n_tasks": len(s),
                       "n_groups": s["group"].nunique(), "rho": rho(s[p], s.final_auprc)}
                want_ci = boot and pop == "matched" and (primary_round is None or c["round"] == primary_round) \
                    and (ci_preds is None or p in ci_preds) and len(s) >= 8
                if want_ci:
                    x, y, gr = s[p].values, s.final_auprc.values, s.group.values
                    row["lo"], row["hi"] = cluster_bootstrap(gr, lambda i: rho(x[i], y[i]), B=boot, seed=seed)
                out.append(row)
    return pd.DataFrame(out)


def availability(task, min_def=0.5, key=PRIMARY):
    a = task.groupby(COND).apply(lambda s: pd.Series({
        "n_tasks": len(s), "n_defined": int((s[f"def_{key}"] >= min_def).sum()),
        "mean_defined_rate": s[f"def_{key}"].mean(), "mean_n_pos": s["n_pos"].mean()}))
    return a.reset_index()


def base_target_matrix(assoc, round_, start="warm", strategy="margin", family=None):
    s = assoc[(assoc.population == "matched") & (assoc["round"] == round_) & (assoc.start == start)
              & (assoc.strategy == strategy) & assoc.predictor.str.startswith("sepcv_")]
    if family is not None:
        s = s[s.family == family]
    return s.pivot_table(index="predictor", columns="learner", values="rho")


def paired_delta(task, preds, base=PRIMARY, round_=3, min_def=0.5, boot=2000, seed=0):
    """Paired cluster-bootstrap of rho(base) - rho(other) on tasks where both are eligible."""
    out = []
    t = task[task["round"] == round_]
    for cond, sub in t.groupby(COND):
        c = dict(zip(COND, cond))
        for p in preds:
            if p == base:
                continue
            s = sub[(sub[f"def_{base}"] >= min_def) & (sub[f"def_{p}"] >= min_def)].dropna(subset=[base, p])
            if len(s) < 8 or s.group.nunique() < 5:
                continue
            x, z, y, g = s[base].values, s[p].values, s.final_auprc.values, s.group.values
            if p in LOWER_IS_EASIER:            # orient so that higher = easier for every predictor
                z = -z
            d = rho(x, y) - rho(z, y)
            lo, hi = cluster_bootstrap(g, lambda i: rho(x[i], y[i]) - rho(z[i], y[i]), B=boot, seed=seed)
            out.append({**c, "base": base, "other": p, "n_tasks": len(s), "delta_rho": d,
                        "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def jackknife(task, preds, round_=3, min_def=0.5):
    """Range of rho when each group is left out in turn (influence of single sources)."""
    out = []
    t = task[task["round"] == round_]
    for cond, sub in t.groupby(COND):
        c = dict(zip(COND, cond))
        for p in preds:
            s = sub[sub[f"def_{p}"] >= min_def].dropna(subset=[p])
            groups = s.group.unique()
            if len(groups) < 6:
                continue
            vals = [rho(s.loc[s.group != g, p], s.loc[s.group != g, "final_auprc"]) for g in groups]
            out.append({**c, "predictor": p, "rho_min": np.nanmin(vals), "rho_max": np.nanmax(vals)})
    return pd.DataFrame(out)


def robust_matched(task, col=PRIMARY, round_=3, min_def=0.5, ref="margin"):
    """rho under each strategy vs the reference strategy, on tasks eligible under both."""
    out = []
    t = task[task["round"] == round_]
    for (fam, start, L), sub in t.groupby(["family", "start", "learner"]):
        r = sub[sub.strategy == ref].set_index("task_id")
        for s_name, s in sub[sub.strategy != ref].groupby("strategy"):
            s = s.set_index("task_id")
            ok = [i for i in s.index if i in r.index and s.at[i, f"def_{col}"] >= min_def
                  and r.at[i, f"def_{col}"] >= min_def]
            if len(ok) < 6:
                continue
            out.append({"family": fam, "start": start, "learner": L, "strategy": s_name,
                        "n_tasks": len(ok), f"rho_{ref}": rho(r.loc[ok, col], r.loc[ok, "final_auprc"]),
                        "rho_strategy": rho(s.loc[ok, col], s.loc[ok, "final_auprc"])})
    return pd.DataFrame(out)
