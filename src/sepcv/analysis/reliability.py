"""Reliability of a single early estimate, and the intra-task frontier.

* ``by_positives`` -- run-level association between SepCV and final AUPRC, binned
  by the number of rare cases discovered at the probe (3, 4, 5, 6-8, 9+), and the
  share of runs whose single estimate puts the task on the same side of the
  median as the task's seed-mean estimate (rank-side agreement).
* ``next_batch`` -- correlation between SepCV at round t and the AUPRC gain of
  the next acquisition batch (round t -> t+1), pooled and per round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sepcv.analysis.load import PRIMARY
from sepcv.analysis.stats import rho

COND = ["family", "start", "strategy", "learner", "round"]
BINS = [(3, 3), (4, 4), (5, 5), (6, 8), (9, 10**9)]


def by_positives(runs, col=PRIMARY):
    out = []
    for cond, sub in runs.groupby(COND):
        c = dict(zip(COND, cond))
        sub = sub.dropna(subset=[col, "final_auprc"])
        if sub.task_id.nunique() < 6:
            continue
        tmean = sub.groupby("task_id")[col].mean()
        med = tmean.median()
        side_task = (tmean >= med)
        for lo, hi in BINS:
            s = sub[(sub.n_pos >= lo) & (sub.n_pos <= hi)]
            if len(s) < 6:
                continue
            agree = ((s[col] >= med).values == side_task.reindex(s.task_id).values).mean()
            out.append({**c, "n_pos_bin": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"),
                        "n_runs": len(s), "n_tasks": s.task_id.nunique(),
                        "rho_run": rho(s[col], s.final_auprc),
                        "sd_within_task": float(s.groupby("task_id")[col].std().mean()),
                        "median_side_agreement": float(agree)})
    return pd.DataFrame(out)


def next_batch(runs, col=PRIMARY):
    out = []
    keys = ["family", "start", "strategy", "learner"]
    for cond, sub in runs.groupby(keys):
        c = dict(zip(keys, cond))
        s = sub.dropna(subset=[col, "auprc_probe", "auprc_next"])
        if len(s) < 10:
            continue
        gain = s.auprc_next - s.auprc_probe
        row = {**c, "n": len(s), "rho_pooled": rho(s[col], gain),
               "rho_level": rho(s[col], s.final_auprc)}
        for r, g in s.groupby("round"):
            row[f"rho_round{r}"] = rho(g[col], g.auprc_next - g.auprc_probe)
        out.append(row)
    return pd.DataFrame(out)


IR_STRATA = [(0, 10), (10, 50), (50, 200), (200, 1e9)]


def availability_by_ir(runs, col=PRIMARY):
    """Share of runs with SepCV defined, by IR stratum, start, learner, round (+ final >= 3 positives)."""
    out = []
    r = runs.copy()
    r["ir_stratum"] = pd.cut(r.ir_eval, [a for a, _ in IR_STRATA] + [1e9], right=False,
                             labels=["<10", "10-50", "50-200", ">=200"])
    for (fam, start, strat, L, strat_ir), s in r.groupby(["family", "start", "strategy", "learner", "ir_stratum"],
                                                        observed=True):
        row = {"family": fam, "start": start, "strategy": strat, "learner": L, "ir_stratum": str(strat_ir),
               "n_tasks": s.task_id.nunique()}
        for rd, g in s.groupby("round"):
            row[f"defined_r{rd}"] = float(g[col].notna().mean())
        row["final_ge3_pos"] = float((s.groupby(["task_id", "seed"]).n_pos_final.first() >= 3).mean())
        out.append(row)
    return pd.DataFrame(out)


def sstar_error(runs, tasks, col="sepcv_logreg_auroc", ref="sstar_logreg"):
    """|SepCV - S*| by number of discovered positives and acquisition strategy."""
    if ref not in tasks:
        return pd.DataFrame()
    r = runs.merge(tasks[["task_id", ref]], on="task_id", how="left").dropna(subset=[col, ref])
    out = []
    for (fam, start, strat, rd), s in r.groupby(["family", "start", "strategy", "round"]):
        for lo, hi in BINS:
            g = s[(s.n_pos >= lo) & (s.n_pos <= hi)]
            if len(g) < 6:
                continue
            err = g[col] - g[ref]
            out.append({"family": fam, "start": start, "strategy": strat, "round": rd,
                        "n_pos_bin": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"),
                        "n_runs": len(g), "mean_error": float(err.mean()),
                        "mean_abs_error": float(err.abs().mean())})
    return pd.DataFrame(out)


def decision_stability(runs, col=PRIMARY, alpha=0.05, tau=0.1, max_pairs=20000, seed=0):
    """How much a single estimate can be trusted for a decision, by discovered positives.

    * rank_stability -- share of run pairs (two different tasks, same n+ bin) ordered
      like the seed-mean estimates of their tasks;
    * rank_accuracy  -- share of the same pairs ordered like the tasks' mean final AUPRC;
    * screen_flip    -- share of runs whose screening decision (score < theta) differs
      from the decision taken on the task's seed-mean estimate; theta is the
      alpha-quantile of the task-mean score among tasks with mean final AUPRC >= tau
      (descriptive, in-sample threshold).
    """
    rng = np.random.default_rng(seed)
    out = []
    for cond, sub in runs.groupby(COND):
        c = dict(zip(COND, cond))
        sub = sub.dropna(subset=[col, "final_auprc"])
        if sub.task_id.nunique() < 6:
            continue
        tm = sub.groupby("task_id")[[col, "final_auprc"]].mean()
        ok = tm[tm.final_auprc >= tau][col]
        theta = np.quantile(ok, alpha) if len(ok) >= 5 else np.nan
        dec_task = (tm[col] < theta)
        for lo, hi in BINS:
            s = sub[(sub.n_pos >= lo) & (sub.n_pos <= hi)]
            if len(s) < 6 or s.task_id.nunique() < 3:
                continue
            sc, tid = s[col].values, s.task_id.values
            i = rng.integers(len(s), size=max_pairs)
            j = rng.integers(len(s), size=max_pairs)
            keep = tid[i] != tid[j]
            i, j = i[keep], j[keep]
            ms, mf = tm[col].reindex(tid).values, tm.final_auprc.reindex(tid).values
            d_run = np.sign(sc[i] - sc[j])
            stab = np.mean(d_run == np.sign(ms[i] - ms[j])) if len(i) else np.nan
            acc = np.mean(d_run == np.sign(mf[i] - mf[j])) if len(i) else np.nan
            flip = np.mean((sc < theta) != dec_task.reindex(tid).values) if theta == theta else np.nan
            out.append({**c, "n_pos_bin": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"),
                        "n_runs": len(s), "n_tasks": s.task_id.nunique(), "rank_stability": stab,
                        "rank_accuracy": acc, "screen_flip": flip})
    return pd.DataFrame(out)
