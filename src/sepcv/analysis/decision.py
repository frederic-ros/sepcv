"""Portfolio decisions from the early probe, evaluated one trajectory per task.

Budget. A portfolio holds M tasks. Every task first receives the exploration
budget n0 (labels at the probe); a fraction f of the tasks (k = round(fM)) is then
funded up to the full budget B. Total labels: M*n0 + k*(B - n0).

Two objectives are reported, on the same draws:

* improvement  U_imp = mean_i [AUPRC_probe_i + funded_i (AUPRC_final_i - AUPRC_probe_i)]
  (an unfunded task keeps the detector it has at the probe; the oracle funds the
  largest improvements);
* deploy       U_dep = (1/M) sum_i funded_i AUPRC_final_i
  (deploy-or-abandon: a detector is deployed only if fully labelled; the oracle
  funds the highest final levels). ``useful_recall`` is the share of tasks reaching
  AUPRC >= ``useful`` that are funded.

Policies fund tasks in decreasing order of an observable score; undefined scores
are funded last. "Rarity, no exploration" skips the exploration phase and spends
the whole budget M*n0 + k*(B - n0) on floor(that / B) fully labelled tasks chosen by
IR (deploy objective only). Each draw samples M tasks without replacement (task
resampling) and one trajectory per task. Paired differences between policies use
the same draws.

Screening (calibrated, LOGO): a run is low-ceiling if its final AUPRC < tau. For
each held-out group, theta is the alpha-quantile of the score among the non-low
runs of the other groups (target false-discard rate alpha). Undefined scores are
never discarded but count in the denominators.

Note: rarity policies use the true pool IR, which an active learner does not
know (an optimistic assumption in favour of rarity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COND = ["family", "start", "strategy", "learner", "round"]

POLICIES = {
    "SepCV+": "sepcv_plus",
    "SepCV-LR (AUROC)": "sepcv_logreg_auroc",
    "SepCV-LR (AP)": "sepcv_logreg_ap",
    "Held-out AP": "holdout_ap",
    "Labeled rare fraction": "rare_frac",
    "Rarity (lower IR first)": "neg_log_ir",
}
NOEXP = "Rarity, no exploration"
PRIMARY_POLICY = "SepCV+"
PAIRS = [(PRIMARY_POLICY, b) for b in ("Rarity (lower IR first)", NOEXP, "Held-out AP",
                                      "Labeled rare fraction", "SepCV-LR (AUROC)")]


class _Pool:
    """Per-task arrays (one row per seed) for fast portfolio draws."""

    def __init__(self, sub, cols):
        self.cols = ["auprc_probe", "final_auprc", "n_lab", "n_lab_final"] + list(cols)
        self.tasks = [g[self.cols].to_numpy(float) for _, g in sub.groupby("task_id")]

    def draw(self, rng, M):
        pick = rng.choice(len(self.tasks), M, replace=False)
        rows = np.array([self.tasks[i][rng.integers(len(self.tasks[i]))] for i in pick])
        return {c: rows[:, j] for j, c in enumerate(self.cols)}


def _ranked_fund(score, k, rng):
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.lexsort((rng.random(len(s)), -s))
    funded = np.zeros(len(s), bool)
    funded[order[:k]] = True
    return funded


def default_portfolio(n_tasks):
    return max(6, int(round(0.6 * n_tasks)))


def allocation(runs, fractions=(0.2, 0.4, 0.6), draws=300, portfolio=None,
               useful=0.5, seed=0, policies=None):
    """Return (per-policy table, paired-difference table)."""
    policies = policies or {k: v for k, v in POLICIES.items() if v in runs}
    rng = np.random.default_rng(seed)
    out, paired = [], []
    for cond, sub in runs.groupby(COND):
        c = dict(zip(COND, cond))
        sub = sub.dropna(subset=["auprc_probe", "final_auprc"])
        pool = _Pool(sub, list(policies.values()))
        N = len(pool.tasks)
        if N < 8:
            continue
        M = min(portfolio or default_portfolio(N), N)
        acc, dif = {}, {}
        for _ in range(draws):
            rows = pool.draw(rng, M)
            p0, p1 = rows["auprc_probe"], rows["final_auprc"]
            n0, B = np.nanmedian(rows["n_lab"]), np.nanmedian(rows["n_lab_final"])
            gain = p1 - p0
            use = p1 >= useful
            for f in fractions:
                k = max(1, int(round(f * M)))
                imp_rand = p0.mean() + (k / M) * gain.mean()
                imp_orc = p0.mean() + np.sort(gain)[::-1][:k].sum() / M
                dep_rand = (k / M) * p1.mean()
                dep_orc = np.sort(p1)[::-1][:k].sum() / M
                funds = {"Oracle (improvement)": _ranked_fund(gain, k, rng),
                         "Oracle (final level)": _ranked_fund(p1, k, rng)}
                for name, col in policies.items():
                    funds[name] = _ranked_fund(rows[col], k, rng)
                vals = {}
                for name, fund in funds.items():
                    imp = p0.mean() + gain[fund].sum() / M
                    dep = p1[fund].sum() / M
                    vals[name] = (imp - imp_rand, dep - dep_rand)
                    acc.setdefault((f, name), []).append(
                        (imp - imp_rand, imp_orc - imp_rand, dep - dep_rand, dep_orc - dep_rand,
                         (fund & use).sum() / max(use.sum(), 1)))
                if "neg_log_ir" in rows and B > n0:
                    k2 = min(M, int((M * n0 + k * (B - n0)) // B))
                    fund2 = _ranked_fund(rows["neg_log_ir"], k2, rng)
                    dep2 = p1[fund2].sum() / M
                    vals[NOEXP] = (np.nan, dep2 - dep_rand)
                    acc.setdefault((f, NOEXP), []).append(
                        (np.nan, imp_orc - imp_rand, dep2 - dep_rand, dep_orc - dep_rand,
                         (fund2 & use).sum() / max(use.sum(), 1)))
                for a, b in PAIRS:
                    if a in vals and b in vals:
                        dif.setdefault((f, a, b), []).append(
                            (vals[a][0] - vals[b][0], vals[a][1] - vals[b][1]))
        for (f, name), v in acc.items():
            v = np.array(v, float)
            row = {**c, "fraction": f, "policy": name, "portfolio": M, "n_tasks": N}
            for j, obj in ((0, "imp"), (2, "dep")):
                g, den = v[:, j], v[:, j + 1]
                if np.all(np.isnan(g)):
                    continue
                row[f"{obj}_gain"] = np.nanmean(g)
                row[f"{obj}_lo"], row[f"{obj}_hi"] = np.nanpercentile(g, [2.5, 97.5])
                row[f"{obj}_recovery"] = np.nanmean(g) / den.mean() if den.mean() > 0 else np.nan
            row["useful_recall"] = v[:, 4].mean()
            out.append(row)
        for (f, a, b), d in dif.items():
            d = np.array(d, float)
            row = {**c, "fraction": f, "a": a, "b": b, "portfolio": M}
            for j, obj in ((0, "imp"), (1, "dep")):
                if np.all(np.isnan(d[:, j])):
                    continue
                row[f"{obj}_diff"] = np.nanmean(d[:, j])
                row[f"{obj}_lo"], row[f"{obj}_hi"] = np.nanpercentile(d[:, j], [2.5, 97.5])
                row[f"{obj}_p_a_better"] = float(np.nanmean(d[:, j] > 0))
            paired.append(row)
    return pd.DataFrame(out), pd.DataFrame(paired)


def screening(runs, taus=(0.05, 0.1, 0.2), alphas=(0.025, 0.05, 0.10), policies=None):
    policies = policies or {k: v for k, v in POLICIES.items() if v in runs}
    out = []
    for cond, sub in runs.groupby(COND):
        c = dict(zip(COND, cond))
        sub = sub.dropna(subset=["final_auprc"])
        groups = sub["group"].values
        if len(np.unique(groups)) < 5:
            continue
        for tau in taus:
            low = sub.final_auprc.values < tau
            n_low_tasks = sub.loc[low, "task_id"].nunique()
            n_low_groups = sub.loc[low, "group"].nunique()
            if low.sum() == 0:
                out.append({**c, "tau": tau, "n_low": 0, "n_low_tasks": 0, "n_low_groups": 0})
                continue
            for alpha in alphas:
                for name, col in policies.items():
                    sc = sub[col].values.astype(float)
                    discard = np.zeros(len(sub), bool)
                    for g in np.unique(groups):
                        te, tr = groups == g, groups != g
                        ref = sc[tr & ~low & ~np.isnan(sc)]
                        if len(ref) < 10:
                            continue
                        theta = np.quantile(ref, alpha)
                        discard[te] = ~np.isnan(sc[te]) & (sc[te] < theta)
                    fin = sub.final_auprc.values
                    missed = low & ~discard
                    out.append({**c, "tau": tau, "alpha": alpha, "policy": name,
                                "n_runs": len(sub), "n_low": int(low.sum()),
                                "n_low_tasks": n_low_tasks, "n_low_groups": n_low_groups,
                                "defined_rate": float(np.mean(~np.isnan(sc))),
                                "low_recall": float((discard & low).sum() / low.sum()),
                                "missed_low_undefined": float(np.isnan(sc[missed]).mean())
                                if missed.any() else np.nan,
                                "false_discard": float((discard & ~low).sum() / max((~low).sum(), 1)),
                                "discard_rate": float(discard.mean()),
                                "auprc_lost_share": float(fin[discard & ~low].sum() / fin.sum())})
    return pd.DataFrame(out)


def screening_calsize(runs, sizes=(5, 10, 20), tau=0.1, alpha=0.05, reps=50, col="sepcv_plus",
                      seed=0):
    """Calibrated screening when theta is learned from only `size` other groups
    (random subsets, `reps` repetitions): stability of the calibration."""
    rng = np.random.default_rng(seed)
    out = []
    for cond, sub in runs.groupby(COND):
        c = dict(zip(COND, cond))
        sub = sub.dropna(subset=["final_auprc"])
        groups = sub["group"].values
        ug = np.unique(groups)
        low = sub.final_auprc.values < tau
        if len(ug) < 6 or low.sum() == 0 or col not in sub:
            continue
        sc = sub[col].values.astype(float)
        for size in list(sizes) + ["all"]:
            if size != "all" and size >= len(ug) - 1:
                continue
            res = []
            for _ in range(1 if size == "all" else reps):
                discard = np.zeros(len(sub), bool)
                for g in ug:
                    others = ug[ug != g]
                    cal = others if size == "all" else rng.choice(others, size, replace=False)
                    m = np.isin(groups, cal) & ~low & ~np.isnan(sc)
                    if m.sum() < 10:
                        continue
                    theta = np.quantile(sc[m], alpha)
                    te = groups == g
                    discard[te] = ~np.isnan(sc[te]) & (sc[te] < theta)
                res.append(((discard & low).sum() / low.sum(),
                            (discard & ~low).sum() / max((~low).sum(), 1)))
            res = np.array(res)
            out.append({**c, "tau": tau, "alpha": alpha, "cal_groups": size, "reps": len(res),
                        "low_recall": res[:, 0].mean(), "false_discard": res[:, 1].mean(),
                        "false_discard_p90": float(np.percentile(res[:, 1], 90))})
    return pd.DataFrame(out)
