"""Successive halving (SH) against early ranking, at equal total annotation budget.

The deploy-or-abandon portfolio of ``decision.allocation`` is replayed with
multi-stage schemes. A portfolio holds M tasks; f*M of them must end fully labelled
(B labels). The reference single-stage policy explores every task to n0 (the
primary probe) and funds the k = round(fM) best by an early score. Its total budget
Btot = M*n0 + k*(B - n0) is imposed on every scheme.

A successive-halving scheme keeps m_0 = M tasks up to L_0 labels, then repeatedly
keeps the best ceil(m_j / eta) of them by a reward measured at L_j, until k remain;
those are labelled to B. Rung levels are geometric between L_0 and B, snapped to the
rounds where the reward is available. L_0 is the largest level such that the scheme
spends at most Btot. Schemes:

* ``SH (oracle reward)``  reward = test AUPRC of the current model (idealized: an
  active learner has no labelled test set), available at every round;
* ``SH (held-out AP)``    reward = held-out AP on L_t (observable), at probe rounds;
* ``SH (SepCV+)``         reward = SepCV+ on L_t (observable), at probe rounds;
* ``SepCV+ -> SH``        first cut by SepCV+ (observable), later cuts by the
  oracle reward: early ranking used to initialize a reward-driven scheme.

All schemes are trajectory-consistent: a task's labelled set at L_j is the prefix of
its stored trajectory, so the rewards and final AUPRC come from the same run.
Undefined scores rank last (random tie-break). Probe rounds include the post-hoc
probes of ``sepcv reprobe`` when present.

Usage: python -m sepcv halving <results...> --out <folder> [--draws 300] [--eta 2 3]
"""
from __future__ import annotations

import glob
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sepcv.analysis.load import PRIMARY, load

LOG = logging.getLogger("sepcv")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
COND = ["family", "start", "strategy", "learner"]


# ----------------------------------------------------------------------------- data
def _curves(result_dirs, strategies):
    """(task, seed, start, strategy, learner) -> (labels per round, test AUPRC per round)."""
    out = {}
    for d in result_dirs:
        for f in glob.glob(os.path.join(str(d), "units", "*.json")):
            with open(f) as fh:
                u = json.load(fh)
            m = u["meta"]
            for tr in u["trajectories"]:
                if tr["strategy"] not in strategies or tr["learner"] == "*":
                    continue
                L = tr["learner"]
                c = tr["curve"][L]
                R = max(int(r) for r in c)
                lab = np.cumsum([len(b) for b in tr["acq"]])[: R + 1].astype(float)
                auc = np.array([c.get(str(r), np.nan) for r in range(R + 1)], float)
                if len(lab) < R + 1:
                    continue
                out[(m["task_id"], m["seed"], m["start"], tr["strategy"], L)] = (lab, auc)
    return out


class _Pool:
    def __init__(self, runs, curves, cols):
        self.tasks = []
        for tid, g in runs.groupby("task_id"):
            seeds = []
            for (seed, start, strat, L), h in g.groupby(["seed", "start", "strategy", "learner"]):
                key = (tid, seed, start, strat, L)
                if key not in curves:
                    continue
                lab, auc = curves[key]
                R = len(lab) - 1
                sc = {c: np.full(R + 1, np.nan) for c in cols}
                for _, row in h.iterrows():
                    r = int(row["round"])
                    if r <= R:
                        for c in cols:
                            sc[c][r] = row[c]
                probed = np.zeros(R + 1, bool)
                probed[h["round"].astype(int).clip(upper=R).values] = True
                seeds.append((lab, auc, sc, probed))
            if seeds:
                self.tasks.append(seeds)

    def draw(self, rng, M):
        pick = rng.choice(len(self.tasks), M, replace=False)
        return [self.tasks[i][rng.integers(len(self.tasks[i]))] for i in pick]


# -------------------------------------------------------------------------- schemes
def _top(score, idx, keep, rng):
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.lexsort((rng.random(len(s)), -s))
    return idx[order[:keep]]


def _survivors(M, k, eta):
    m = [M]
    while m[-1] > k:
        m.append(max(k, math.ceil(m[-1] / eta)))
    return m


def _plan(lab, m, allowed, budget):
    """Rung rounds r_0 < ... < r_{J-1} < R (final), allowed[j] = rounds where the reward of
    cut j is available. Returns the plan that spends the most without exceeding budget."""
    R = len(lab) - 1
    J = len(m) - 1                                   # number of cuts
    if J == 0:
        return [R] if m[0] * lab[R] <= budget else None
    best, best_cost = None, -1.0
    for r0 in allowed[0]:
        if r0 >= R:
            continue
        L0, LB = lab[r0], lab[R]
        rounds, ok = [r0], True
        for j in range(1, J):
            target = L0 * (LB / L0) ** (j / J)
            cand = [r for r in allowed[j] if r > rounds[-1] and r < R]
            if not cand:
                ok = False
                break
            rounds.append(min(cand, key=lambda r: abs(lab[r] - target)))
        if not ok:
            continue
        levels = [lab[r] for r in rounds] + [LB]
        cost = m[0] * levels[0] + sum(m[j] * (levels[j] - levels[j - 1]) for j in range(1, J + 1))
        if cost <= budget + 1e-9 and cost > best_cost:
            best, best_cost = rounds + [R], cost
    return best


def run_scheme(port, k, budget, eta, reward, rng):
    """reward: list of callables (task_tuple, round) -> score, one per cut (last one reused)."""
    M = len(port)
    lab = port[0][0]
    R = len(lab) - 1
    m = _survivors(M, k, eta)
    J = len(m) - 1
    allowed = []
    for j in range(J):
        kind = reward[min(j, len(reward) - 1)][0]
        if kind == "oracle":
            allowed.append(list(range(R)))
        else:                                        # observable score: probe rounds only
            pr = np.flatnonzero(np.all([t[3] for t in port], axis=0))
            allowed.append([int(r) for r in pr if r < R])
    plan = _plan(lab, m, allowed, budget)
    if plan is None:
        return np.nan, np.nan
    alive = np.arange(M)
    for j in range(J):
        kind, col = reward[min(j, len(reward) - 1)]
        r = plan[j]
        score = np.array([port[i][1][r] if kind == "oracle" else port[i][2][col][r] for i in alive])
        alive = _top(score, alive, m[j + 1], rng)
    levels = [lab[r] for r in plan]
    cost = m[0] * levels[0] + sum(m[j] * (levels[j] - levels[j - 1]) for j in range(1, J + 1))
    dep = sum(port[i][1][R] for i in alive) / M
    return dep, cost / budget


def halving(runs, curves, n0_round=3, fractions=FRACTIONS, etas=(2, 3), draws=300, seed=0):
    cols = [c for c in (PRIMARY, "holdout_ap") if c in runs]
    rng = np.random.default_rng(seed)
    res, paired = [], []
    for cond, sub in runs.groupby(COND):
        c = dict(zip(COND, cond))
        pool = _Pool(sub, curves, cols)
        N = len(pool.tasks)
        if N < 8:
            continue
        M = max(6, int(round(0.6 * N)))
        acc, dif = {}, {}
        for _ in range(draws):
            port = pool.draw(rng, M)
            R = len(port[0][0]) - 1
            if any(len(t[0]) != R + 1 for t in port):
                continue
            n0, B = port[0][0][n0_round], port[0][0][R]
            fin = np.array([t[1][R] for t in port])
            s0 = np.array([t[2][PRIMARY][n0_round] for t in port])
            for f in fractions:
                k = max(1, int(round(f * M)))
                budget = M * n0 + k * (B - n0)
                rand = (k / M) * fin.mean()
                vals = {"SepCV+ ranking (single stage)":
                        (fin[_top(s0, np.arange(M), k, rng)].sum() / M, 1.0),
                        "Oracle (final level)": (np.sort(fin)[::-1][:k].sum() / M, 1.0)}
                for eta in etas:
                    vals[f"SH (oracle reward), eta={eta}"] = run_scheme(
                        port, k, budget, eta, [("oracle", None)], rng)
                    if "holdout_ap" in cols:
                        vals[f"SH (held-out AP), eta={eta}"] = run_scheme(
                            port, k, budget, eta, [("obs", "holdout_ap")], rng)
                    vals[f"SH (SepCV+), eta={eta}"] = run_scheme(
                        port, k, budget, eta, [("obs", PRIMARY)], rng)
                    vals[f"SepCV+ -> SH (oracle reward), eta={eta}"] = run_scheme(
                        port, k, budget, eta, [("obs", PRIMARY), ("oracle", None)], rng)
                for name, (dep, used) in vals.items():
                    acc.setdefault((f, name), []).append((dep - rand, used))
                for eta in etas:
                    for a, b in ((f"SepCV+ -> SH (oracle reward), eta={eta}", f"SH (oracle reward), eta={eta}"),
                                 ("SepCV+ ranking (single stage)", f"SH (oracle reward), eta={eta}"),
                                 ("SepCV+ ranking (single stage)", f"SH (held-out AP), eta={eta}"),
                                 (f"SH (SepCV+), eta={eta}", f"SH (held-out AP), eta={eta}"),
                                 (f"SH (SepCV+), eta={eta}", "SepCV+ ranking (single stage)")):
                        if a in vals and b in vals:
                            dif.setdefault((f, a, b), []).append(vals[a][0] - vals[b][0])
        for (f, name), v in acc.items():
            v = np.array(v, float)
            g = v[:, 0]
            ok = ~np.isnan(g)
            if not ok.any():
                continue
            lo, hi = np.percentile(g[ok], [2.5, 97.5])
            res.append({**c, "fraction": f, "policy": name, "portfolio": M, "n_tasks": N,
                        "dep_gain": g[ok].mean(), "dep_lo": lo, "dep_hi": hi,
                        "feasible": ok.mean(), "budget_used": np.nanmean(v[:, 1])})
        for (f, a, b), d in dif.items():
            d = np.array(d, float)
            d = d[~np.isnan(d)]
            if len(d) == 0:
                continue
            lo, hi = np.percentile(d, [2.5, 97.5])
            paired.append({**c, "fraction": f, "a": a, "b": b, "dep_diff": d.mean(),
                           "dep_lo": lo, "dep_hi": hi, "p_a_better": float((d > 0).mean()),
                           "n_draws": len(d)})
    return pd.DataFrame(res), pd.DataFrame(paired)


def run_halving(result_dirs, out, draws=300, etas=(2, 3), strategies=("margin",)):
    import warnings
    warnings.filterwarnings("ignore")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs, _ = load(result_dirs)
    fams = set(runs.family.unique())
    if {"tabular", "domain"} <= fams:
        runs = pd.concat([runs, runs[runs.family.isin(["tabular", "domain"])].assign(
            family="tabular+domain")], ignore_index=True)
    runs = runs[runs.strategy.isin(strategies)]
    curves = _curves(result_dirs, strategies)
    LOG.info("halving: %d probe rows, %d trajectories", len(runs), len(curves))
    res, paired = halving(runs, curves, etas=etas, draws=draws)
    res.to_csv(out / "halving.csv", index=False)
    paired.to_csv(out / "halving_paired.csv", index=False)

    def md(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join(str(v) for v in r) + " |"
                            for r in df.itertuples(index=False)]) + "\n"
    R = ["# Successive halving vs early ranking (deploy-or-abandon, equal total budget)\n",
         f"Results: {', '.join(map(str, result_dirs))} · draws: {draws} · eta: {etas}\n",
         "Gains in U_dep over random funding after exploration to the primary probe. "
         "`budget_used` = labels spent / budget of the single-stage policy. "
         "Oracle reward = test AUPRC (not observable in practice).\n"]
    for fam in sorted(res.family.unique()) if len(res) else []:
        R.append(f"\n## Family: {fam}\n")
        r = res[res.family == fam].assign(
            v=lambda x: [f"{g:.3f} [{lo:.3f}; {hi:.3f}] ({u:.2f})" for g, lo, hi, u in
                         zip(x.dep_gain, x.dep_lo, x.dep_hi, x.budget_used)])
        R.append("\n### Deploy gain over random [95%] (budget used), rows: start, learner, policy\n")
        R.append(md(r.pivot_table(index=["start", "learner", "policy"], columns="fraction",
                                  values="v", aggfunc="first").reset_index()))
        p = paired[paired.family == fam]
        if len(p):
            p = p.assign(v=[f"{d:+.3f} [{lo:+.3f}; {hi:+.3f}]" for d, lo, hi in
                            zip(p.dep_diff, p.dep_lo, p.dep_hi)])
            R.append("\n### Paired differences a − b [95%], rows: start, learner, a, b\n")
            R.append(md(p.pivot_table(index=["start", "learner", "a", "b"], columns="fraction",
                                      values="v", aggfunc="first").reset_index()))
    (out / "halving_report.md").write_text("\n".join(R), encoding="utf-8")
    LOG.info("halving report written to %s", out / "halving_report.md")
    return out
