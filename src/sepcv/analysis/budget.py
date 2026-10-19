"""How reliability and cost jointly set the value of early prediction.

For each exploration budget n0 (the labels at probe rounds 1-4: 38, 46, 54, 62 by
default) the deploy-or-abandon allocation is re-run with the diagnostic measured
at that probe, on a fine grid of funded fractions f. A larger n0 makes the
diagnostic more available and more reliable, but every task pays n0 labels
before any is funded, while the exploration-free allocation spends the same total
budget on fully labelled tasks. The output is a value map (n0 x f) of the paired
difference SepCV+ - exploration-free allocation, the break-even fraction for each
n0, and the reliability of the diagnostic at each n0.

Usage: python -m sepcv budget <results...> --out <folder>
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sepcv.analysis import decision
from sepcv.analysis.load import PRIMARY, load
from sepcv.analysis.stats import rho

LOG = logging.getLogger("sepcv")
FRACTIONS = tuple(np.round(np.arange(0.1, 0.91, 0.1), 2))
POLICIES = {"SepCV+": PRIMARY, "Held-out AP": "holdout_ap", "Rarity (lower IR first)": "neg_log_ir"}


def reliability_by_round(runs, min_def=0.5):
    out = []
    for (fam, start, strat, L, rd), s in runs.groupby(["family", "start", "strategy", "learner", "round"]):
        g = s.groupby("task_id")
        share = g[PRIMARY].apply(lambda v: v.notna().mean())
        t = g[[PRIMARY, "final_auprc"]].mean()[share >= min_def]
        d = s.dropna(subset=[PRIMARY])
        out.append({"family": fam, "start": start, "strategy": strat, "learner": L, "round": rd,
                    "n0": float(s.n_lab.median()), "defined_runs": float(s[PRIMARY].notna().mean()),
                    "n_tasks_defined": int((share >= min_def).sum()),
                    "rho_task": rho(t[PRIMARY], t.final_auprc),
                    "rho_run": rho(d[PRIMARY], d.final_auprc)})
    return pd.DataFrame(out)


def break_even(paired):
    """Smallest funded fraction at which SepCV+ catches up with the exploration-free allocation."""
    p = paired[(paired.a == "SepCV+") & (paired.b == decision.NOEXP)]
    out = []
    for key, q in p.groupby(["family", "start", "strategy", "learner", "round"]):
        q = q.sort_values("fraction")
        f, d = q.fraction.values, q.dep_diff.values
        be = np.nan
        for i in range(len(f)):
            if d[i] >= 0:
                be = f[i] if i == 0 else f[i - 1] + (f[i] - f[i - 1]) * (-d[i - 1]) / (d[i] - d[i - 1])
                break
        sig = q[(q.dep_lo > 0)].fraction
        out.append(dict(zip(["family", "start", "strategy", "learner", "round"], key),
                        break_even_f=be, first_f_significant=float(sig.min()) if len(sig) else np.nan))
    return pd.DataFrame(out)


def figure(paired, rel, out, learner="rf", start="warm"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                          # noqa: BLE001
        return
    for fam in paired.family.unique():
        p = paired[(paired.family == fam) & (paired.a == "SepCV+") & (paired.b == decision.NOEXP)
                   & (paired.start == start) & (paired.strategy == "margin")]
        if p.empty:
            continue
        learners = sorted(p.learner.unique())
        fig, axes = plt.subplots(1, len(learners), figsize=(3.2 * len(learners), 3.0), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, L in zip(axes, learners):
            for rd, q in p[p.learner == L].groupby("round"):
                q = q.sort_values("fraction")
                n0 = rel[(rel.family == fam) & (rel.start == start) & (rel.strategy == "margin")
                         & (rel.learner == L) & (rel["round"] == rd)].n0
                lab = f"$n_0$={int(n0.iloc[0])}" if len(n0) else f"round {rd}"
                line, = ax.plot(q.fraction, q.dep_diff, marker="o", ms=3, label=lab)
                ax.fill_between(q.fraction, q.dep_lo, q.dep_hi, color=line.get_color(), alpha=0.12)
            ax.axhline(0, color="grey", lw=0.7)
            ax.set_title(L)
            ax.set_xlabel("funded fraction $f$")
        axes[0].set_ylabel("$U_{dep}$: SepCV+ $-$ no exploration")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"fig_budget_{fam.replace('+', '_')}_{start}.pdf")
        plt.close(fig)


def run_budget(result_dirs, out, draws=300, strategies=("margin",)):
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
    LOG.info("budget analysis: %d runs, rounds %s", len(runs), sorted(runs["round"].unique()))
    pol = {k: v for k, v in POLICIES.items() if v in runs}
    alloc, paired = decision.allocation(runs, fractions=FRACTIONS, draws=draws, policies=pol)
    rel = reliability_by_round(runs)
    be = break_even(paired)
    alloc.to_csv(out / "budget_allocation.csv", index=False)
    paired.to_csv(out / "budget_paired.csv", index=False)
    rel.to_csv(out / "budget_reliability.csv", index=False)
    be.to_csv(out / "budget_break_even.csv", index=False)
    for start in ("warm", "cold"):
        figure(paired, rel, out, start=start)

    def md(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join("–" if (isinstance(v, float) and v != v) else str(v)
                                              for v in r) + " |" for r in df.itertuples(index=False)]) + "\n"
    R = ["# Exploration budget: reliability, cost and decision value\n",
         f"Results: {', '.join(map(str, result_dirs))} · strategies: {', '.join(strategies)} · draws: {draws}\n"]
    for fam in sorted(rel.family.unique()):
        R.append(f"\n## Family: {fam}\n")
        R.append("\n### Diagnostic at each exploration budget (SepCV+)\n")
        R.append(md(rel[rel.family == fam].drop(columns=["family", "strategy"]).round(3)))
        R.append("\n### Break-even funded fraction (SepCV+ = exploration-free allocation) and first "
                 "fraction where SepCV+ is significantly ahead\n")
        R.append(md(be[be.family == fam].drop(columns=["family", "strategy"]).round(3)))
        p = paired[(paired.family == fam) & (paired.a == "SepCV+")]
        if len(p):
            p = p.assign(v=[f"{d:+.3f} [{lo:+.3f}; {hi:+.3f}]" for d, lo, hi in
                            zip(p.dep_diff, p.dep_lo, p.dep_hi)])
            for b in p.b.unique():
                R.append(f"\n### Paired U_dep difference: SepCV+ − {b} (rows: start, learner, round)\n")
                R.append(md(p[p.b == b].pivot_table(index=["start", "learner", "round"], columns="fraction",
                                                    values="v", aggfunc="first").reset_index()))
        a = alloc[(alloc.family == fam) & (alloc.policy.isin(["SepCV+", decision.NOEXP, "Oracle (final level)"]))]
        if len(a):
            R.append("\n### Absolute deploy gain over random funding (rows: start, learner, round, policy)\n")
            a = a.assign(v=[f"{g:.3f}" for g in a.dep_gain])
            R.append(md(a.pivot_table(index=["start", "learner", "round", "policy"], columns="fraction",
                                      values="v", aggfunc="first").reset_index()))
    (out / "budget_report.md").write_text("\n".join(R), encoding="utf-8")
    with open(out / "budget_meta.json", "w") as fh:
        json.dump({"results": list(map(str, result_dirs)), "draws": draws,
                   "fractions": list(map(float, FRACTIONS))}, fh, indent=1)
    LOG.info("budget report written to %s", out / "budget_report.md")
    return out
