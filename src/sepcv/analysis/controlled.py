"""Controlled factorial check on synthetic tasks (rarity x separation x geometry x d).

The design is a full, balanced factorial, so main effects are orthogonal and the
share of variance of a task-level outcome explained by each factor (eta^2) is
read directly from the level means. Outcomes:
  * final AUPRC (attainable performance under the protocol);
  * observability = share of runs in which SepCV-LR is defined at the probe.
The analysis also asks whether early estimators track the attainable level
overall, within a fixed IR (rarity carries no information there) and per geometry.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from sepcv.analysis.stats import cluster_bootstrap, rho

FACTORS = ["geometry", "ir", "sep", "d"]
TRACK = ["sepcv_plus", "sepcv_ens", "sepcv_logreg_auroc", "sepcv_rf_auroc", "sepcv_knn_auroc", "holdout_ap", "rare_frac",
         "neg_log_ir"]
_PAT = re.compile(r"(?P<geometry>[a-z]+)_ir(?P<ir>[\d.]+)_sep(?P<sep>[\d.]+)_d(?P<d>\d+)")


def parse(runs):
    r = runs[runs.family == "synthetic"].copy()
    if r.empty:
        return r
    f = r.dataset.str.extract(_PAT)
    r["geometry"] = f.geometry
    r["ir"] = f.ir.astype(float)
    r["sep"] = f.sep.astype(float)
    r["d"] = f.d.astype(int)
    return r


def task_means(r, round_):
    q = r[r["round"] == round_].copy()
    q["defined"] = q["sepcv_logreg_auroc"].notna().astype(float)
    keys = ["start", "strategy", "learner", "task_id"] + FACTORS
    agg = {"final_auprc": "mean", "defined": "mean", "n_pos": "mean"}
    agg.update({p: "mean" for p in TRACK if p in q})
    t = q.groupby(keys).agg(agg).reset_index()
    t["defined_runs"] = t["defined"]
    for p in TRACK:                     # own population: estimator defined in >= 50% of seeds
        if p in q:
            share = q.groupby(keys)[p].apply(lambda s: s.notna().mean()).reset_index(drop=True)
            t.loc[share.values < 0.5, p] = np.nan
    return t


def eta2(t, y):
    """Main-effect and two-way shares of variance in a balanced factorial."""
    v = t[y].astype(float)
    m, ss = v.mean(), ((v - v.mean()) ** 2).sum()
    out = {}
    if ss <= 0:
        return out
    main = {}
    for f in FACTORS:
        if t[f].nunique() > 1:
            g = t.groupby(f)[y]
            main[f] = float((g.count() * (g.mean() - m) ** 2).sum() / ss)
            out[f] = main[f]
    for a, b in (("ir", "sep"), ("geometry", "sep"), ("geometry", "ir")):
        if a in main and b in main:
            g = t.groupby([a, b])[y]
            cell = float((g.count() * (g.mean() - m) ** 2).sum() / ss)
            out[f"{a}x{b}"] = max(cell - main[a] - main[b], 0.0)
    out["residual"] = max(1.0 - sum(out.values()), 0.0)
    return out


def variance_shares(t):
    rows = []
    for (start, strat, L), q in t.groupby(["start", "strategy", "learner"]):
        for y, lab in (("final_auprc", "final AUPRC"), ("defined_runs", "SepCV-LR defined at probe")):
            rows.append({"start": start, "strategy": strat, "learner": L, "outcome": lab,
                         **{k: round(v, 3) for k, v in eta2(q, y).items()}})
    return pd.DataFrame(rows)


def by_level(t, factor):
    g = t.groupby(["start", "strategy", "learner", factor]).agg(
        final_auprc=("final_auprc", "mean"), defined=("defined_runs", "mean"),
        n_pos=("n_pos", "mean")).reset_index()
    return g


def tracking(t, boot=2000):
    rows = []
    for (start, strat, L), q in t.groupby(["start", "strategy", "learner"]):
        for p in TRACK:
            if p not in q:
                continue
            s = q[q[p].notna()]
            if len(s) < 6:
                continue
            x, y, grp = s[p].values, s.final_auprc.values, s.task_id.values
            r_all = rho(x, y)
            lo, hi = cluster_bootstrap(grp, lambda i: rho(x[i], y[i]), B=boot) if boot else (np.nan, np.nan)
            within = [(len(z), rho(z[p], z.final_auprc)) for _, z in s.groupby("ir") if len(z) >= 6]
            within = [(n, v) for n, v in within if v == v]
            w = (sum(n * v for n, v in within) / sum(n for n, _ in within)) if within else np.nan
            row = {"start": start, "strategy": strat, "learner": L, "predictor": p,
                   "n_tasks": len(s), "rho_all": r_all, "lo": lo, "hi": hi, "rho_within_ir": w}
            for geo, z in s.groupby("geometry"):
                row[f"rho_{geo}"] = rho(z[p], z.final_auprc) if len(z) >= 6 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def figure(t, out, learner="rf"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                          # noqa: BLE001
        return
    q = t[(t.start == "warm") & (t.strategy == "margin") & (t.learner == learner)]
    if q.empty:
        return
    geos = sorted(q.geometry.unique())
    fig, axes = plt.subplots(2, len(geos), figsize=(3.3 * len(geos), 5.2), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for j, geo in enumerate(geos):
        z = q[q.geometry == geo]
        for ir, w in z.groupby("ir"):
            m = w.groupby("sep")[["final_auprc", "defined_runs"]].mean()
            axes[0, j].plot(m.index, m.final_auprc, marker="o", label=f"IR {ir:g}")
            axes[1, j].plot(m.index, m.defined_runs, marker="o", label=f"IR {ir:g}")
        axes[0, j].set_title(geo)
        axes[1, j].set_xlabel("separation $s$")
    axes[0, 0].set_ylabel(f"final AUPRC ({learner})")
    axes[1, 0].set_ylabel("SepCV-LR defined at probe")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_controlled.pdf")
    plt.close(fig)


def analyse_controlled(runs, out, primary_round, boot=2000):
    """Returns markdown lines (empty when there is no synthetic family)."""
    r = parse(runs)
    if r.empty:
        return []
    t = task_means(r, primary_round)
    vs = variance_shares(t)
    lev_ir, lev_sep = by_level(t, "ir"), by_level(t, "sep")
    tr = tracking(t, boot=boot)
    t.to_csv(out / "controlled_tasks.csv", index=False)
    vs.to_csv(out / "controlled_variance.csv", index=False)
    lev_ir.to_csv(out / "controlled_by_ir.csv", index=False)
    lev_sep.to_csv(out / "controlled_by_sep.csv", index=False)
    tr.to_csv(out / "controlled_tracking.csv", index=False)
    figure(t, out)

    def md(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join("–" if (isinstance(v, float) and v != v) else str(v)
                                    for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join(lines) + "\n"

    R = ["\n## Controlled factorial check (synthetic)\n",
         f"Tasks: {t.task_id.nunique()} · factors: " +
         ", ".join(f"{f}={sorted(t[f].unique().tolist())}" for f in FACTORS) + "\n"]
    R.append("\n### Share of variance (eta², balanced factorial, task means, margin)\n")
    R.append(md(vs[vs.strategy == "margin"].drop(columns="strategy").round(3)))
    for name, lev, f in (("IR", lev_ir, "ir"), ("separation", lev_sep, "sep")):
        R.append(f"\n### Means by {name} level (margin; averaged over the other factors)\n")
        q = lev[lev.strategy == "margin"].copy()
        q["v"] = [f"{a:.3f} · {b:.2f} · {c:.1f}" for a, b, c in zip(q.final_auprc, q.defined, q.n_pos)]
        R.append("Cells: final AUPRC · share of runs with SepCV-LR defined at probe · positives at probe\n\n")
        R.append(md(q.pivot_table(index=["start", "learner"], columns=f, values="v",
                                  aggfunc="first").reset_index()))
    R.append(f"\n### Tracking the attainable level (Spearman with final AUPRC, round {primary_round}, "
             "own population)\n")
    R.append("rho_within_ir = task-weighted mean of the correlation computed within each IR level "
             "(rarity carries no information there).\n\n")
    q = tr[tr.strategy == "margin"].copy()
    R.append(md(q.drop(columns="strategy").round(2)))
    return R
