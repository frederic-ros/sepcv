"""Regenerate the data-driven figures of the paper from the analysis reports.

Usage:  python scripts/paper_figures.py <reports_dir> <out_dir>

<reports_dir> is the folder written by ``reproduce.py`` (or ``sepcv analyse/budget``):

    reports/controlled/        sepcv analyse results/controlled
    reports/tabular/           sepcv analyse results/rare27 results/domains_v2
    reports/images/            sepcv analyse results/images_eswa_tasks
    reports/budget_tabular/    sepcv budget  results/rare27 results/domains_v2

Figures written (file names as included by the LaTeX source):
    fig_questions.pdf                    Fig. 1
    fig_controlled.pdf                   Fig. 2   (copied from reports/controlled)
    fig_rarity_vs_sepcv.pdf              Fig. 3
    fig_budget.pdf                       Fig. 4
    fig_availability_ir.pdf              Fig. E.8
    fig_base_target_tabular_domain.pdf   Fig. E.9
Figures A.5, B.6 and C.7 do not depend on the experiments; see fig_variance.py,
fig_bayes.py and fig_geometries.py.

Common setting unless stated: real tabular tasks (tabular + domain), warm start, margin
sampling, primary probe (round 3 = 54 labels).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

TAB, START, STRAT, PROBE = "tabular+domain", "warm", "margin", 3
LEARNERS = ["logreg", "rf", "gbdt", "knn"]
LNAME = {"logreg": "LR", "rf": "RF", "gbdt": "GBDT", "knn": "$k$-NN"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
TINT = ["#eaf2fc", "#fdeee7", "#e6f6ef", "#eeecfb"]
LCOL = {"logreg": "#2a78d6", "rf": "#eb6834", "gbdt": "#008300", "knn": "#e5334b"}


def sel(df, fam=TAB, start=START, strat=STRAT, **kw):
    m = (df.family == fam) & (df.start == start) & (df.strategy == strat)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m]


# ----------------------------------------------------------------------- Fig. 1
def fig_questions(R, out):
    var = pd.read_csv(R / "controlled" / "controlled_variance.csv")
    v = var[(var.start == START) & (var.strategy == STRAT) & (var.learner == "rf")
            & (var.outcome == "final AUPRC")].iloc[0]
    avail = sel(pd.read_csv(R / "tabular" / "availability_by_ir.csv"))
    stab = sel(pd.read_csv(R / "tabular" / "decision_stability.csv"), round=PROBE)
    paired = sel(pd.read_csv(R / "budget_tabular" / "budget_paired.csv"),
                 a="SepCV+", b="Rarity, no exploration", fraction=0.7)
    rel = sel(pd.read_csv(R / "budget_tabular" / "budget_reliability.csv"), learner="rf")
    n0 = dict(zip(rel["round"], rel.n0))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    W, H = 8.2, 6.9
    fig = plt.figure(figsize=(W, H))
    ax0 = fig.add_axes([0, 0, 1, 1]); ax0.set_xlim(0, W); ax0.set_ylim(0, H); ax0.axis("off")
    cw, ch, gx, gy, x0, y0 = 3.85, 3.25, 0.3, 0.2, 0.05, 0.05
    POS = [(x0, y0 + ch + gy), (x0 + cw + gx, y0 + ch + gy), (x0, y0), (x0 + cw + gx, y0)]
    # Card texts summarize the results sections of the paper.
    cards = [
        ("1", "Is the task learnable?", "difficulty under the protocol",
         "Set mainly by separability; rarity explains ~10%.\n"
         "The imbalance ratio does not forecast it on new sources."),
        ("2", "Can it be measured yet?", "observability: $n_+\\geq 3$",
         "Delayed by rarity, start and learner: at IR$\\geq$50,\n"
         "5–16% of LR-margin runs vs 78–90% of RF-margin runs."),
        ("3", "Can we act on one estimate?", "reliability of a single score",
         "Computable from 3 positives, but screening decisions\n"
         "flip in 24% of runs (mean); 5% from 9 positives on."),
        ("4", "Is it worth the labels?", "value of the prediction",
         "Pays only if many tasks can be funded, and only after\n"
         "a short exploration (46–62 of 190 labels)."),
    ]
    for i, (num, q, sub, ans) in enumerate(cards):
        x, y = POS[i]
        ax0.add_patch(FancyBboxPatch((x, y), cw, ch, boxstyle="round,pad=0,rounding_size=0.12",
                                     fc=TINT[i], ec="none"))
        ax0.add_patch(FancyBboxPatch((x, y + ch - 0.07), cw, 0.07, boxstyle="square,pad=0",
                                     fc=COL[i], ec="none"))
        ax0.add_patch(Circle((x + 0.32, y + ch - 0.42), 0.2, fc=COL[i], ec="none"))
        ax0.text(x + 0.32, y + ch - 0.42, num, ha="center", va="center", color="white",
                 fontsize=10, fontweight="bold")
        ax0.text(x + 0.62, y + ch - 0.34, q, ha="left", va="center", color=INK, fontsize=9.4,
                 fontweight="bold")
        ax0.text(x + 0.62, y + ch - 0.58, sub, ha="left", va="center", color=INK2, fontsize=7.6,
                 style="italic")
        ax0.text(x + 0.18, y + 0.12, ans, ha="left", va="bottom", color=INK, fontsize=7.4,
                 linespacing=1.35)
    for (xa, ya) in [(x0 + cw + 0.03, y0 + ch + gy + ch / 2), (x0 + cw + 0.03, y0 + ch / 2)]:
        ax0.add_patch(Polygon([[xa, ya + 0.2], [xa + gx - 0.06, ya], [xa, ya - 0.2]], closed=True,
                              fc="#c3c2b7", ec="none"))

    def inset(i, left=0.6):
        x, y = POS[i]
        a = fig.add_axes([(x + left) / W, (y + 0.95) / H, (cw - left - 0.35) / W, 1.45 / H])
        a.set_facecolor("none")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(length=2, labelsize=6.5)
        return a

    # 1: share of variance of the final AUPRC (controlled factorial, RF, warm, margin)
    a = inset(0, left=0.95)
    lab, val = ["separation", "geometry", "rarity (IR)"], [v["sep"], v["geometry"], v["ir"]]
    a.barh(range(3)[::-1], val, height=0.5, color=COL[0], edgecolor="none")
    for k, x in enumerate(val):
        a.text(x + 0.02, 2 - k, f"{x:.2f}", va="center", fontsize=6.8, color=INK)
    a.set_yticks(range(3)[::-1], lab); a.set_xlim(0, 0.85); a.set_xticks([0, 0.4, 0.8])
    a.set_xlabel("share of variance of final AUPRC", fontsize=6.5, color=INK2)
    a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)

    # 2: share of runs with the diagnostic defined at 54 labels, by IR stratum
    a = inset(1)
    strata = ["<10", "10-50", "50-200", ">=200"]
    xs = np.arange(4)
    get = lambda L: [avail[(avail.learner == L) & (avail.ir_stratum == s)].defined_r3.iloc[0]  # noqa: E731
                     for s in strata]
    lr, rf = get("logreg"), get("rf")
    a.plot(xs, rf, color=COL[1], lw=1.8, marker="o", ms=4, mec="white", mew=0.8)
    a.plot(xs, lr, color=INK2, lw=1.4, ls="--", marker="o", ms=3.5, mec="white", mew=0.8)
    a.text(3.08, rf[3], "RF", va="center", fontsize=6.8, color=INK)
    a.text(3.08, lr[3], "LR", va="center", fontsize=6.8, color=INK)
    a.set_xticks(xs, ["<10", "10–50", "50–200", "≥200"]); a.set_xlim(-0.2, 3.6)
    a.set_ylim(0, 1.05); a.set_yticks([0, 0.5, 1]); a.set_yticklabels(["0", "50%", "100%"])
    a.set_xlabel("imbalance ratio", fontsize=6.5, color=INK2)
    a.set_ylabel("defined at 54 labels", fontsize=6.5, color=INK2)

    # 3: screening decisions that flip, by discovered positives (mean and range over learners)
    a = inset(2)
    bins = ["3", "4", "5", "6-8", "9+"]
    g = stab.assign(n_pos_bin=stab.n_pos_bin.astype(str)).groupby("n_pos_bin").screen_flip.agg(["mean", "min", "max"]).reindex(bins)
    flip, lo, hi = g["mean"].values, g["min"].values, g["max"].values
    a.bar(range(5), flip, width=0.55, color=COL[2], edgecolor="none")
    a.errorbar(range(5), flip, yerr=[flip - lo, hi - flip], fmt="none", ecolor=INK2,
               elinewidth=0.7, capsize=1.8)
    a.set_xticks(range(5), ["3", "4", "5", "6–8", "≥9"]); a.set_ylim(0, 0.4)
    a.set_yticks([0, 0.2, 0.4]); a.set_yticklabels(["0", "20%", "40%"])
    a.set_xlabel("rare cases discovered", fontsize=6.5, color=INK2)
    a.set_ylabel("decisions that flip", fontsize=6.5, color=INK2)

    # 4: net value vs exploration budget (SepCV+ minus exploration-free allocation, f = 0.7)
    a = inset(3)
    curves = {}
    for L in ("rf", "knn"):
        q = paired[paired.learner == L].sort_values("round")
        curves[L] = ([n0[r] for r in q["round"]], q.dep_diff.values, q.dep_lo.values)
    xs_rf, rfv, rflo = curves["rf"]
    sig = [x for x, lo_ in zip(xs_rf, rflo) if lo_ > 0]
    if sig:
        a.axvspan(min(sig) - 2, max(sig) + 2, color=COL[3], alpha=0.10, lw=0)
    a.axhline(0, color=MUTED, lw=0.6)
    a.plot(curves["knn"][0], curves["knn"][1], color=INK2, lw=1.4, ls="--", marker="o", ms=3.5,
           mec="white", mew=0.8)
    a.plot(xs_rf, rfv, color=COL[3], lw=1.8, marker="o", ms=4, mec="white", mew=0.8)
    a.text(max(xs_rf) + 2, rfv[-1], "RF", va="center", fontsize=6.8, color=INK)
    a.text(max(xs_rf) + 2, curves["knn"][1][-1], "k-NN", va="center", fontsize=6.8, color=INK)
    a.set_xlim(32, 142); a.set_xticks([38, 54, 78, 102, 126]); a.set_ylim(-0.02, 0.085)
    a.set_yticks([0, 0.04, 0.08])
    a.set_xlabel("exploration budget (labels per task)", fontsize=6.5, color=INK2)
    a.set_ylabel("net value vs. no exploration", fontsize=6.5, color=INK2)
    a.text(36, 0.08, "70% of tasks funded", fontsize=6.3, color=INK2, va="center")
    fig.savefig(out / "fig_questions.pdf")
    plt.close(fig)
    plt.rcdefaults()


# ----------------------------------------------------------------------- Fig. 3
def fig_rarity_vs_sepcv(R, out):
    def get(sub, fam):
        t = pd.read_csv(R / sub / "task_level.csv")
        return sel(t, fam=fam, learner="rf", round=PROBE).copy()
    tab, img = get("tabular", TAB), get("images", "image")
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#9a9993", "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                         "font.family": "serif", "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

    def scat(a, df, x, col, mk, lab=None, hollow=None):
        d = df if hollow is None else df[~hollow]
        a.scatter(d[x], d.final_auprc, s=30, marker=mk, facecolor=col, edgecolor="white",
                  linewidth=0.6, label=lab, zorder=3)
        if hollow is not None and hollow.any():
            h = df[hollow]
            a.scatter(h[x], h.final_auprc, s=30, marker=mk, facecolor="none", edgecolor=col,
                      linewidth=1.1, zorder=3)

    for df, col, mk, lab in ((tab, BLUE, "o", f"Real tabular ({len(tab)})"),
                             (img, ORANGE, "^", f"Image embeddings ({len(img)})")):
        scat(ax[0], df, "ir_eval", col, mk, lab)
    for df, col, mk in ((tab, BLUE, "o"), (img, ORANGE, "^")):
        scat(ax[1], df, "sepcv_plus", col, mk, hollow=(df.def_sepcv_plus < 0.5).values)
    r1 = spearmanr(tab.ir_eval, tab.final_auprc)[0]
    el, eli = tab[tab.def_sepcv_plus >= 0.5], img[img.def_sepcv_plus >= 0.5]
    r2, r3 = spearmanr(el.sepcv_plus, el.final_auprc)[0], spearmanr(eli.sepcv_plus, eli.final_auprc)[0]
    # correlations in the panel titles, away from the data
    ax[0].set_title(f"(a) Rarity\n$\\rho$ = {r1:+.2f} (tabular tasks)", loc="left", fontsize=10.5,
                    color=INK)
    ax[1].set_title(f"(b) Early separability\n$\\rho$ = {r2:.2f} (tabular), {r3:.2f} (images)",
                    loc="left", fontsize=10.5, color=INK)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Imbalance ratio IR (log scale)")
    ax[1].set_xlabel("SepCV$^+$ after 54 of 190 labels")
    ax[0].set_ylabel("Final rare-class AUPRC (RF)")
    # the two tasks of similar rarity, labelled with leader lines into empty space
    offsets = {("protein_homo", 0): (22, -62), ("protein_homo", 1): (-20, -80),
               ("abalone_19", 0): (18, 40), ("abalone_19", 1): (-62, 52)}
    for name, lbl in (("tabular|protein_homo", "protein_homo"), ("tabular|abalone_19", "abalone_19")):
        rw = tab[tab.task_id == name]
        if rw.empty:
            continue
        rw = rw.iloc[0]
        for k, (a, x) in enumerate(((ax[0], rw.ir_eval), (ax[1], rw.sepcv_plus))):
            a.scatter([x], [rw.final_auprc], s=80, facecolor="none", edgecolor=INK, linewidth=1.0,
                      zorder=4)
            a.annotate(lbl, (x, rw.final_auprc), xytext=offsets[(lbl, k)], textcoords="offset points",
                       fontsize=9, color=INK, fontfamily="monospace", zorder=5,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c3c2b7", lw=0.5),
                       arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkB=5))
    for a in ax:
        a.set_ylim(-0.03, 1.05); a.grid(axis="y", color=GRID, linewidth=0.5); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[1].axvline(0.5, color="#c3c2b7", lw=0.6, ls=(0, (3, 3)), zorder=1)
    h, l = ax[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK2,
                    markersize=6))
    l.append("SepCV$^+$ defined in < half of the runs")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig_rarity_vs_sepcv.pdf")
    plt.close(fig)
    plt.rcdefaults()


# ----------------------------------------------------------------------- Fig. 4
def fig_budget(R, out):
    B = R / "budget_tabular"
    rel = sel(pd.read_csv(B / "budget_reliability.csv"))
    pr = sel(pd.read_csv(B / "budget_paired.csv"), a="SepCV+", b="Rarity, no exploration")
    be = sel(pd.read_csv(B / "budget_break_even.csv"))
    n0 = rel.groupby("round").n0.median().to_dict()
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.0), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    a = ax[0]
    for L in LEARNERS:
        q = rel[rel.learner == L].sort_values("round")
        a.plot(q.n0, q.rho_task, color=LCOL[L], marker="o", ms=3, lw=1.8, label=LNAME[L])
        a.plot(q.n0, q.defined_runs, color=LCOL[L], ls=":", lw=1.4)
    a.set_ylim(0.4, 1.0); a.set_xlabel("exploration budget $n_0$ (labels)")
    a.set_ylabel("solid: $\\rho$(SepCV$^+$, final AUPRC)\ndotted: share of runs defined", fontsize=7)
    a.set_title("(a) reliability and availability", fontsize=9); a.legend(fontsize=7)
    a = ax[1]
    q = pr[pr.learner == "rf"]
    rounds = sorted(q["round"].unique()); fr = sorted(q.fraction.unique())
    M = np.full((len(rounds), len(fr)), np.nan); S = np.zeros_like(M, bool)
    for _, r in q.iterrows():
        i, j = rounds.index(r["round"]), fr.index(r.fraction)
        M[i, j], S[i, j] = r.dep_diff, (r.dep_lo > 0) or (r.dep_hi < 0)
    im = a.imshow(M, origin="lower", cmap="RdBu", vmin=-0.15, vmax=0.15, aspect="auto")
    for i in range(len(rounds)):
        for j in range(len(fr)):
            if not np.isnan(M[i, j]):
                a.text(j, i, f"{100 * M[i, j]:+.0f}" + ("*" if S[i, j] else ""), ha="center",
                       va="center", fontsize=6, color="white" if abs(M[i, j]) > 0.09 else INK)
    a.set_xticks(range(len(fr)), [f"{f:.1f}" for f in fr], fontsize=7)
    a.set_yticks(range(len(rounds)), [str(int(n0[r])) for r in rounds], fontsize=7)
    a.set_xlabel("funded fraction $f$"); a.set_ylabel("$n_0$")
    a.set_title("(b) SepCV$^+$ $-$ no exploration, RF ($U_{dep}\\times100$)", fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    a = ax[2]
    for L in LEARNERS:
        qb = be[be.learner == L].sort_values("round")
        x = [n0[r] for r in qb["round"]]
        if L == "logreg":
            a.plot(x, qb.break_even_f, color=LCOL[L], marker="o", ms=3, ls=":", lw=0.9,
                   label="LR (unstable)")
        else:
            a.plot(x, qb.break_even_f, color=LCOL[L], marker="o", ms=3, lw=1.8, label=LNAME[L])
    a.set_ylim(0.2, 1.0); a.set_xlabel("exploration budget $n_0$ (labels)")
    a.set_ylabel("break-even funded fraction"); a.set_title("(c) when exploring pays", fontsize=9)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_budget.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------- Fig. E.8
def fig_availability_ir(R, out):
    av = pd.read_csv(R / "tabular" / "availability_by_ir.csv")
    labels = [38, 46, 54, 62]
    cols = {"<10": "#1baf7a", "10-50": "#6250d6", "50-200": "#eb6834", ">=200": "#e5334b"}
    names = {"<10": "IR <10", "10-50": "IR 10–50", "50-200": "IR 50–200", ">=200": "IR ≥200"}
    fig, ax = plt.subplots(2, 2, figsize=(5.2, 3.8), sharex=True, sharey=True)
    for i, start in enumerate(("warm", "cold")):
        for j, L in enumerate(("logreg", "rf")):
            a = ax[i, j]
            q = sel(av, start=start, learner=L)
            for s, c in cols.items():
                r = q[q.ir_stratum == s]
                if r.empty:
                    continue
                r = r.iloc[0]
                a.plot(labels, [r.defined_r1, r.defined_r2, r.defined_r3, r.defined_r4], color=c,
                       marker="o", ms=2.5, lw=1.2, label=f"{names[s]} ({int(r.n_tasks)})")
            a.set_title(f"{start} start, {LNAME[L]}-margin", fontsize=7.5)
            a.tick_params(labelsize=6.5); a.set_ylim(-0.03, 1.03)
            if j == 0:
                a.set_ylabel("share of runs with SepCV defined", fontsize=6.5)
            if i == 1:
                a.set_xlabel("labels acquired", fontsize=6.5)
    ax[0, 0].legend(fontsize=5.5, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "fig_availability_ir.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------- Fig. E.9
def fig_base_target(R, out):
    s = sel(pd.read_csv(R / "tabular" / "association.csv"), round=PROBE, population="matched")
    rows = [("sepcv_plus", "SepCV$^+$"), ("sepcv_ens", "bases mean, no shrinkage"),
            ("sepcv_shr_lr", "LR base, shrunk"), ("sepcv_logreg_auroc", "SepCV-LR"),
            ("sepcv_rf_auroc", "SepCV-RF"), ("sepcv_knn_auroc", "SepCV-kNN")]
    lcols = ["logreg", "rf", "gbdt", "knn"]
    M = np.array([[s[(s.predictor == p) & (s.learner == L)].rho.iloc[0] for L in lcols]
                  for p, _ in rows])
    fig, a = plt.subplots(figsize=(4.0, 2.6))
    im = a.imshow(M, cmap="viridis", vmin=0.5, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                   color="white" if M[i, j] < 0.7 else INK)
    a.set_xticks(range(4), ["LR", "RF", "GBDT", "kNN"], fontsize=6.5)
    a.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=6.5)
    a.set_xlabel("downstream learner", fontsize=7)
    fig.colorbar(im, ax=a, fraction=0.05, pad=0.03).ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_base_target_tabular_domain.pdf")
    plt.close(fig)


def main(reports, out):
    R, out = Path(reports), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_questions(R, out)
    shutil.copy(R / "controlled" / "fig_controlled.pdf", out / "fig_controlled.pdf")
    fig_rarity_vs_sepcv(R, out)
    fig_budget(R, out)
    fig_availability_ir(R, out)
    fig_base_target(R, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
