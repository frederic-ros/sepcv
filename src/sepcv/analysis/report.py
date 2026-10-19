"""Full analysis: CSV tables for every condition + a readable report.md."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sepcv.analysis import ablation, controlled, decision, diagnostic, forecast, reliability
from sepcv.analysis.load import PRIMARY, SHRINK_K, load, predictors
from sepcv.analysis.stats import rho

LOG = logging.getLogger("sepcv")
CI_PREDS = ["sepcv_plus", "sepcv_ens", "sepcv_shr_lr", "sepcv_logreg_auroc", "sepcv_logreg_ap", "sepcv_rf_auroc", "sepcv_knn_auroc",
            "holdout_ap", "rare_frac", "obs_lce", "log_f1", "n3_rare", "neg_log_ir", "probe_oracle"]
LABEL = {"sepcv_plus": "SepCV+", "sepcv_ens": "SepCV bases mean (no shrinkage)",
         "sepcv_shr_lr": "SepCV-LR shrunk (no base mean)", "sepcv_logreg_auroc": "SepCV-LR AUROC", "sepcv_logreg_ap": "SepCV-LR AP",
         "sepcv_rf_auroc": "SepCV-RF AUROC", "sepcv_rf_ap": "SepCV-RF AP",
         "sepcv_knn_auroc": "SepCV-kNN AUROC", "sepcv_knn_ap": "SepCV-kNN AP",
         "holdout_ap": "Held-out AP", "rare_frac": "Labeled rare fraction",
         "obs_lce": "Observable LCE", "log_f1": "Fisher F1 (log)", "n3": "N3", "n3_rare": "N3 (rare)",
         "neg_log_ir": "Rarity (-log IR)", "probe_oracle": "Probe test AUPRC (oracle)"}


def _fmt(v, lo=None, hi=None, nd=2):
    if v is None or v != v:
        return "–"
    s = f"{v:.{nd}f}"
    if lo is not None and lo == lo and hi == hi:
        s += f" [{lo:.{nd}f}; {hi:.{nd}f}]"
    return s


def _md(df):
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _figures(assoc, out, primary_round):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                          # noqa: BLE001
        return
    for fam, a in assoc.groupby("family"):
        s = a[(a.population == "matched") & (a.start == "warm") & (a.strategy == "margin")]
        if s.empty:
            continue
        learners = sorted(s.learner.unique())
        fig, axes = plt.subplots(1, len(learners), figsize=(3.6 * len(learners), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, L in zip(axes, learners):
            for p in [PRIMARY, "sepcv_logreg_auroc", "holdout_ap", "rare_frac", "neg_log_ir"]:
                q = s[(s.learner == L) & (s.predictor == p)].sort_values("round")
                if len(q):
                    ax.plot(q["round"], q["rho"], marker="o", label=LABEL[p])
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(f"target: {L}")
            ax.set_xlabel("probe round")
        axes[0].set_ylabel("Spearman with final AUPRC")
        axes[-1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / f"fig_earliness_{fam.replace('+', '_')}.pdf")
        plt.close(fig)
        m = diagnostic.base_target_matrix(assoc[assoc.family == fam], primary_round)
        if len(m):
            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            im = ax.imshow(m.values, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(m.shape[1]), m.columns)
            ax.set_yticks(range(m.shape[0]), [LABEL.get(i, i) for i in m.index], fontsize=7)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if m.values[i, j] < 0.6 else "black")
            ax.set_xlabel("target learner")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out / f"fig_base_target_{fam.replace('+', '_')}.pdf")
            plt.close(fig)


def analyse(result_dirs, out, boot=2000, primary_round=None, draws=300):
    import warnings
    warnings.filterwarnings("ignore")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs, tasks = load(result_dirs)
    # post-hoc probes (sepcv reprobe) serve the exploration-budget analysis only
    runs = runs[~runs["extra"]].drop(columns="extra")
    fams = set(runs.family.unique())
    if {"tabular", "domain"} <= fams:
        runs = pd.concat([runs, runs[runs.family.isin(["tabular", "domain"])].assign(
            family="tabular+domain")], ignore_index=True)
    preds = predictors(runs)
    rounds = sorted(runs["round"].unique())
    primary_round = primary_round or (3 if 3 in rounds else rounds[len(rounds) // 2])
    LOG.info("%d runs, %d tasks, families=%s, primary round=%s",
             len(runs), runs.task_id.nunique(), sorted(runs.family.unique()), primary_round)

    task = diagnostic.task_level(runs, preds)
    assoc = diagnostic.association(task, preds, boot=boot, primary_round=primary_round,
                                   ci_preds=CI_PREDS)
    avail = diagnostic.availability(task)
    fc, fc_pair = forecast.forecast(runs, boot=boot, primary_round=primary_round)
    prim = runs[runs["round"] == primary_round]
    alloc, pdiff = decision.allocation(prim, draws=draws)
    screen = decision.screening(prim)
    pdelta = diagnostic.paired_delta(task, preds, round_=primary_round, boot=boot)
    jack = diagnostic.jackknife(task, [PRIMARY, "sepcv_logreg_auroc", "neg_log_ir"], round_=primary_round)
    robm = diagnostic.robust_matched(task, round_=primary_round)
    av_ir = reliability.availability_by_ir(runs)
    sserr = reliability.sstar_error(runs, tasks, col="sepcv_logreg_auroc")
    stab = reliability.decision_stability(prim)
    calsz = decision.screening_calsize(prim)
    sens = ablation.sensitivity(runs, primary_round)
    fails = ablation.failures(task, primary_round)
    for name, df in (("forecast_paired", fc_pair), ("assoc_paired_delta", pdelta), ("jackknife", jack),
                     ("robust_matched", robm), ("availability_by_ir", av_ir), ("sstar_error", sserr),
                     ("decision_stability", stab), ("screening_calsize", calsz),
                     ("sensitivity_k", sens), ("failures", fails)):
        df.to_csv(out / f"{name}.csv", index=False)

    relia = reliability.by_positives(prim)
    nextb = reliability.next_batch(runs)
    relia.to_csv(out / "reliability_by_positives.csv", index=False)
    nextb.to_csv(out / "next_batch.csv", index=False)
    task.to_csv(out / "task_level.csv", index=False)
    assoc.to_csv(out / "association.csv", index=False)
    avail.to_csv(out / "availability.csv", index=False)
    fc.to_csv(out / "forecast_logo.csv", index=False)
    alloc.to_csv(out / "allocation.csv", index=False)
    pdiff.to_csv(out / "allocation_paired.csv", index=False)
    screen.to_csv(out / "screening.csv", index=False)
    tasks.to_csv(out / "tasks.csv", index=False)
    _figures(assoc, out, primary_round)

    # ------------------------------------------------------------------ report
    R = [f"# SepCV analysis\n\nResults: {', '.join(map(str, result_dirs))}  \n"
         f"Runs: {len(runs)} · tasks: {runs.task_id.nunique()} · primary probe round: {primary_round}  \n"
         "Association = Spearman between task-mean estimator and task-mean final AUPRC, on the "
         "matched population (tasks where SepCV+ and held-out AP are defined in ≥50% of "
         "seeds); 95% CI = bootstrap over groups.  \n"
         f"Primary diagnostic: SepCV+ = 0.5 + n+/(n+ + {SHRINK_K}) × (mean of the LR, RF, k-NN "
         "CV-AUROC bases − 0.5); defined exactly when SepCV-LR is (n+ ≥ 3).\n"]
    for fam in sorted(runs.family.unique()):
        R.append(f"\n## Family: {fam}\n")
        a = assoc[(assoc.family == fam) & (assoc.population == "matched")]
        R.append("### Availability of SepCV+ / SepCV-LR (tasks defined in ≥50% of seeds)\n")
        av = avail[(avail.family == fam) & (avail.strategy == "margin")]
        if len(av):
            R.append(_md(av.pivot_table(index=["start", "learner"], columns="round",
                                        values="n_defined").reset_index()))
        for start in sorted(a.start.unique()):
            s = a[(a.start == start) & (a.strategy == "margin") & (a["round"] == primary_round)]
            if s.empty:
                continue
            R.append(f"\n### Association, margin, start={start}, round {primary_round}\n")
            tab = {}
            for _, r in s.iterrows():
                tab.setdefault(LABEL.get(r.predictor, r.predictor), {})[r.learner] = \
                    _fmt(r.rho, r.get("lo"), r.get("hi"))
            n = s.groupby("learner").n_tasks.max().to_dict()
            df = pd.DataFrame(tab).T
            df.columns = [f"{c} (n={n.get(c)})" for c in df.columns]
            R.append(_md(df.reset_index().rename(columns={"index": "estimator"})))
            own = assoc[(assoc.family == fam) & (assoc.population == "own") & (assoc.start == start)
                        & (assoc.strategy == "margin") & (assoc["round"] == primary_round)
                        & (assoc.predictor == PRIMARY)]
            if len(own):
                R.append("\nSepCV+ on its own population (defined from 3 positives): " + ", ".join(
                    f"{r.learner} ρ={_fmt(r.rho)} (n={r.n_tasks})" for _, r in own.iterrows()) + "\n")
            pdl = pdelta[(pdelta.family == fam) & (pdelta.start == start) & (pdelta.strategy == "margin")] \
                if len(pdelta) else pdelta
            if len(pdl):
                R.append("\nPaired Δρ = ρ(SepCV+) − ρ(other), cluster bootstrap, same tasks:\n\n")
                R.append(_md(pdl.assign(other=pdl.other.map(lambda x: LABEL.get(x, x)),
                                        v=[_fmt(d, lo, hi) + f" (n={n})" for d, lo, hi, n in
                                           zip(pdl.delta_rho, pdl.lo, pdl.hi, pdl.n_tasks)])
                             .pivot_table(index="other", columns="learner", values="v", aggfunc="first")
                             .reset_index()))
            jk = jack[(jack.family == fam) & (jack.start == start) & (jack.strategy == "margin")] \
                if len(jack) else jack
            if len(jk):
                R.append("\nLeave-one-group-out range of ρ (influence of single sources):\n\n")
                R.append(_md(jk[["learner", "predictor", "rho_min", "rho_max"]].round(2)))
        st = a[(a.predictor == PRIMARY) & (a["round"] == primary_round)]
        if st.strategy.nunique() > 1:
            R.append("\n### Robustness to the acquisition strategy (SepCV+)\n")
            R.append(_md(st.pivot_table(index=["start", "strategy"], columns="learner",
                                        values="rho").round(2).reset_index()))
        rm = robm[robm.family == fam] if len(robm) else robm
        if len(rm):
            R.append("\nSame comparison on tasks eligible under BOTH margin and the other strategy:\n\n")
            R.append(_md(rm.round(2)))
        R.append(f"\n### Base × target learner (warm, margin, round {primary_round})\n")
        m = diagnostic.base_target_matrix(assoc[assoc.family == fam], primary_round)
        R.append(_md(m.round(2).rename(index=LABEL).reset_index()))
        f = fc[(fc.family == fam) & (fc.population == "common") & (fc["round"] == primary_round)
               & (fc.strategy == "margin")] if len(fc) else fc
        if len(f) == 0:
            R.append("\n_Grouped forecasting and decision analyses need at least 5 groups (and 6 tasks) "
                     "in this family._\n")
        if len(f):
            R.append(f"\n### Grouped leave-one-group-out forecasting (margin, round {primary_round}), task-level R²\n")
            tab = {}
            for _, r in f.iterrows():
                tab.setdefault(r["set"], {})[f"{r.start}/{r.learner}"] = \
                    _fmt(r.r2_task, r.get("r2_task_lo"), r.get("r2_task_hi"))
            R.append(_md(pd.DataFrame(tab).T.reset_index().rename(columns={"index": "predictor set"})))
            cov = f[f["set"] == forecast.PRIMARY_SET][["start", "learner", "coverage_common", "n_tasks"]]
            R.append("\nCoverage (share of runs with a defined forecast):\n\n" + _md(cov.round(2)))
            fo = fc[(fc.family == fam) & (fc.population == "own") & (fc["round"] == primary_round)
                    & (fc.strategy == "margin") & fc["set"].isin(
                        [forecast.PRIMARY_SET, "SepCV-LR (AUROC)", forecast.TWO_PART,
                         "Undefined flag alone (all runs)", "Rarity, all runs"])]
            if len(fo):
                R.append("\nOwn population (diagnostic alone, n+≥3) and ALL runs (two-part forecaster, "
                         "flag alone, rarity):\n\n")
                fo = fo.assign(r2=[_fmt(v, lo, hi) for v, lo, hi in
                                   zip(fo.r2_task, fo.get("r2_task_lo", np.nan), fo.get("r2_task_hi", np.nan))])
                R.append(_md(fo[["start", "learner", "set", "coverage_runs", "n_tasks", "r2",
                                 "rho_task"]].round(3)))
            fp = fc_pair[(fc_pair.family == fam) & (fc_pair.strategy == "margin")] if len(fc_pair) else fc_pair
            if len(fp):
                R.append("\nPaired ΔR² (task level) = R²(a) − R²(b), same tasks (population in brackets):\n\n")
                R.append(_md(fp.assign(pair=fp.base + " − " + fp.other + " [" + fp.population + "]",
                                       v=[_fmt(d, lo, hi) + f" (n={n})" for d, lo, hi, n in
                                          zip(fp.delta_r2, fp.lo, fp.hi, fp.n_tasks)])
                             .pivot_table(index="pair", columns=["start", "learner"], values="v",
                                          aggfunc="first").reset_index()))
        al = alloc[(alloc.family == fam) & (alloc.strategy == "margin")] if len(alloc) else alloc
        if len(al):
            R.append(f"\n### Budget allocation (margin, round {primary_round}, portfolio of "
                     f"{int(al.portfolio.iloc[0])} of {int(al.n_tasks.iloc[0])} tasks per draw)\n\n"
                     "Cells: deploy objective gain over random [2.5–97.5%] · deploy oracle recovery │ "
                     "improvement objective oracle recovery │ recall of tasks reaching AUPRC ≥ 0.5\n")
            al = al.assign(v=[f"{_fmt(dg, dlo, dhi, 3)} · {_fmt(dr)} │ {_fmt(ir)} │ {_fmt(ur)}"
                              for dg, dlo, dhi, dr, ir, ur in
                              zip(al.get("dep_gain"), al.get("dep_lo"), al.get("dep_hi"),
                                  al.get("dep_recovery"), al.get("imp_recovery"), al.useful_recall)])
            for start in sorted(al.start.unique()):
                q = al[al.start == start]
                R.append(f"\nstart={start}\n\n")
                R.append(_md(q.pivot_table(index=["learner", "policy"], columns="fraction",
                                           values="v", aggfunc="first").reset_index()))
            pd_ = pdiff[(pdiff.family == fam) & (pdiff.strategy == "margin")] if len(pdiff) else pdiff
            if len(pd_):
                R.append("\nPaired differences (same draws): deploy Δ [CI] (P a better) │ improvement Δ [CI]\n\n")
                pd_ = pd_.assign(pair=pd_.a + " − " + pd_.b,
                                 v=[f"{_fmt(d, lo, hi, 3)} (p={p:.2f}) │ {_fmt(di, li, hi2, 3)}"
                                    for d, lo, hi, p, di, li, hi2 in
                                    zip(pd_.get("dep_diff"), pd_.get("dep_lo"), pd_.get("dep_hi"),
                                        pd_.get("dep_p_a_better"), pd_.get("imp_diff"),
                                        pd_.get("imp_lo"), pd_.get("imp_hi"))])
                R.append(_md(pd_.pivot_table(index=["start", "learner", "pair"], columns="fraction",
                                             values="v", aggfunc="first").reset_index()))
        sc = screen[(screen.family == fam) & (screen.strategy == "margin") & (screen.tau == 0.1)
                    & (screen.get("alpha", 0.05) == 0.05)] if len(screen) and "family" in screen else screen
        if len(sc) and "policy" in sc:
            R.append("\n### Calibrated screening (low ceiling = final AUPRC < 0.10, target false-discard 5%, LOGO)\n")
            R.append(_md(sc[["start", "learner", "policy", "n_low", "n_low_tasks", "n_low_groups",
                             "low_recall", "missed_low_undefined", "false_discard",
                             "discard_rate", "auprc_lost_share"]].round(3)))
            R.append("\n_Rarity policies use the true pool IR, unknown to an active learner (optimistic)._\n")
            sa = screen[(screen.family == fam) & (screen.strategy == "margin") & (screen.tau == 0.1)
                        & (screen.policy == "SepCV+")]
            if len(sa):
                R.append("\nSepCV+ screening, sensitivity to the target false-discard rate α (τ = 0.10):\n\n")
                R.append(_md(sa[["start", "learner", "alpha", "low_recall", "false_discard",
                                 "auprc_lost_share"]].round(3)))
            cz = calsz[(calsz.family == fam) & (calsz.strategy == "margin")] if len(calsz) else calsz
            if len(cz):
                R.append("\nSepCV+ screening calibrated on k random other groups (α = 5%, τ = 0.10; "
                         "mean over repetitions, p90 of the realized false-discard rate):\n\n")
                R.append(_md(cz[["start", "learner", "cal_groups", "reps", "low_recall", "false_discard",
                                 "false_discard_p90"]].round(3)))
        rl = relia[(relia.family == fam) & (relia.strategy == "margin")] if len(relia) else relia
        if len(rl):
            R.append(f"\n### Reliability of a single estimate by discovered positives (SepCV+, margin, round {primary_round})\n")
            R.append(_md(rl[["start", "learner", "n_pos_bin", "n_runs", "n_tasks", "rho_run",
                             "sd_within_task", "median_side_agreement"]].round(3)))
        ds = stab[(stab.family == fam) & (stab.strategy == "margin")] if len(stab) else stab
        if len(ds):
            R.append("\n### Decision stability of a single estimate by discovered positives (SepCV+, margin)\n")
            R.append("rank_stability / rank_accuracy: share of run pairs (different tasks, same bin) "
                     "ordered like the task-mean score / the task-mean final AUPRC; screen_flip: share "
                     "of runs whose screening decision differs from the task-mean decision.\n\n")
            R.append(_md(ds[["start", "learner", "n_pos_bin", "n_runs", "n_tasks", "rank_stability",
                             "rank_accuracy", "screen_flip"]].round(3)))
        sk = sens[(sens.family == fam) & (sens.strategy.isin(["margin", "random"]))] if len(sens) else sens
        if len(sk):
            R.append(f"\n### SepCV+ components: base(s) × shrinkage constant k (Spearman with final AUPRC, round {primary_round})\n")
            R.append("LR = SepCV-LR base only; mean3 = mean of the LR, RF and k-NN bases; k = 0: no shrinkage; "
                     f"SepCV+ = mean3 k={SHRINK_K}.\n\n")
            R.append(_md(sk.drop(columns="family").round(2)))
        fl = fails[(fails.family == fam) & (fails.learner == "rf")] if len(fails) else fails
        if len(fl):
            R.append("\n### Failure cases (warm, margin, RF target; all learners in failures.csv)\n")
            R.append(_md(fl.drop(columns=["family", "learner"])))
        nb = nextb[(nextb.family == fam)] if len(nextb) else nextb
        if len(nb):
            R.append("\n### Intra-task frontier: SepCV+ vs next-batch AUPRC gain (pooled over probe rounds)\n")
            R.append(_md(nb.round(3)))
        ai = av_ir[(av_ir.family == fam) & (av_ir.strategy == "margin")] if len(av_ir) else av_ir
        if len(ai):
            R.append("\n### Observability: share of runs with SepCV+ (= SepCV-LR) defined, by IR stratum (margin)\n")
            R.append(_md(ai.drop(columns=["family", "strategy"]).round(2)))
        se = sserr[(sserr.family == fam) & (sserr["round"] == primary_round)] if len(sserr) else sserr
        if len(se):
            R.append("\n### Diagnostic error vs full-data reference S* (LR), by discovered positives and strategy\n")
            R.append(_md(se.drop(columns=["family", "round"]).round(3)))
        ref = tasks[tasks.family.isin(["tabular", "domain"])] if fam == "tabular+domain" \
            else tasks[tasks.family == fam]
        sstar = [c for c in ref.columns if c.startswith("sstar_")]
        tl = task[(task.family == fam) & (task.start == "warm") & (task.strategy == "margin")
                  & (task["round"] == primary_round)]
        if sstar and len(tl):
            R.append("\n### Full-data references S* (validation only) vs final AUPRC, warm margin\n")
            fin = tl.pivot_table(index="task_id", columns="learner", values="final_auprc")
            rr = ref.set_index("task_id")[sstar].reindex(fin.index)
            R.append(_md(pd.DataFrame({s: {L: round(rho(rr[s], fin[L]), 2) for L in fin.columns}
                                       for s in sstar}).T.reset_index()))
    R += controlled.analyse_controlled(runs, out, primary_round, boot=boot)
    (out / "report.md").write_text("\n".join(R), encoding="utf-8")
    with open(out / "analysis_meta.json", "w") as fh:
        json.dump({"results": list(map(str, result_dirs)), "primary_round": int(primary_round),
                   "boot": boot, "draws": draws}, fh, indent=1)
    LOG.info("report written to %s", out / "report.md")
    return out
