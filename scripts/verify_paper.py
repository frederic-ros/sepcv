"""Check that regenerated reports reproduce the key numbers reported in the paper.

Usage:  python scripts/verify_paper.py <reports_dir>

Each check reads one report table and compares the value(s) with the number printed in the
paper, after rounding to the precision used in the text. The script prints one line per
check and exits with status 1 if any check fails. Settings: warm start, margin sampling,
primary probe (round 3, 54 labels), unless stated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEARNERS = ["logreg", "rf", "gbdt", "knn"]
TWO_PART = "SepCV+ + undefined flag (all runs)"


def sel(df, fam=None, start="warm", strat="margin", **kw):
    m = np.ones(len(df), bool)
    if fam is not None:
        m &= (df.family == fam).values
    if start is not None:
        m &= (df.start == start).values
    if strat is not None:
        m &= (df.strategy == strat).values
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m]


def by_learner(df, col, learners=LEARNERS):
    return [float(df[df.learner == L][col].iloc[0]) for L in learners]


def checks(R):
    c = []
    tab = "tabular+domain"

    def rd(*parts):
        return pd.read_csv(R.joinpath(*parts))

    # Claim 1 -- controlled factorial
    v = pd.read_csv(R / "controlled" / "controlled_variance.csv")
    v = v[(v.start == "warm") & (v.strategy == "margin") & (v.outcome == "final AUPRC")]
    c.append(("Sec. 6.1  variance share of separation, range over learners", "0.49-0.71",
              (v.sep.min(), v.sep.max()), (0.49, 0.71), 0.005))
    c.append(("Sec. 6.1  variance share of rarity, range over learners", "0.09-0.11",
              (v.ir.min(), v.ir.max()), (0.09, 0.11), 0.005))
    av = sel(pd.read_csv(R / "tabular" / "availability_by_ir.csv"), fam=tab)
    hi = av[av.ir_stratum.isin(["50-200", ">=200"])]
    lr = hi[hi.learner == "logreg"].defined_r3
    c.append(("Sec. 6.1  defined at 54 labels, IR>=50, LR-margin", "5-16%",
              (lr.min(), lr.max()), (0.05, 0.16), 0.005))

    # Claim 2 -- association and forecasting
    a = sel(pd.read_csv(R / "tabular" / "association.csv"), fam=tab, round=3,
            predictor="sepcv_plus", population="own")
    c.append(("Table 4   rho(SepCV+) LR/RF/GBDT/kNN", "0.78/0.80/0.77/0.87",
              by_learner(a, "rho"), [0.78, 0.80, 0.77, 0.87], 0.005))
    f = sel(pd.read_csv(R / "tabular" / "forecast_logo.csv"), fam=tab, round=3, set=TWO_PART,
            population="own")
    c.append(("Table 5   two-part forecaster R2 LR/RF/GBDT/kNN", "0.23/0.62/0.49/0.69",
              by_learner(f, "r2_task"), [0.23, 0.62, 0.49, 0.69], 0.005))
    ai = sel(pd.read_csv(R / "images" / "association.csv"), fam="image", round=3,
             predictor="sepcv_plus", population="own")
    r = by_learner(ai, "rho", ["rf", "gbdt", "knn"])
    c.append(("Sec. 6.2  images rho(SepCV+), RF/GBDT/kNN range", "0.83-0.91",
              (min(r), max(r)), (0.83, 0.91), 0.005))
    fi = sel(pd.read_csv(R / "images" / "forecast_logo.csv"), fam="image", round=3, set=TWO_PART,
             population="own")
    r = by_learner(fi, "r2_task", ["rf", "gbdt", "knn"])
    c.append(("Sec. 6.2  images two-part R2, RF/GBDT/kNN range", "0.58-0.68",
              (min(r), max(r)), (0.58, 0.68), 0.005))
    sy = sel(pd.read_csv(R / "controlled" / "association.csv"), fam="synthetic", round=3,
             predictor="sepcv_plus", population="own")
    r = by_learner(sy, "rho")
    c.append(("Sec. 6.2  synthetic rho(SepCV+), range over learners", "0.83-0.93",
              (min(r), max(r)), (0.83, 0.93), 0.005))

    # Claim 3 -- reliability
    d = sel(pd.read_csv(R / "tabular" / "decision_stability.csv"), fam=tab, round=3)
    d = d.assign(n_pos_bin=d.n_pos_bin.astype(str))
    f3, f9 = d[d.n_pos_bin == "3"].screen_flip, d[d.n_pos_bin == "9+"].screen_flip
    c.append(("Sec. 6.3  screening flips with 3 positives", "10-35%",
              (f3.min(), f3.max()), (0.10, 0.35), 0.005))
    c.append(("Sec. 6.3  screening flips from 9 positives, max", "<=13%",
              (f9.max(),), (0.13,), 0.005))

    # Claim 4 -- decisions
    al = sel(pd.read_csv(R / "tabular" / "allocation.csv"), fam=tab, round=3, learner="rf")
    vals = [float(al[(al.policy == "SepCV+") & (al.fraction == fr)].dep_gain.iloc[0])
            for fr in (0.2, 0.4, 0.6)]
    c.append(("Table 6   SepCV+ deploy gain, RF, f=0.2/0.4/0.6", "0.055/0.089/0.096",
              vals, [0.055, 0.089, 0.096], 0.0005))
    vals = [float(al[(al.policy == "Oracle (final level)") & (al.fraction == fr)].dep_gain.iloc[0])
            for fr in (0.2, 0.4, 0.6)]
    c.append(("Table 6   oracle deploy gain, RF, f=0.2/0.4/0.6", "0.077/0.126/0.133",
              vals, [0.077, 0.126, 0.133], 0.0005))
    bp = sel(pd.read_csv(R / "budget_tabular" / "budget_paired.csv"), fam=tab, learner="rf",
             a="SepCV+", b="Rarity, no exploration")
    best = bp.loc[bp.dep_diff.idxmax()]
    c.append(("Sec. 6.4  max RF gain over exploration-free (value, round, f)", "+0.066, 54 labels, 0.7",
              (best.dep_diff, best["round"], best.fraction), (0.066, 3, 0.7), 0.0005))
    sc = sel(pd.read_csv(R / "tabular" / "screening.csv"), fam=tab, round=3, tau=0.1, alpha=0.05,
             policy="SepCV+")
    c.append(("Sec. 6.4  screening recall of low-ceiling runs", "19-43%",
              (sc.low_recall.min(), sc.low_recall.max()), (0.19, 0.43), 0.005))
    c.append(("Sec. 6.4  realized false-discard rate", "4-6%",
              (sc.false_discard.min(), sc.false_discard.max()), (0.04, 0.06), 0.005))
    h = pd.read_csv(R / "halving_tabular" / "halving.csv")
    h = h[(h.family == tab) & (h.start == "warm") & (h.learner == "rf") & (h.fraction == 0.4)
          & (h.policy == "SepCV+ ranking (single stage)")]
    c.append(("Table E.10 single-stage SepCV+, RF, f=0.4", "0.088",
              (float(h.dep_gain.iloc[0]),), (0.088,), 0.0005))
    return c


def main(reports):
    R = Path(reports)
    fails = 0
    try:
        items = checks(R)
    except Exception as e:                                       # noqa: BLE001
        print(f"[FAIL] could not read the reports: {e.__class__.__name__}: {e}")
        sys.exit(1)
    for name, printed, got, exp, tol in items:
        got, exp = np.atleast_1d(np.asarray(got, float)), np.atleast_1d(np.asarray(exp, float))
        ok = np.all(np.abs(got - exp) <= tol + 1e-9)
        fails += not ok
        shown = "/".join(f"{g:.3f}" for g in got)
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: paper {printed}; reports {shown}")
    print(f"\n{'All checks passed.' if not fails else f'{fails} check(s) failed.'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main(sys.argv[1])
