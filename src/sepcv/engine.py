"""One unit = one (task, seed, start). All strategies and learners of a unit share
the same split and the same initial labeled set (paired design).

For every trajectory the unit records:
  * probe rows (estimators on L_t at each probe round);
  * the test AUPRC curve of the downstream learner(s);
  * the final AUPRC after the full budget;
  * the acquisition order (initial labels, then each batch), so any further
    estimator can be computed post hoc without re-running the trajectory.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score

from sepcv import estimators as E
from sepcv.config import stable_seed
from sepcv.data import materialize
from sepcv.learners import fit, pos_proba
from sepcv.strategies import LEARNER_FREE, select


def initial_labels(ytr, n_init, start, rng):
    lab = np.zeros(len(ytr), bool)
    if start == "warm":       # warm start: one rare, one majority, rest uniform
        lab[rng.choice(np.flatnonzero(ytr == 1))] = True
        lab[rng.choice(np.flatnonzero(ytr == 0))] = True
        lab[rng.choice(np.flatnonzero(~lab), n_init - 2, replace=False)] = True
    elif start == "cold":     # no guaranteed positive
        lab[rng.choice(len(ytr), n_init, replace=False)] = True
    else:
        raise ValueError(start)
    return lab


def _auprc(learner, seed, Xl, yl, Xte, yte):
    model, prior = fit(learner, seed, Xl, yl)
    return float(average_precision_score(yte, pos_proba(model, Xte))), prior


def trajectory(D, lab0, strategy, learner, cfg, unit_seed):
    Xtr, Xte, ytr, yte = D
    pc, ec = cfg["protocol"], cfg["estimators"]
    learners = pc["learners"]
    lab = lab0.copy()
    acq = [np.flatnonzero(lab0).tolist()]
    rng = np.random.default_rng(stable_seed(unit_seed, "acq", strategy, learner))
    probes, curve = [], {L: {} for L in ([learner] if learner else learners)}
    for rnd in range(pc["rounds"] + 1):
        lseed = stable_seed(unit_seed, "fit", rnd)
        p_pool = None
        pool = np.flatnonzero(~lab)
        if learner is not None:
            model, _ = fit(learner, lseed, Xtr[lab], ytr[lab])
            curve[learner][rnd] = float(average_precision_score(yte, pos_proba(model, Xte)))
            if rnd < pc["rounds"]:
                p_pool = pos_proba(model, Xtr[pool])
        elif rnd in pc["probe_rounds"] or rnd == pc["rounds"]:
            for L in learners:
                curve[L][rnd], _ = _auprc(L, lseed, Xtr[lab], ytr[lab], Xte, yte)
        if rnd in pc["probe_rounds"]:
            row = E.probe(Xtr[lab], ytr[lab], ec, stable_seed(unit_seed, "probe", rnd))
            probes.append({"round": rnd, **row})
        if rnd == pc["rounds"] or len(pool) == 0:
            break
        local = select(strategy, p_pool, Xtr[pool], Xtr[lab], pc["batch"], rng)
        picked = pool[local]
        lab[picked] = True
        acq.append(picked.tolist())
    final_round = max(curve[next(iter(curve))])
    return {"strategy": strategy, "learner": learner or "*",
            "probes": probes,
            "curve": {L: {str(k): v for k, v in c.items()} for L, c in curve.items()},
            "final": {L: c[final_round] for L, c in curve.items()},
            "n_pos_final": int(ytr[lab].sum()), "n_lab_final": int(lab.sum()),
            "acq": acq}


def run_unit(spec, seed_idx, start, cfg, with_references=False):
    t0 = time.time()
    master = cfg["master_seed"]
    data_seed = stable_seed(master, "data", spec["task_id"], seed_idx)
    Xtr, Xte, ytr, yte, info = materialize(spec, cfg, data_seed)
    unit_seed = stable_seed(master, "unit", spec["task_id"], seed_idx, start)
    rng = np.random.default_rng(stable_seed(master, "init", spec["task_id"], seed_idx, start))
    lab0 = initial_labels(ytr, cfg["protocol"]["n_init"], start, rng)
    D = (Xtr, Xte, ytr, yte)
    trajs = []
    for s in cfg["protocol"]["strategies"]:
        if s in LEARNER_FREE:
            trajs.append(trajectory(D, lab0, s, None, cfg, unit_seed))
        else:
            for L in cfg["protocol"]["learners"]:
                trajs.append(trajectory(D, lab0, s, L, cfg, unit_seed))
    meta = {k: v for k, v in spec.items() if k != "path"}
    meta.update(info, seed=seed_idx, start=start, n_rare_init=int(ytr[lab0].sum()))
    if with_references and cfg["references"]["enabled"]:
        rc = cfg["references"]
        for b in rc["bases"]:
            meta[f"sstar_{b}"] = E.reference(Xtr, ytr, b, data_seed, rc.get("max_n", 5000))
    meta["secs"] = round(time.time() - t0, 1)
    return {"meta": meta, "trajectories": trajs}
