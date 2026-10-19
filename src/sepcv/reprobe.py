"""Post-hoc probes at extra rounds, from the stored acquisition order (no new trajectories).

Every unit file stores, for each trajectory, the initial labeled set and each
acquired batch. The labeled set at round r is therefore known exactly, and the
early estimators can be recomputed at rounds that were not probed during the run.
The task data are rebuilt with the same seeds; a consistency check compares the
number of rare cases at the original probe rounds with the stored values and skips
the unit on any mismatch.

Extra probe rows are written to <out_dir>/units_extra/<unit>.json and merged by the
loader. Only learner-specific trajectories (margin, most-likely-positive) have a
test-AUPRC curve at every round, so only they are re-probed.

Usage: python -m sepcv reprobe <config> --rounds 6 9 12 [--n-jobs 8]
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path

import numpy as np

from sepcv import estimators as E
from sepcv.config import stable_seed
from sepcv.data import materialize
from sepcv.runner import plan, unit_path

LOG = logging.getLogger("sepcv")


def extra_path(out_dir, task_id, seed, start):
    p = unit_path(out_dir, task_id, seed, start)
    return p.parent.parent / "units_extra" / p.name


def _labeled_at(acq, rnd):
    idx = list(acq[0])
    for b in acq[1:rnd + 1]:
        idx += list(b)
    return np.array(idx, int)


def reprobe_unit(spec, seed_idx, start, cfg, rounds, strategies):
    master = cfg["master_seed"]
    data_seed = stable_seed(master, "data", spec["task_id"], seed_idx)
    Xtr, Xte, ytr, yte, info = materialize(spec, cfg, data_seed)
    unit_seed = stable_seed(master, "unit", spec["task_id"], seed_idx, start)
    with open(unit_path(cfg["out_dir"], spec["task_id"], seed_idx, start)) as f:
        u = json.load(f)
    out = []
    for tr in u["trajectories"]:
        if tr["strategy"] not in strategies or tr["learner"] == "*":
            continue
        acq = tr["acq"]
        for p in tr["probes"]:                                   # consistency check
            lab = _labeled_at(acq, p["round"])
            if int(ytr[lab].sum()) != p["n_pos"] or len(lab) != p["n_lab"]:
                raise RuntimeError(f"replay mismatch at round {p['round']} ({tr['strategy']}/{tr['learner']})")
        rows = []
        for rnd in rounds:
            if rnd >= len(acq):
                continue
            lab = _labeled_at(acq, rnd)
            row = E.probe(Xtr[lab], ytr[lab], cfg["estimators"], stable_seed(unit_seed, "probe", rnd))
            rows.append({"round": int(rnd), **row})
        out.append({"strategy": tr["strategy"], "learner": tr["learner"], "probes": rows})
    return {"task_id": spec["task_id"], "seed": seed_idx, "start": start, "rounds": list(rounds),
            "trajectories": out}


def _job(spec, seed, start, cfg, rounds, strategies, path):
    try:
        res = reprobe_unit(spec, seed, start, cfg, rounds, strategies)
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, path)
        return None
    except Exception as exc:                                   # noqa: BLE001
        return {"task_id": spec["task_id"], "seed": seed, "start": start,
                "error": repr(exc), "trace": traceback.format_exc()[-1500:]}


def reprobe(cfg, rounds=(6, 9, 12), strategies=("margin",)):
    from joblib import Parallel, delayed
    out = Path(cfg["out_dir"])
    (out / "units_extra").mkdir(parents=True, exist_ok=True)
    _, _, jobs = plan(cfg)
    todo = [(sp, s, st) for sp, s, st in jobs
            if unit_path(out, sp["task_id"], s, st).exists()
            and not extra_path(out, sp["task_id"], s, st).exists()]
    LOG.info("re-probing %d units at rounds %s (%s)", len(todo), list(rounds), ", ".join(strategies))
    errs = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(
        delayed(_job)(sp, s, st, cfg, rounds, strategies, extra_path(out, sp["task_id"], s, st))
        for sp, s, st in todo)
    errs = [e for e in errs if e]
    if errs:
        with open(out / "reprobe_errors.jsonl", "a") as f:
            for e in errs:
                f.write(json.dumps(e) + "\n")
        LOG.warning("%d units failed -- see %s", len(errs), out / "reprobe_errors.jsonl")
    return len(todo) - len(errs)
