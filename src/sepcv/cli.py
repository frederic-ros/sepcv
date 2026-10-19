"""Command line: python -m sepcv {check,run,analyse,smoke} ..."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path


def _setup_logging(out_dir=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(out_dir) / "sepcv.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)


def cmd_check(args):
    from sepcv.config import load_config
    from sepcv.runner import plan
    cfg = load_config(args.config)
    _setup_logging()
    specs, skipped, jobs = plan(cfg)
    by = {}
    for s in specs:
        by.setdefault(s["family"], []).append(s)
    for fam, ss in by.items():
        print(f"{fam:10s} {len(ss):5d} tasks  ({len({s['group'] for s in ss})} groups)")
    pc = cfg["protocol"]
    n_traj = sum(len(pc["learners"]) if s not in ("random", "coreset") else 1 for s in pc["strategies"])
    print(f"units: {len(jobs)}  trajectories per unit: {n_traj}  skipped image tasks: {len(skipped)}")
    if args.details:
        from sepcv.data import count_missing, load_npz_xy
        print("\n# tasks: task_id | group | domain | missing values in the stored features")
        for s in specs:
            miss = ""
            if s.get("path") and s["kind"] not in ("image", "synthetic"):
                try:
                    miss = count_missing(load_npz_xy(s["path"])[0])
                except Exception as e:                       # noqa: BLE001
                    miss = f"? ({e.__class__.__name__})"
            print(f"  {s['task_id']} | {s['group']} | {s.get('domain')} | {miss}")
        if skipped:
            print("\n# skipped (positives available < min_rare)")
            for k in skipped:
                print(f"  {k['task_id']}  n_pos_possible={k['n_pos_possible']}  min_rare={k['min_rare']}")
    if args.time and specs:
        from sepcv.engine import run_unit
        t = time.time()
        run_unit(specs[0], 0, pc["starts"][0], cfg)
        dt = time.time() - t
        print(f"one unit of '{specs[0]['task_id']}': {dt:.1f}s -> rough total "
              f"{dt * len(jobs) / max(cfg['n_jobs'], 1) / 3600:.1f} h with n_jobs={cfg['n_jobs']}")


def cmd_run(args):
    from sepcv.config import load_config
    from sepcv.runner import run
    cfg = load_config(args.config)
    if args.n_jobs:
        cfg["n_jobs"] = args.n_jobs
    _setup_logging(cfg["out_dir"])
    run(cfg, dry=args.dry)


def cmd_analyse(args):
    from sepcv.analysis.report import analyse
    _setup_logging()
    analyse([Path(r) for r in args.results], Path(args.out), boot=args.boot)


def cmd_reprobe(args):
    from sepcv.config import load_config
    from sepcv.reprobe import reprobe
    cfg = load_config(args.config)
    if args.n_jobs:
        cfg["n_jobs"] = args.n_jobs
    _setup_logging(cfg["out_dir"])
    reprobe(cfg, rounds=tuple(args.rounds), strategies=tuple(args.strategies))


def cmd_budget(args):
    from sepcv.analysis.budget import run_budget
    _setup_logging()
    run_budget([Path(r) for r in args.results], Path(args.out), draws=args.draws)


def cmd_halving(args):
    from sepcv.analysis.halving import run_halving
    _setup_logging()
    run_halving([Path(r) for r in args.results], Path(args.out), draws=args.draws,
                etas=tuple(args.eta))


def cmd_smoke(args):
    """End-to-end check on synthetic data (a few minutes, no external files)."""
    from sepcv.analysis.report import analyse
    from sepcv.config import DEFAULTS, _merge
    from sepcv.runner import run
    tmp = Path(args.out or tempfile.mkdtemp(prefix="sepcv_smoke_"))
    cfg = _merge(DEFAULTS, {
        "name": "smoke", "out_dir": str(tmp / "results"), "n_jobs": args.n_jobs or 1,
        "sources": [{"kind": "synthetic", "geometries": ["gauss", "multimodal"],
                     "irs": [10, 40], "seps": [0.3, 1.0, 2.0, 3.5], "dims": [8], "n": 1500}],
        "protocol": {"seeds": 2, "starts": ["warm", "cold"], "rounds": 8,
                     "learners": ["logreg", "rf", "knn"], "strategies": ["margin", "random"]},
        "estimators": {"cv_repeats": {"logreg": 2, "knn": 2, "rf": 1}, "rf_trees": 30},
        "references": {"enabled": True, "bases": ["logreg", "knn"]},
    })
    _setup_logging(cfg["out_dir"])
    run(cfg)
    analyse([Path(cfg["out_dir"])], tmp / "report", boot=200)
    print(f"smoke OK -> {tmp}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sepcv")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="list tasks and units without running")
    c.add_argument("config"); c.add_argument("--time", action="store_true", help="time one unit")
    c.add_argument("--details", action="store_true", help="list every task, its group, and skipped tasks")
    c.set_defaults(func=cmd_check)
    r = sub.add_parser("run", help="run (resumable)")
    r.add_argument("config"); r.add_argument("--n-jobs", type=int); r.add_argument("--dry", action="store_true")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("analyse", help="analyse one or more result folders")
    a.add_argument("results", nargs="+"); a.add_argument("--out", required=True)
    a.add_argument("--boot", type=int, default=2000)
    a.set_defaults(func=cmd_analyse)
    q = sub.add_parser("reprobe", help="recompute estimators at extra rounds from stored trajectories")
    q.add_argument("config"); q.add_argument("--rounds", type=int, nargs="+", default=[6, 9, 12])
    q.add_argument("--strategies", nargs="+", default=["margin"]); q.add_argument("--n-jobs", type=int)
    q.set_defaults(func=cmd_reprobe)
    b = sub.add_parser("budget", help="value of early prediction vs exploration budget (no new runs)")
    b.add_argument("results", nargs="+"); b.add_argument("--out", required=True)
    b.add_argument("--draws", type=int, default=300)
    b.set_defaults(func=cmd_budget)
    h = sub.add_parser("halving", help="successive halving vs early ranking (no new runs)")
    h.add_argument("results", nargs="+"); h.add_argument("--out", required=True)
    h.add_argument("--draws", type=int, default=300)
    h.add_argument("--eta", type=int, nargs="+", default=[2, 3])
    h.set_defaults(func=cmd_halving)
    s = sub.add_parser("smoke", help="synthetic end-to-end check")
    s.add_argument("--out"); s.add_argument("--n-jobs", type=int)
    s.set_defaults(func=cmd_smoke)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
