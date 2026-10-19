"""Resumable parallel runner: one JSON file per (task, seed, start)."""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import time
import traceback
from pathlib import Path

from sepcv import __version__
from sepcv.config import config_hash
from sepcv.data import discover_tasks
from sepcv.engine import run_unit

LOG = logging.getLogger("sepcv")


def unit_path(out_dir, task_id, seed, start):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", task_id)
    return Path(out_dir) / "units" / f"{safe}__s{seed}__{start}.json"


def _job(spec, seed, start, cfg, path):
    try:
        res = run_unit(spec, seed, start, cfg,
                       with_references=(seed == 0 and start == cfg["protocol"]["starts"][0]))
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, path)
        return None
    except Exception as exc:                                   # noqa: BLE001
        return {"task_id": spec["task_id"], "seed": seed, "start": start,
                "error": repr(exc), "trace": traceback.format_exc()[-1500:]}


def plan(cfg):
    specs, skipped = discover_tasks(cfg)
    pc = cfg["protocol"]
    jobs = [(sp, s, st) for s in range(pc["seeds"]) for st in pc["starts"] for sp in specs]
    return specs, skipped, jobs


def run(cfg, dry=False):
    from joblib import Parallel, delayed

    out = Path(cfg["out_dir"])
    (out / "units").mkdir(parents=True, exist_ok=True)
    specs, skipped, jobs = plan(cfg)
    todo = [(sp, s, st) for sp, s, st in jobs if not unit_path(out, sp["task_id"], s, st).exists()]
    manifest = {"version": __version__, "config_hash": config_hash(cfg), "config": cfg,
                "n_tasks": len(specs), "n_units": len(jobs), "skipped_tasks": skipped,
                "tasks": [{k: v for k, v in sp.items()} for sp in specs],
                "python": platform.python_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    old = out / "run_manifest.json"
    if old.exists():
        prev = json.load(open(old))
        if prev.get("config_hash") != manifest["config_hash"]:
            raise RuntimeError(
                f"{out} already holds results of a different configuration "
                f"({prev.get('config_hash')} vs {manifest['config_hash']}); use another out_dir.")
        if todo and not dry:           # resuming: record the version that completes the run
            prev.setdefault("resumed", []).append(
                {"version": __version__, "started": manifest["started"], "n_todo": len(todo)})
            with open(old, "w") as f:
                json.dump(prev, f, indent=1, default=str)
        manifest = prev                # never overwrite the provenance of existing results
    elif not dry:
        with open(old, "w") as f:
            json.dump(manifest, f, indent=1, default=str)
    LOG.info("%d tasks, %d units, %d to run (%d image tasks skipped)",
             len(specs), len(jobs), len(todo), len(skipped))
    if dry or not todo:
        return manifest
    errs = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(
        delayed(_job)(sp, s, st, cfg, unit_path(out, sp["task_id"], s, st)) for sp, s, st in todo)
    errs = [e for e in errs if e]
    if errs:
        with open(out / "errors.jsonl", "a") as f:
            for e in errs:
                f.write(json.dumps(e) + "\n")
        LOG.warning("%d units failed -- see %s", len(errs), out / "errors.jsonl")
    return manifest
