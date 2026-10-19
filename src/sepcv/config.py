"""Configuration: JSON file merged over defaults, plus deterministic seeding."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

DEFAULTS = {
    "name": "run",
    "master_seed": 20261013,
    "out_dir": "results/run",
    "n_jobs": 2,
    "sources": [],
    # Task construction (tabular and domain datasets)
    "task": {
        "n_max": 10_000,        # pool cap; stratified subsampling preserves the IR
        "min_rare": 100,        # cap enlarged so that at least this many rare cases remain
        "test_fraction": 0.3,
        "standardize": True,
    },
    # Active-learning protocol (budget of the companion study by default)
    "protocol": {
        "seeds": 5,
        "starts": ["warm", "cold"],
        "n_init": 30,
        "batch": 8,
        "rounds": 20,
        "probe_rounds": [1, 2, 3, 4],
        "learners": ["logreg", "rf", "gbdt", "knn"],
        "strategies": ["margin", "random"],
        "tie_break": "random",
    },
    # Early estimators computed on the labeled set at each probe round
    "estimators": {
        "bases": ["logreg", "rf", "knn"],
        "metrics": ["auroc", "ap"],
        "min_positives": 3,
        "cv_folds_max": 3,
        "cv_repeats": {"logreg": 5, "knn": 5, "rf": 3, "gbdt": 3},
        "rf_trees": 100,
        "holdout": True,
        "complexity": True,
    },
    # Full-data separability references (validation only, never used for decisions)
    "references": {"enabled": True, "bases": ["logreg", "rf", "knn"], "max_n": 5000},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        user = json.load(f)
    cfg = _merge(DEFAULTS, user)
    cfg["_config_path"] = str(path.resolve())
    # relative paths in the config are resolved against the current directory
    return cfg


def config_hash(cfg: dict) -> str:
    keep = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("n_jobs", "out_dir")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from any sequence of printable parts."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF
