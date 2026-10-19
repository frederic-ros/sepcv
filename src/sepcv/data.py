"""Data sources and task construction.

Three source kinds, all producing *task specs* (small picklable dicts) that are
materialized inside worker processes:

* ``tabular``     -- a folder of binary datasets (.xlsx with ``feature_*`` +
                     ``target_binary`` columns as in the Rare27 export, .csv, or
                     .npz with X/y). Converted once to .npz in a local cache.
* ``eswa_cache``  -- the prepared cache of the companion study, Ros et al. (``data/cache/<id>.npz``
                     + ``manifest.json``); already IR-preserving, temporal splits
                     (``is_test``) are honoured.
* ``embeddings``  -- frozen image embeddings ``<root>/<dataset>/<backbone>.npz``
                     (keys X|embeddings and y|labels, multiclass y). Each class
                     becomes a one-vs-rest rare task at a target IR, which yields a
                     *portfolio* of tasks per dataset.

Rarity is always preserved: tabular pools larger than ``n_max`` are subsampled
by stratified sampling at constant IR, enlarged to keep ``min_rare`` rare cases.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

import numpy as np

LOG = logging.getLogger("sepcv")


# --------------------------------------------------------------------------- #
# Low-level readers
# --------------------------------------------------------------------------- #
def _read_table(path: Path):
    import pandas as pd
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    feats = [c for c in df.columns if str(c).startswith("feature_")]
    if "target_binary" in df.columns:
        y = df["target_binary"].to_numpy()
    else:                                    # fall back: last column is the target
        y = df.iloc[:, -1].to_numpy()
    if not feats:
        feats = [c for c in df.columns if c not in ("target_binary", "target_original", df.columns[-1])]
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = np.asarray(y)
    if set(np.unique(y)) - {0, 1}:          # {-1, 1} or labels: minority value -> 1
        vals, cnt = np.unique(y, return_counts=True)
        y = (y == vals[np.argmin(cnt)]).astype(int)
    return X, y.astype(int)


def load_npz_xy(path):
    z = np.load(path, allow_pickle=False)
    X = z["X"] if "X" in z else z["embeddings"]
    y = z["y"] if "y" in z else z["labels"]
    extra = {k: z[k] for k in ("t", "is_test") if k in z}
    return np.asarray(X, np.float32 if X.dtype == np.float32 else float), np.asarray(y), extra


def count_missing(X):
    return int(np.isnan(np.asarray(X, float)).sum())


def _impute(Xtr, Xte):
    """Median imputation fitted on the training part only, then applied to both parts."""
    Xtr, Xte = np.asarray(Xtr, float).copy(), np.asarray(Xte, float).copy()
    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        med = np.nanmedian(Xtr, axis=0) if len(Xtr) else np.zeros(Xtr.shape[1])
        med = np.where(np.isnan(med), 0.0, med)
        for A in (Xtr, Xte):
            r, c = np.where(np.isnan(A))
            A[r, c] = med[c]
    return Xtr, Xte


# --------------------------------------------------------------------------- #
# Discovery: config sources -> task specs
# --------------------------------------------------------------------------- #
def _select(names, wanted):
    if wanted in (None, "all"):
        return sorted(names)
    missing = [w for w in wanted if w not in names]
    for m in missing:
        LOG.warning("dataset %s not found -- skipped", m)
    return [w for w in wanted if w in names]


def _tabular_specs(src, cache_dir):
    root = Path(src["path"])
    files = {}
    for ext in ("*.npz", "*.xlsx", "*.csv"):
        for f in sorted(root.glob(ext)):
            if f.stem.lower() in ("datasets_summary", "readme"):
                continue
            files.setdefault(f.stem, f)
    specs = []
    cache = Path(cache_dir) / "tabular"
    cache.mkdir(parents=True, exist_ok=True)
    for name in _select(list(files), src.get("datasets", "all")):
        f = files[name]
        if f.suffix == ".npz":
            npz = f
        else:
            npz = cache / f"{name}.npz"
            if not npz.exists():
                LOG.info("converting %s -> %s", f.name, npz)
                X, y = _read_table(f)
                np.savez_compressed(npz, X=X, y=y)
        _, y, _ = load_npz_xy(npz)
        y = (np.asarray(y) == 1).astype(int)
        n_rare = int((y == 1).sum())
        specs.append(dict(kind="tabular", family=src.get("family", "tabular"),
                          task_id=f"{src.get('family', 'tabular')}|{name}", dataset=name,
                          group=src.get("group_prefix", "") + name, domain=src.get("domain", "uci"),
                          path=str(npz), ir_nominal=(len(y) - n_rare) / max(n_rare, 1),
                          subsample=src.get("subsample", True)))
    return specs


# Datasets that derive from one source share a group (LOGO and bootstrap units).
SOURCE_PATTERNS = [(r"^polish_bankruptcy_y\d+$", "polish_bankruptcy"),
                   (r"^unsw_nb15_.+$", "unsw_nb15"),
                   (r"^cicids2017_.+$", "cicids2017")]


def source_group(name, extra_patterns=()):
    import re
    for pat, grp in list(extra_patterns) + SOURCE_PATTERNS:
        if re.match(pat, name):
            return grp
    return name


def _eswa_specs(src):
    root = Path(src["path"])
    mpath = Path(src.get("manifest", root / "manifest.json"))
    manifest = {}
    if mpath.exists():
        raw = json.load(open(mpath, encoding="utf-8"))
        if isinstance(raw, dict) and "datasets" in raw:
            raw = raw["datasets"]
        if isinstance(raw, list):
            raw = {r.get("id", r.get("dataset")): r for r in raw}
        manifest = raw
    excl = tuple(src.get("exclude_prefix", []))
    names = [p.stem for p in root.glob("*.npz") if not (excl and p.stem.startswith(excl))]
    specs = []
    for name in _select(names, src.get("datasets", "all")):
        m = manifest.get(name, {})
        specs.append(dict(kind="tabular", family=src.get("family", "domain"),
                          task_id=f"{src.get('family', 'domain')}|{name}", dataset=name,
                          group=(m.get(src["group_field"]) if src.get("group_field") else None)
                          or source_group(name, [tuple(x) for x in src.get("group_patterns", [])]),
                          domain=m.get("domain", "unknown"),
                          path=str(root / f"{name}.npz"), ir_nominal=m.get("ir_nominal"),
                          subsample=src.get("subsample", False)))
    return specs


def _embedding_specs(src):
    root = Path(src["path"])
    backbones = src.get("backbones", ["dinov2_vits14", "resnet18"])
    entries = src.get("datasets", "all")
    if entries == "all":
        entries = sorted(p.name for p in root.iterdir() if p.is_dir())
    specs, skipped = [], []
    for e in entries:
        e = {"dataset": e} if isinstance(e, str) else dict(e)
        name = e["dataset"]
        irs = e.get("target_irs", src.get("target_irs", [99]))
        max_pool = e.get("max_pool", src.get("max_pool", 10_000))
        min_rare = e.get("min_rare", src.get("min_rare", 50))
        rarity = e.get("rarity", src.get("rarity", "one_vs_rest"))
        for bb in e.get("backbones", backbones):
            f = root / name / f"{bb}.npz"
            if not f.exists():
                LOG.warning("embedding %s missing -- skipped", f)
                continue
            y = np.load(f, allow_pickle=False)
            y = y["y"] if "y" in y else y["labels"]
            classes, counts = np.unique(y, return_counts=True)
            base = dict(kind="image", family=src.get("family", "image"), dataset=name,
                        group=name, domain=e.get("domain", "image"), backbone=bb,
                        path=str(f), max_pool=max_pool, min_rare=min_rare)
            if rarity == "natural":
                specs.append({**base, "task_id": f"image|{name}|{bb}|natural", "rarity": "natural",
                              "positive_class": None, "ir_target": None})
                continue
            pcs = e.get("positive_classes", src.get("positive_classes", "all"))
            if pcs == "all":
                pcs = list(classes)
            elif isinstance(pcs, str) and pcs.startswith("largest:"):   # k most frequent classes
                k = int(pcs.split(":")[1])
                pcs = [c for c in classes[np.argsort(-counts, kind="stable")][:k]]
            elif isinstance(pcs, dict):                       # {"max_classes": k}
                k = min(int(pcs["max_classes"]), len(classes))
                from sepcv.config import stable_seed
                rng = np.random.default_rng(stable_seed("classes", name))
                pcs = sorted(rng.choice(classes, k, replace=False).tolist())
            for pc in pcs:
                n_c = int(counts[classes == pc][0]) if pc in classes else 0
                n_rest = int(len(y) - n_c)
                for ir in irs:
                    n_pos = min(n_c, int(max_pool // (1 + ir)), int(n_rest // ir))
                    tid = f"image|{name}|{bb}|{pc}|{ir}"
                    if n_pos < min_rare:
                        skipped.append(dict(task_id=tid, n_pos_possible=n_pos, min_rare=min_rare))
                        continue
                    specs.append({**base, "task_id": tid, "rarity": "one_vs_rest",
                                  "positive_class": pc.item() if hasattr(pc, "item") else pc,
                                  "ir_target": ir, "n_pos": n_pos})
    if skipped:
        LOG.info("%d image tasks skipped (fewer than min_rare positives)", len(skipped))
    return specs, skipped


def discover_tasks(cfg) -> tuple[list[dict], list[dict]]:
    specs, skipped = [], []
    cache_dir = Path(cfg["out_dir"]) / "_datacache"
    for src in cfg["sources"]:
        kind = src["kind"]
        if kind == "tabular":
            specs += _tabular_specs(src, cache_dir)
        elif kind == "eswa_cache":
            specs += _eswa_specs(src)
        elif kind == "embeddings":
            s, k = _embedding_specs(src)
            specs += s
            skipped += k
        elif kind == "synthetic":
            from sepcv.synthetic import synthetic_specs
            specs += synthetic_specs(src)
        else:
            raise ValueError(f"unknown source kind '{kind}'")
    ids = [s["task_id"] for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids across sources")
    return specs, skipped


# --------------------------------------------------------------------------- #
# Materialization: spec + seed -> train/test arrays
# --------------------------------------------------------------------------- #
def stratified_subsample(X, y, n_max, min_rare, rng):
    n = len(y)
    if n <= n_max:
        return X, y
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = max(int(round(len(pos) * n_max / n)), min(min_rare, len(pos)))
    n_neg = min(len(neg), int(round(n_pos * len(neg) / len(pos))))
    keep = np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    return X[keep], y[keep]


def one_vs_rest(X, y, positive_class, ir, max_pool, rng, n_pos=None):
    pos = np.flatnonzero(y == positive_class)
    rest = np.flatnonzero(y != positive_class)
    if n_pos is None:
        n_pos = min(len(pos), int(max_pool // (1 + ir)), int(len(rest) // ir))
    n_neg = int(round(n_pos * ir))
    keep_p = rng.choice(pos, n_pos, replace=False)
    keep_n = rng.choice(rest, n_neg, replace=False)
    idx = np.r_[keep_p, keep_n]
    return X[idx], np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]


def materialize(spec, cfg, seed):
    """Return Xtr, Xte, ytr, yte, info for one (task, seed)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    tcfg = cfg["task"]
    rng = np.random.default_rng(seed)
    is_test = None
    if spec["kind"] == "synthetic":
        from sepcv.synthetic import generate
        X, y = generate(spec, seed)
    else:
        X, y, extra = load_npz_xy(spec["path"])
        if spec["kind"] == "image":
            if spec.get("rarity") == "natural":
                y = (y == 1).astype(int)
                X, y = stratified_subsample(X, y, spec["max_pool"], spec["min_rare"], rng)
            else:
                X, y = one_vs_rest(X, y, spec["positive_class"], spec["ir_target"],
                                   spec["max_pool"], rng, spec.get("n_pos"))
        else:
            y = (np.asarray(y) == 1).astype(int)
            is_test = extra.get("is_test")
            if spec.get("subsample", True) and is_test is None:
                X, y = stratified_subsample(X, y, tcfg["n_max"], tcfg["min_rare"], rng)
    X = np.asarray(X, float)
    n_missing = count_missing(X)
    if is_test is not None:
        is_test = np.asarray(is_test).astype(bool)
        Xtr, Xte, ytr, yte = X[~is_test], X[is_test], y[~is_test], y[is_test]
        split = "given"
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=tcfg["test_fraction"],
                                              stratify=y, random_state=seed % (2**31 - 1))
        split = "stratified"
    Xtr, Xte = _impute(Xtr, Xte)          # medians from the training part only (no test leakage)
    if tcfg.get("standardize", True):
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    n_rare = int(y.sum())
    info = dict(n_used=int(len(y)), n_rare=n_rare, ir_eval=(len(y) - n_rare) / max(n_rare, 1),
                d=int(X.shape[1]), n_train=int(len(ytr)), n_rare_train=int(ytr.sum()),
                n_rare_test=int(yte.sum()), split=split, n_missing=n_missing,
                impute="train_median")
    return Xtr, Xte, ytr, yte, info
