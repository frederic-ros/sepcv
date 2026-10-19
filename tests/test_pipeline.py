"""Offline end-to-end tests on synthetic fixtures (run: pytest -q)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sepcv.config import DEFAULTS, _merge          # noqa: E402
from sepcv.data import discover_tasks, materialize, one_vs_rest, stratified_subsample  # noqa: E402
from sepcv.engine import run_unit                    # noqa: E402
from sepcv.estimators import sepcv                   # noqa: E402

TINY = {"protocol": {"seeds": 2, "starts": ["warm", "cold"], "rounds": 5, "probe_rounds": [1, 2, 3],
                     "learners": ["logreg", "knn"], "strategies": ["margin", "random"]},
        "estimators": {"cv_repeats": {"logreg": 1, "knn": 1, "rf": 1}, "rf_trees": 10,
                       "bases": ["logreg", "knn"]},
        "references": {"enabled": True, "bases": ["logreg"]}}


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory):
    out = tmp_path_factory.mktemp("fx")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fixtures.py"), str(out)], check=True)
    return out


def test_subsample_preserves_ir():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200, int), np.zeros(30000, int)]
    X = np.zeros((len(y), 2))
    _, ys = stratified_subsample(X, y, 10_000, 100, rng)
    assert abs((len(ys) - ys.sum()) / ys.sum() - 150) < 2


def test_one_vs_rest_ir():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 1200)
    X = np.zeros((len(y), 3))
    _, yt = one_vs_rest(X, y, 3, 49, 10_000, rng)
    assert yt.sum() == 200 and (len(yt) - yt.sum()) == 200 * 49


def test_sepcv_undefined_below_three_positives():
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.zeros(40, int); y[:2] = 1
    ec = _merge(DEFAULTS, TINY)["estimators"]
    assert np.isnan(sepcv(X, y, "logreg", ec, 0)["sepcv_logreg_auroc"])
    y[:3] = 1
    assert not np.isnan(sepcv(X, y, "logreg", ec, 0)["sepcv_logreg_auroc"])


def test_discovery_and_units(fixtures, tmp_path):
    cfg = _merge(DEFAULTS, {**TINY, "out_dir": str(tmp_path / "r"), "sources": [
        {"kind": "tabular", "path": str(fixtures / "tabular")},
        {"kind": "eswa_cache", "path": str(fixtures / "cache"), "family": "domain"},
        {"kind": "embeddings", "path": str(fixtures / "embeddings"), "backbones": ["bbA", "bbB"],
         "target_irs": [20], "min_rare": 20, "max_pool": 3000,
         "positive_classes": {"max_classes": 3}}]})
    specs, skipped = discover_tasks(cfg)
    fams = {s["family"] for s in specs}
    assert fams == {"tabular", "domain", "image"}
    img = [s for s in specs if s["family"] == "image"]
    assert len({s["group"] for s in img}) == 2           # classes and backbones share a group
    dom1 = [s for s in specs if s["dataset"] == "dom1"][0]
    *_, info = materialize(dom1, cfg, 1)
    assert info["split"] == "given"
    for sp in (specs[0], img[0]):
        res = run_unit(sp, 0, "warm", cfg, with_references=True)
        assert res["meta"]["n_rare_init"] >= 1
        assert {t["strategy"] for t in res["trajectories"]} == {"margin", "random"}
        acq = res["trajectories"][0]["acq"]
        assert len(acq) == 1 + 5 and len(acq[0]) == 30


def test_cli_run_and_analyse(fixtures, tmp_path):
    cfg = {**TINY, "name": "t", "out_dir": str(tmp_path / "res"), "n_jobs": 1,
           "sources": [{"kind": "embeddings", "path": str(fixtures / "embeddings"),
                        "backbones": ["bbA"], "target_irs": [20], "min_rare": 20,
                        "max_pool": 3000, "positive_classes": {"max_classes": 4}},
                       {"kind": "tabular", "path": str(fixtures / "tabular")}]}
    cpath = tmp_path / "c.json"
    json.dump(cfg, open(cpath, "w"))
    env = {"PYTHONPATH": str(ROOT / "src")}
    import os
    env = {**os.environ, **env}
    subprocess.run([sys.executable, "-m", "sepcv", "run", str(cpath)], check=True, env=env)
    subprocess.run([sys.executable, "-m", "sepcv", "analyse", str(tmp_path / "res"), "--out",
                    str(tmp_path / "rep"), "--boot", "50"], check=True, env=env)
    rep = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
    assert "Association" in rep and "leave-one-group-out" in rep


def test_controlled_factorial(tmp_path):
    from sepcv.analysis.report import analyse
    from sepcv.runner import run
    cfg = _merge(DEFAULTS, {**TINY, "out_dir": str(tmp_path / "c"), "sources": [
        {"kind": "synthetic", "geometries": ["gauss", "multimodal"], "irs": [5, 20],
         "seps": [1, 4], "dims": [6], "n": 1200}]})
    cfg["protocol"]["seeds"] = 1
    run(cfg)
    out = analyse([tmp_path / "c"], tmp_path / "rep", boot=20)
    txt = (out / "report.md").read_text(encoding="utf-8")
    assert "Controlled factorial check" in txt
    assert (out / "controlled_variance.csv").exists()


def test_reprobe_and_budget(tmp_path):
    from sepcv.analysis.budget import run_budget
    from sepcv.reprobe import reprobe
    from sepcv.runner import run
    cfg = _merge(DEFAULTS, {**TINY, "out_dir": str(tmp_path / "b"), "sources": [
        {"kind": "synthetic", "geometries": ["gauss"], "irs": [5, 20, 40],
         "seps": [0.5, 1.5, 3, 4.5], "dims": [6], "n": 1200}]})
    cfg["protocol"]["seeds"] = 1
    cfg["protocol"]["rounds"] = 6
    run(cfg)
    assert reprobe(cfg, rounds=(5,), strategies=("margin",)) > 0
    out = run_budget([tmp_path / "b"], tmp_path / "rep", draws=20)
    rel = (out / "budget_reliability.csv").read_text(encoding="utf-8")
    assert ",5," in rel
    from sepcv.analysis.halving import run_halving
    out = run_halving([tmp_path / "b"], tmp_path / "sh", draws=20)
    import pandas as pd
    h = pd.read_csv(out / "halving.csv")
    assert {"SH (oracle reward), eta=2", "SepCV+ -> SH (oracle reward), eta=2"} <= set(h.policy)
    assert (h.budget_used <= 1.0 + 1e-9).all()
