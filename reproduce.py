"""Reproduce every experiment, report, table and figure of the paper.

    python reproduce.py all            # everything, in order (several days on 8 cores)
    python reproduce.py <stage> ...    # one or more stages

Stages, in order:
    run        run the trajectories of the four experiment configurations (resumable)
    reprobe    post-hoc probes at 78, 102 and 126 labels from the stored acquisition order
    analyse    main reports: controlled factorial, real tabular tasks, image embeddings
    budget     exploration-budget analysis (tabular and images)
    halving    successive halving vs single-stage ranking (tabular and images)
    figures    all figures of the paper, written to paper_figures/
    verify     check the regenerated reports against the numbers printed in the paper

To regenerate reports and figures from the archived results (Zenodo) without rerunning the
experiments, unpack the archive's results/ folder here and run:
    python reproduce.py analyse budget halving figures verify

Options: --n-jobs N (run, reprobe), --data DIR (default: data), --results DIR (default:
results), --reports DIR (default: reports). A non-default --data changes the configuration
hash, so archived results can then not be resumed with `run`. Do not call `run` on archived
results before archiving them: it rewrites their run_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGS = ["controlled", "rare27", "domains_v2", "images_eswa_tasks"]
REPROBED = ["rare27", "domains_v2", "images_eswa_tasks"]
STAGES = ["run", "reprobe", "analyse", "budget", "halving", "figures", "verify"]


def sh(*args):
    """Run a Python command with this repository's src/ first on the path."""
    print("+", " ".join(map(str, args)), flush=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])}
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT, env=env)


def config_path(name, a):
    """The shipped configuration, or a copy redirected to other data/result folders.

    With the default folders the shipped file is used as is, so that its hash matches the
    run_manifest.json of the archived results and runs can be resumed."""
    if a.data == "data" and a.results == "results":
        return Path("configs") / f"{name}.json"
    cfg = json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    cfg["out_dir"] = (Path(a.results) / name).as_posix()      # not hashed
    if a.data != "data":                                       # data paths are hashed
        for src in cfg.get("sources", []):
            for key in ("path", "manifest"):
                if key in src:
                    src[key] = (Path(a.data) / Path(src[key]).relative_to("data")).as_posix()
    tmp = Path(tempfile.gettempdir()) / f"sepcv_{name}.json"
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return tmp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--reports", default="reports")
    p.add_argument("--n-jobs", type=int)
    a = p.parse_args()
    stages = STAGES if "all" in a.stages else [s for s in STAGES if s in a.stages]
    res, rep = Path(a.results), Path(a.reports)
    tab = [res / "rare27", res / "domains_v2"]
    img = [res / "images_eswa_tasks"]
    nj = ["--n-jobs", a.n_jobs] if a.n_jobs else []
    for st in stages:
        print(f"\n===== {st} =====", flush=True)
        if st == "run":
            for c in CONFIGS:
                sh("-m", "sepcv", "run", config_path(c, a), *nj)
        elif st == "reprobe":
            for c in REPROBED:
                sh("-m", "sepcv", "reprobe", config_path(c, a), "--rounds", 6, 9, 12, *nj)
        elif st == "analyse":
            sh("-m", "sepcv", "analyse", res / "controlled", "--out", rep / "controlled")
            sh("-m", "sepcv", "analyse", *tab, "--out", rep / "tabular")
            sh("-m", "sepcv", "analyse", *img, "--out", rep / "images")
        elif st == "budget":
            sh("-m", "sepcv", "budget", *tab, "--out", rep / "budget_tabular")
            sh("-m", "sepcv", "budget", *img, "--out", rep / "budget_images")
        elif st == "halving":
            sh("-m", "sepcv", "halving", *tab, "--out", rep / "halving_tabular")
            sh("-m", "sepcv", "halving", *img, "--out", rep / "halving_images")
        elif st == "figures":
            out = ROOT / "paper_figures"
            sh("scripts/paper_figures.py", rep, out)
            sh("scripts/fig_variance.py", out / "fig_variance_decomposition.pdf")
            sh("scripts/fig_bayes.py", out / "fig_bayes.pdf")
            sh("scripts/fig_geometries.py", out / "fig_geometries.pdf")
            sh("scripts/graphical_abstract.py", out)
        elif st == "verify":
            sh("scripts/verify_paper.py", rep)


if __name__ == "__main__":
    main()
