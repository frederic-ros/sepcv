"""Assemble the Zenodo deposit of the paper from a completed reproduction.

    python tools/make_zenodo.py --out zenodo_upload

Run it from the repository root after ``python reproduce.py analyse budget halving figures
verify`` has succeeded. It reads data/, results/, reports/ and paper_figures/ (override with
--data, --results, --reports, --figures) and writes one folder ready for upload:

    zenodo_upload/
      README.md                  description of the deposit (from tools/zenodo_README.md)
      LICENSES_DATA.md           sources and licences of the datasets (to complete by hand)
      MANIFEST.sha256            checksum of every file in the deposit
      zenodo.json                metadata for the Zenodo upload form / API
      sepcv-<version>.zip        software snapshot (this repository, without data or results)
      data_rare27.zip            27 imbalanced-learn datasets (.xlsx)
      data_domain_cache.zip      prepared domain datasets (.npz + manifest.json), without
                                 UNSW-NB15 (not redistributable) and BAF (own licence)
      data_domain_baf.zip        Bank Account Fraud, prepared (CC BY-NC-SA 4.0)
      data_embeddings.zip        cached image embeddings (<dataset>/<backbone>.npz)
      results_controlled.zip     trajectories, one JSON per (task, seed, start)
      results_rare27.zip           "      + units_extra/ (post-hoc probes)
      results_domains_v2.zip       "
      results_images_eswa_tasks.zip
      reports.zip                all analysis reports (CSV + Markdown + figures)
      paper_figures.zip          the figures of the paper

Large folders are zipped one archive per experiment so that each download stays manageable.
Logs, temporary files and caches are excluded. The script refuses to run if verify_paper.py
does not pass on the reports to be archived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sepcv import __version__  # noqa: E402

EXPERIMENTS = ["controlled", "rare27", "domains_v2", "images_eswa_tasks"]
DATA = {"rare27": "data_rare27.zip", "cache": "data_domain_cache.zip",
        "embeddings": "data_embeddings.zip"}
# Licence handling (see tools/LICENSES_DATA.md): files of data/<folder> whose name starts with
# one of these prefixes are not deposited ...
DATA_EXCLUDE = {"cache": ("unsw_nb15_", "imblearn_", "standard_")}
# ... or are deposited in a separate archive under their own licence.
DATA_SEPARATE = {"cache": {"baf_": "data_domain_baf.zip"}}
SKIP_SUFFIX = {".log", ".tmp", ".pyc"}
SKIP_NAMES = {"Thumbs.db", "desktop.ini"}
SKIP_PARTS = {"__pycache__", ".git", ".pytest_cache", ".venv", "data", "results", "reports",
              "paper_figures", "zenodo_upload", ".ipynb_checkpoints", "build", "dist",
              "_datacache"}


def zip_dir(src: Path, dst: Path, arc_root: str, skip_parts=frozenset({"_datacache"}), exclude=None,
            keep=None):
    n = 0
    files = sorted(Path(d) / f for d, _, ns in os.walk(src, followlinks=True) for f in ns)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            rel = f.relative_to(src)
            if f.suffix in SKIP_SUFFIX or f.name in SKIP_NAMES or set(rel.parts[:-1]) & skip_parts:
                continue
            if any(p.endswith(".egg-info") for p in rel.parts[:-1]):
                continue
            if exclude is not None and exclude in f.resolve().parents:
                continue
            if keep is not None and not keep(f.name):
                continue
            if f.name.endswith(".tmp") or (f.name.startswith(".") and f.name != ".gitignore"):
                continue
            z.write(f, f"{arc_root}/{rel.as_posix()}")
            n += 1
    print(f"  {dst.name}: {n} files, {dst.stat().st_size / 1e6:.1f} MB")
    return n


def sha256(f: Path):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "zenodo_upload"))
    p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--reports", default=str(ROOT / "reports"))
    p.add_argument("--figures", default=str(ROOT / "paper_figures"))
    p.add_argument("--skip-verify", action="store_true")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data, res, rep, figs = Path(a.data), Path(a.results), Path(a.reports), Path(a.figures)

    for d in [data / k for k in DATA] + [res / e for e in EXPERIMENTS] + [rep, figs]:
        if not d.exists():
            sys.exit(f"missing folder: {d}")
    if not a.skip_verify:
        print("checking the reports against the paper ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_paper.py"), str(rep)], check=True)

    print("software")
    snap = out / f"sepcv-{__version__}.zip"
    tag = f"v{__version__}"
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                         capture_output=True)
    if git.returncode == 0:            # exactly the tagged GitHub release
        subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip",
                        f"--prefix=sepcv-{__version__}/", "-o", str(snap.resolve()), tag], check=True)
        print(f"  {snap.name}: git archive of tag {tag}")
    else:
        print(f"  (no git tag {tag}: zipping the working tree)")
        zip_dir(ROOT, snap, f"sepcv-{__version__}", skip_parts=SKIP_PARTS, exclude=out.resolve())
    print("data")
    for k, name in DATA.items():
        drop = DATA_EXCLUDE.get(k, ())
        sep = DATA_SEPARATE.get(k, {})
        own = tuple(drop) + tuple(sep)
        zip_dir(data / k, out / name, f"data/{k}", keep=lambda n, own=own: not n.startswith(own))
        for prefix, sname in sep.items():
            zip_dir(data / k, out / sname, f"data/{k}", keep=lambda n, pr=prefix: n.startswith(pr))
        if drop:
            print(f"  not deposited (licence or unused): data/{k}/{'*, '.join(drop)}*")
    print("results")
    for e in EXPERIMENTS:
        zip_dir(res / e, out / f"results_{e}.zip", f"results/{e}")
    print("reports and figures")
    zip_dir(rep, out / "reports.zip", "reports")
    zip_dir(figs, out / "paper_figures.zip", "paper_figures")

    for name in ("zenodo_README.md", "LICENSES_DATA.md"):
        src = ROOT / "tools" / name
        txt = src.read_text(encoding="utf-8").replace("{VERSION}", __version__)
        (out / ("README.md" if name == "zenodo_README.md" else name)).write_text(txt, encoding="utf-8")
    meta = json.loads((ROOT / "tools" / "zenodo.json").read_text(encoding="utf-8"))
    meta["version"] = __version__
    (out / "zenodo.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"{sha256(f)}  {f.name}" for f in sorted(out.iterdir())
             if f.is_file() and f.name not in ("MANIFEST.sha256", "zenodo.json")]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(f"\ndeposit ready in {out} ({len(lines)} files). Fill in LICENSES_DATA.md before upload.")


if __name__ == "__main__":
    main()
