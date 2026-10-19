"""Build tiny offline fixtures that mimic the real data layouts (for tests only).

  <out>/embeddings/<dataset>/<backbone>.npz   X float32, y multiclass   (image embedding cache)
  <out>/cache/<id>.npz + manifest.json        X, y (+ is_test)          (prepared domain cache)
  <out>/tabular/<name>.xlsx                   feature_* + target_binary (Rare27 export)

These are synthetic blobs, not real data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, make_classification


def main(out):
    out = Path(out)
    rng = np.random.default_rng(0)
    # embeddings: digits (10 classes) and a 20-class blob set, two "backbones"
    dig = load_digits()
    Xb, yb = make_classification(n_samples=8000, n_features=48, n_informative=20, n_classes=20,
                                 n_clusters_per_class=1, class_sep=1.2, random_state=0)
    for name, (X, y) in {"digits": (dig.data, dig.target), "blobs20": (Xb, yb)}.items():
        for bb, proj in (("bbA", None), ("bbB", rng.normal(size=(X.shape[1], 32)))):
            Z = X if proj is None else np.tanh(X @ proj / np.sqrt(X.shape[1]))
            d = out / "embeddings" / name
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{bb}.npz", X=Z.astype(np.float32), y=y)
    # prepared cache: two domain sets, one with a temporal split
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i, (ir, sep) in enumerate(((30, 1.5), (80, 0.6))):
        X, y = make_classification(n_samples=4000, n_features=12, weights=[ir / (ir + 1)],
                                   class_sep=sep, random_state=i)
        extra = {"is_test": np.arange(len(y)) >= int(0.7 * len(y))} if i == 1 else {}
        np.savez_compressed(cache / f"dom{i}.npz", X=X, y=y, **extra)
        manifest[f"dom{i}"] = {"domain": "finance", "origin": "src_a" if i == 0 else "src_b",
                               "ir_eval": float((y == 0).sum() / (y == 1).sum()), "split": "x"}
    json.dump(manifest, open(cache / "manifest.json", "w"))
    # Rare27-like xlsx
    tab = out / "tabular"
    tab.mkdir(parents=True, exist_ok=True)
    for i, (ir, sep) in enumerate(((10, 2.0), (25, 1.0), (60, 0.4), (15, 0.2), (40, 1.5), (8, 0.8))):
        X, y = make_classification(n_samples=1500, n_features=8, weights=[ir / (ir + 1)],
                                   class_sep=sep, random_state=10 + i)
        df = pd.DataFrame(X, columns=[f"feature_{j:03d}" for j in range(X.shape[1])])
        df["target_original"] = np.where(y == 1, 1, -1)
        df["target_binary"] = y
        df.to_excel(tab / f"tab{i}.xlsx", index=False)
    print(f"fixtures written to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/fixtures")
