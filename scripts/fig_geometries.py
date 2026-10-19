"""Illustration of the three synthetic geometries (paper figure).

Two informative coordinates, no rotation, IR 20, separations s = 2 and s = 4.
Usage: python scripts/fig_geometries.py [out.pdf]
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from sepcv.synthetic import generate  # noqa: E402


def main(out="fig_geometries.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sepcv.synthetic as S
    S._rotation = lambda d, rng: np.eye(d)          # show the informative plane
    geos = [("gauss", "one shifted mode"), ("multimodal", "three pockets"), ("moon", "half circle")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True)
    for i, s in enumerate((2, 4)):
        for j, (g, lab) in enumerate(geos):
            X, y = generate(dict(n=4000, ir=20, sep=s, d=2, geometry=g), 3)
            ax = axes[i, j]
            ax.scatter(X[y == 0, 0], X[y == 0, 1], s=2, c="0.75", lw=0, rasterized=True)
            ax.scatter(X[y == 1, 0], X[y == 1, 1], s=5, c="C3", lw=0, rasterized=True)
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect("equal")
            if i == 0:
                ax.set_title(f"{g} ({lab})", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"$s={s}$")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("written", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
