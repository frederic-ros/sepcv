"""Bayes-optimal AUPRC in the binormal model (Fig. 1 of the paper) + summary numbers.

AUPRC*(m, pi) = integral of precision d(recall), with TPR(t) = 1 - Phi(t - m),
FPR(t) = 1 - Phi(t). Usage: python scripts/fig_bayes.py [out.pdf]
"""
import sys

import numpy as np
from scipy.stats import norm


def auprc_star(m, pi, n=4000):
    t = np.linspace(-8, 8 + m, n)[::-1]              # thresholds from high to low
    tpr, fpr = norm.sf(t - m), norm.sf(t)
    prec = pi * tpr / np.maximum(pi * tpr + (1 - pi) * fpr, 1e-300)
    return float(np.sum(np.diff(np.r_[0.0, tpr]) * prec))


def main(out="fig_bayes.pdf"):
    ms = np.linspace(0, 6, 121)
    pis = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    grid = np.array([[auprc_star(m, p) for p in pis] for m in ms])
    rarity_effect = grid.max(1) - grid.min(1)                 # at fixed m, across pi
    sep_effect = grid.max(0) - grid.min(0)                     # at fixed pi, across m
    print(f"max effect of rarity at fixed m (pi in [{pis[0]}, {pis[-1]}]): "
          f"{rarity_effect.max():.3f} at m={ms[rarity_effect.argmax()]:.2f}")
    print(f"effect of separability at fixed pi (m in [0, 6]): {sep_effect.min():.3f}-{sep_effect.max():.3f}")
    pis_b = [0.0077, 0.02, 0.05, 0.1]                          # benchmark range (IR 9-130)
    g2 = np.array([[auprc_star(m, p) for p in pis_b] for m in ms])
    print(f"max effect of rarity, pi in [{pis_b[0]}, {pis_b[-1]}]: {(g2.max(1) - g2.min(1)).max():.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        for j, p in enumerate(pis):
            ax.plot(ms, grid[:, j], label=f"$\\pi$={p}")
        ax.set_xlabel("separability margin $m$")
        ax.set_ylabel("Bayes-optimal AUPRC")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(out)
        print("written", out)
    except Exception as e:                                     # noqa: BLE001
        print("no figure:", e)


if __name__ == "__main__":
    main(*sys.argv[1:])
