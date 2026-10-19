"""Downstream learners (settings shared with the companion study) plus k-NN.

All learners are class-weighted (k-NN uses distance weighting, which it cannot
class-weight). Hyperparameters are fixed a priori.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def make_learner(name: str, seed: int = 0, rf_trees: int = 200):
    seed = int(seed) % (2**31 - 1)
    if name == "logreg":
        return LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear",
                                  random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=rf_trees, class_weight="balanced_subsample",
                                      min_samples_leaf=2, random_state=seed, n_jobs=1)
    if name == "gbdt":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15,
                                              min_samples_leaf=2, l2_regularization=1.0,
                                              class_weight="balanced", early_stopping=False,
                                              random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    raise ValueError(f"unknown learner '{name}'")


class PriorModel:
    """Stand-in while the labeled set holds a single class (cold start)."""
    classes_ = np.array([0, 1])

    def __init__(self, prior=0.5):
        self.prior = prior

    def predict_proba(self, X):
        return np.tile([1 - self.prior, self.prior], (len(X), 1))


def fit(name, seed, X, y, rf_trees=200):
    if len(np.unique(y)) < 2:
        return PriorModel(), True
    m = make_learner(name, seed, rf_trees)
    if name == "knn" and len(y) < 5:
        m.set_params(n_neighbors=max(1, len(y)))
    return m.fit(X, y), False


def pos_proba(model, X):
    p = model.predict_proba(X)
    cls = list(model.classes_)
    return p[:, cls.index(1)] if 1 in cls else np.zeros(len(X))
