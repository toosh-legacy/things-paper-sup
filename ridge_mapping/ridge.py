"""
§ Ridge mapping — how much can we recover from each model?

Ported from things-sim `notebooks/hspose_alignment/ridge_mapping/0{1,2,3,4}_ridge_<model>.ipynb`
(one notebook per model, identical except for the model name).

Each model is allowed one piece of help: a single fitted mapping from its
representation into the human dimensions. A mapping of this kind can rotate the
space, stretch some directions relative to others and combine existing
dimensions — but it cannot add information that was not there. That is why it is
the right tool: if the mapping recovers a lot of human structure, the structure
was in the representation all along, just oriented in a way § Raw similarity
could not read.

Ridge keeps the fit from latching onto noise when there are far more input
dimensions than objects, and alpha is chosen automatically over config.ALPHAS.

The safeguard is the split: objects are divided into config.N_FOLDS groups and
each group is predicted by a mapping fitted only on the other four, so no
object's predicted human description is ever informed by its own true one. The
folds depend on the seed alone, so every model — and behavioural training's branches B and C —
sees exactly the same split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import config


def folds(n_samples: int) -> list:
    """The one object split used by the ridge mapping, its nonlinear counterpart and extra/."""
    splitter = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)  # seed only — never the model
    return list(splitter.split(np.arange(n_samples)))             # materialised so every caller gets identical folds


def ridge_out_of_fold(X: np.ndarray, Y: np.ndarray):
    """Out-of-fold RidgeCV predictions of the human dimensions, plus the alphas.

    Returns (Y_oof [n, k], one chosen alpha per fold). Every row of Y_oof was
    predicted by a mapping that never saw that row's true human description.
    """
    Y_oof = np.zeros_like(Y)                                      # filled fold by fold; every row written exactly once
    alphas = []                                                   # an alpha pinned at an end of the grid = too narrow a grid
    for fold, (train, test) in enumerate(folds(len(X))):
        scaler = StandardScaler().fit(X[train])                   # scaling statistics from the training fold only,
        model = RidgeCV(alphas=config.ALPHAS).fit(scaler.transform(X[train]), Y[train])  # so held-out objects leak nothing
        Y_oof[test] = model.predict(scaler.transform(X[test]))    # predict the objects this mapping never saw
        alphas.append(float(model.alpha_))
        print(f"fold {fold}: alpha = {model.alpha_:.3g}")
    return Y_oof, alphas


def per_dimension_recovery(Y: np.ndarray, Y_oof: np.ndarray) -> pd.DataFrame:
    """Pearson r, p and R^2 for each human dimension, one row per dimension.

    The headline number in the supplement is the mean Pearson r, but which
    dimensions a model fails to recover is more informative than the average, so
    the per-dimension table is kept. Shared with the nonlinear map so both are
    scored identically.
    """
    rows = []
    for dimension in range(Y.shape[1]):
        r, p = pearsonr(Y[:, dimension], Y_oof[:, dimension])      # correlation between true and out-of-fold predicted
        residual = np.sum((Y[:, dimension] - Y_oof[:, dimension]) ** 2)  # unexplained variance...
        total = np.sum((Y[:, dimension] - Y[:, dimension].mean()) ** 2)  # ...against the variance there was to explain
        rows.append({"dim": dimension, "pearson_r": r, "p_value": p, "r2": 1 - residual / total})
    return pd.DataFrame(rows)
