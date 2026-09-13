"""
Stage 3 — ridge mapping: how much can we recover from each model?

Each model is allowed one piece of help: a single fitted mapping from its
representation into the human dimensions. A mapping of this kind can rotate the
space, stretch some directions relative to others, and combine existing
dimensions — but it cannot add information that was not there to begin with.
That is exactly why it is the right tool: if the mapping recovers a lot of human
structure, the structure was in the representation all along, just oriented in a
way stage 2 could not read.

Ridge regression keeps the fit from latching onto noise when there are far more
input dimensions than objects, and the regularisation strength is chosen
automatically over config.ALPHA_GRID rather than by hand.

The critical safeguard is how it is tested. Objects are split into config.N_FOLDS
groups and each group is predicted by a mapping fitted only on the other four, so
**no object's predicted human description is ever informed by its own true one**.
Without this, a mapping with thousands of free parameters could score well by
memorising and the numbers would mean nothing. The same folds are used for every
model, so the comparison across models is exact.

Fills the supplement's recovery table (mean Pearson r per model) and produces
the mapped vectors stage 4 scores.
"""

from __future__ import annotations

import numpy as np


def fold_indices(n_objects: int):
    """Yield the same (train, test) object folds for every model.

    KFold with shuffle and config.SEED — fixed, shared, and never re-drawn per
    model, otherwise the across-model comparison stops being like-for-like.
    """
    raise NotImplementedError


def fit_fold(X_train, Y_train, X_test):
    """Fit one fold and predict the held-out objects.

    Standardise the features with statistics from the *training* fold only, then
    RidgeCV over the alpha grid. Return the predictions and the chosen alpha
    (worth logging: an alpha pinned at an end of the grid means the grid is too
    narrow).
    """
    raise NotImplementedError


def out_of_fold_predictions(X, Y):
    """Assemble the full out-of-fold prediction matrix, same shape as Y.

    Every row here was predicted by a mapping that never saw that row's target.
    """
    raise NotImplementedError


def per_dimension_recovery(Y_true, Y_pred):
    """Pearson r and R^2 for each human dimension, one row per dimension.

    The headline number in the supplement is the mean Pearson r across
    dimensions; keep the per-dimension table too, since which dimensions a model
    fails to recover is more informative than the average.
    """
    raise NotImplementedError


def run(models: list[str] | None = None) -> None:
    """Fit, score and save the out-of-fold mapped vectors for each model.

    Save the mapped vectors at full [N_OBJECTS, k] size with zero rows for
    invalid objects, so stage 4 can index them with raw triplet indices.
    """
    raise NotImplementedError
