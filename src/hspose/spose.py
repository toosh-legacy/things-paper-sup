"""
Stage 1 — hSPOSE, the human yardstick.

SPoSE (Hebart et al. 2020) turns millions of odd-one-out responses into a short
list of numbers per concept, where every number is non-negative and most are
zero. Non-negativity is what makes the recovered dimensions readable as ordinary
attributes (*is food*, *is metallic*, *is animate*): a dimension can say an
object has a property strongly, weakly or not at all, never negatively. The L1
sparsity is what leaves each object described by the few properties that
genuinely apply to it rather than by a little of everything.

The embedding is fitted to people's *choices*, not to similarity ratings: it is
adjusted until the pair a participant treated as similar is the pair it scores
as most likely.

We refit rather than download the published embedding so that the human
reference comes from the same objects and the same data split as everything it
is compared against. The recipe is followed exactly, sparsity setting included.

Output: results/spose/hspose.npy [N_OBJECTS, k] plus the three reproduction
checks that fill the supplement's first table.
"""

from __future__ import annotations

import numpy as np


def triplet_nll_and_accuracy(embedding, triplets):
    """The SPoSE choice model on one batch: (mean NLL, accuracy).

    For a row [i, j, k], (i, j) is the pair the participant kept together, so it
    is the correct answer. Take the non-negative part of each object's weights,
    score the three pairs by dot product, softmax over those three, and treat the
    (i, j) column as the target class. Accuracy is how often that column wins.
    """
    raise NotImplementedError


def sparsity_penalty(embedding):
    """L1 penalty at config.LAMBDA_L1, plus a soft push toward non-negativity.

    Scale the L1 term by the number of objects so the setting means the same
    thing regardless of how many concepts are in the fit.
    """
    raise NotImplementedError


def fit(train, test, seed: int | None = None):
    """Fit a free [N_OBJECTS, INIT_DIM] embedding on human triplets.

    Free means exactly that: the parameters are the object weights themselves,
    with no model features as input — this is a description of the human data,
    not of any network. Initialise small and positive, optimise NLL + sparsity
    with Adam, evaluate held-out NLL each epoch, early-stop on config.PATIENCE,
    and restore the best-scoring weights before returning.

    Returns the raw [N_OBJECTS, INIT_DIM] weights and the per-epoch history.
    """
    raise NotImplementedError


def prune(weights: np.ndarray) -> np.ndarray:
    """Drop dimensions no object uses, then sort by total weight.

    Keep a dimension when its largest object weight exceeds config.PRUNE_THR.
    The fit starts from a deliberately generous INIT_DIM, so the final count is
    something the procedure discovers rather than something we choose — report
    it, don't target it.
    """
    raise NotImplementedError


def similarity_matrix(embedding: np.ndarray) -> np.ndarray:
    """Dot-product similarity over the non-negative embedding, [n, n].

    This is hSPOSE's own similarity, the human side of every comparison in
    stages 2 and 3.
    """
    raise NotImplementedError


def check_against_published(hspose: np.ndarray, valid: np.ndarray) -> float:
    """Third reproduction check: correlate our similarity matrix with theirs.

    Build both similarity matrices over the same valid objects, take the upper
    triangle of each, and correlate. High agreement means the refit describes the
    same structure even where individual dimensions are ordered differently.
    """
    raise NotImplementedError


def run() -> None:
    """Fit, prune, save, and write the reproduction summary.

    The summary row carries the three numbers the supplement reports beside
    Hebart et al.'s: held-out triplet accuracy, dimensions kept after pruning,
    and the similarity-matrix correlation with their published 49-d embedding.
    """
    raise NotImplementedError
