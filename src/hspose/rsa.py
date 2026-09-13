"""
Stage 2 — raw similarity.

Does a model's representation already resemble the human one before we do
anything to it? The two spaces cannot be compared directly — different numbers
of dimensions, and dimensions that mean different things — so we compare
*patterns of similarity* instead: how similar the model considers every possible
pair of objects, against the same thing for hSPOSE. Does this model also treat
dolphin-shark as a closer pair than dolphin-helicopter?

The correlation is rank-based, because only the ordering of pairs is comparable
between two spaces on unrelated scales. Nothing is trained or fitted here.

A low score is not by itself evidence that a model lacks human-relevant
information: comparing similarity in a model's own coordinates treats all of its
dimensions as equally important, so structure that is present but spread across
unhelpful directions will not show up. Stage 3 is what tells those two cases
apart.

Fills the supplement's "Raw similarity" table (one Spearman r per model).
"""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """L2-normalise the rows, then return the [n, n] cosine similarity matrix."""
    raise NotImplementedError


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the unique object pairs of a similarity matrix, k=1 (no diagonal)."""
    raise NotImplementedError


def rsa(sim_a: np.ndarray, sim_b: np.ndarray) -> tuple[float, float]:
    """Spearman-correlate two similarity matrices over their unique pairs."""
    raise NotImplementedError


def run(models: list[str] | None = None) -> None:
    """Score each model's raw geometry against hSPOSE's and save the table.

    Restrict both spaces to the same valid objects before building either
    matrix, so every model's r is computed over an identical set of pairs.
    """
    raise NotImplementedError
