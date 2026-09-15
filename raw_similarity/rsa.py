"""
§ Raw similarity — does a model already resemble the human geometry?

Ported from things-sim `notebooks/hspose_alignment/rsa/01-04_rsa_raw_<model>.ipynb`
and `05_rsa_before_after_all_models.ipynb`.

Does a model's representation already resemble the human one before we do
anything to it? The two spaces cannot be compared directly — different numbers of
dimensions, and dimensions that mean different things — so we compare *patterns
of similarity*: how similar the model considers every possible pair of objects,
against the same thing for hSPOSE. Does this model also treat dolphin-shark as a
closer pair than dolphin-helicopter?

The correlation is rank-based, because only the ordering of pairs is comparable
between two spaces on unrelated scales. Nothing is trained or fitted here.

A low score is not by itself evidence that a model lacks human-relevant
information: comparing similarity in a model's own coordinates treats all of its
dimensions as equally important, so structure that is present but spread across
unhelpful directions will not show up. § Ridge mapping is what tells those cases apart,
which is why this step also reports the mapped and human-trained versions beside
the raw one.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from common import shared


def rsa(A: np.ndarray, B: np.ndarray) -> float:
    """Spearman correlation between two spaces' pairwise cosine geometries.

    Both arrays must already be restricted to the same objects, in the same order,
    so every model's r is computed over an identical set of object pairs.
    """
    pairs_a = shared.upper_triangle(shared.cosine_similarity_matrix(A))  # each unordered object pair once
    pairs_b = shared.upper_triangle(shared.cosine_similarity_matrix(B))
    return float(spearmanr(pairs_a, pairs_b)[0])                  # rank-based: only the ordering of pairs is comparable
