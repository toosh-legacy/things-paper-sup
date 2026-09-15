"""
§ Odd-one-out: raw representations vs. the mapped ones.

Ported from things-sim `notebooks/hspose_alignment/ridge_mapping/05-08_oddoneout_<model>.ipynb`.

§ Raw similarity and § Ridge mapping are geometry metrics. This one runs each
representation through the actual behavioural task: for every held-out triplet
the model is given the same three objects a participant saw, the two it considers
most similar are worked out by the cosine rule in common/shared.py, and the
leftover object is scored against what the participant actually chose.

Done twice per model — the untouched representation and the ridge-mapped one —
on exactly the same triplets, because a gain column comparing two different test
sets would mean nothing. Neither column can be inflated by having seen the
answers: the scoring triplets were held out of the hSPOSE fit, and the mapped
vectors come from mappings that never saw those objects' human descriptions.

Read every percentage against chance below and the human-human ceiling above.
"""

from __future__ import annotations

import numpy as np


def scorable_triplets(triplets: np.ndarray, *vector_sets: np.ndarray) -> np.ndarray:
    """Keep the triplets whose three objects have a real vector in *every* set.

    A row is unusable when it is NaN (no image) or all-zero (an object outside the
    out-of-fold mapped set), and both columns must be scored on the same triplets,
    so the two exclusions are unioned before filtering.
    """
    unusable = np.zeros(len(vector_sets[0]), dtype=bool)          # one flag per object, across all representations
    for vectors in vector_sets:
        unusable |= np.isnan(vectors).any(axis=1)                 # no usable image for this object...
        unusable |= np.abs(vectors).sum(axis=1) == 0              # ...or an all-zero row, which carries no direction
    return triplets[~unusable[triplets].any(axis=1)]              # drop a triplet if any of its three objects is unusable
