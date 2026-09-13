"""
Stage 4 — odd-one-out: raw representations vs. the mapped ones.

Stages 2 and 3 are about overall structure. This one asks the stricter question:
does the model reproduce individual human decisions? For each held-out triplet
the model is given the same three objects a participant saw, the two it
considers most similar are worked out, and the remaining object is taken as its
answer and scored against what the participant actually chose.

Done twice per model — once on the model's own untouched representation, once on
the stage 3 mapped version. Neither column can be inflated by having seen the
answers: the scoring triplets were held out of the hSPOSE fit, and the mapped
predictions come from mappings that never saw those objects' human descriptions.

The gain column is the behavioural counterpart of stage 3's recovery scores.
Where a model improves on both, it is not merely producing a similarity structure
that correlates better in aggregate — it is answering more of the actual
questions the way a person answered them.

Read every percentage against chance below and the human-human ceiling above
(see anchors.py).
"""

from __future__ import annotations

import numpy as np


def predict(vectors: np.ndarray, triplets: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """The decision rule, used identically for raw and mapped vectors.

    L2-normalise the vectors once, then for each triplet score its three pairs by
    cosine similarity: the winning pair stays together and the leftover object is
    the model's odd-one-out. Chunk the triplets — there are hundreds of thousands
    of them and the pairwise scores do not need to exist all at once.

    Returning the reordered triplet (kept pair first, odd one out last) rather
    than a bare label keeps the output directly comparable with the human rows.
    """
    raise NotImplementedError


def accuracy(vectors: np.ndarray, triplets: np.ndarray) -> float:
    """Percentage of triplets where the model's odd-one-out matches the human's."""
    raise NotImplementedError


def scorable_triplets(triplets: np.ndarray, *vector_sets) -> np.ndarray:
    """Keep only triplets whose three objects have a real vector in every set.

    Both columns must be scored on exactly the same triplets, or the gain column
    compares two different test sets.
    """
    raise NotImplementedError


def run(models: list[str] | None = None) -> None:
    """Score raw vs. ridge-mapped for each model and save the table.

    Rows: raw %, ridge-mapped %, gain. The EVA02 and IMIC-B rows carry the
    central comparison in both columns — if the two differ before mapping but
    converge after it, person-identity training reorganised the model's object
    structure without destroying it; if the difference survives the mapping, that
    training genuinely cost the model information human judgements depend on.
    """
    raise NotImplementedError
