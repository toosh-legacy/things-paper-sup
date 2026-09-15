"""
§ Reference points — the scale every percentage is read against.

Percentages on this task cannot be read against 100 %, because the task has no
correct answer. Three anchors set the scale: chance, the free human embedding
(what hSPOSE itself scores), and the human-human ceiling.

The ceiling is the one that matters. Shown *dolphin, shark, helicopter* almost
everyone picks the helicopter; shown *knife, fork, scissors* people disagree.
Scoring a model against "the human's answer" scores it against one participant's
judgement on a question other participants would have answered differently, so
the gap between the ceiling and 100 % is disagreement between people, not model
error — the meaningful range runs from chance to the ceiling.

things-sim took the ceiling from Hebart et al. (0.6722) rather than measuring it;
the estimator below re-measures it from repeated triplets when the split has any,
and the run script reports both.
"""

from __future__ import annotations

import numpy as np

from common import config


def measured_ceiling(triplets: np.ndarray) -> float | None:
    """How often two participants shown the same three objects agree, or None.

    Estimator: over every *pair* of responses to the same unordered object triple,
    the proportion that named the same odd one out. Using all pairs rather than
    first-vs-second uses every repeat and weights a triple by how often it was
    answered, which is also how often it reaches the scored set. Returns None when
    the split contains no repeated triple, which is when the paper's figure is the
    only number available.
    """
    triples = np.sort(triplets, axis=1)                            # unordered: same three objects, however they were shown
    keys = (triples[:, 0].astype(np.int64) * config.N_OBJECTS ** 2  # pack the sorted indices into one integer id,
            + triples[:, 1] * config.N_OBJECTS + triples[:, 2])     # so "same three objects" is a plain equality test
    answers = triplets[:, 2]                                       # the object this participant called the odd one out

    order = np.argsort(keys, kind="stable")                        # group identical triples by sorting once
    keys, answers = keys[order], answers[order]
    blocks = np.split(answers, np.flatnonzero(np.diff(keys)) + 1)  # one block = every response to one object triple

    agreeing, total = 0, 0                                         # numerator and denominator of the ceiling
    for block in blocks:
        if len(block) < 2:                                         # a triple answered once says nothing about agreement
            continue
        counts = np.bincount(block)                                # how many participants chose each object
        agreeing += int((counts * (counts - 1)).sum())             # ordered response pairs that agree
        total += len(block) * (len(block) - 1)                     # all ordered response pairs for this triple
    return agreeing / total if total else None                     # None -> no repeats in this split
