"""
Reference points — the scale the percentages are read against.

Percentages on this task cannot be read against 100 %, because the task has no
correct answer. Three anchors set the scale:

    chance               a random three-way guess
    free human embedding what hSPOSE itself scores — a purpose-built human model
    human-human ceiling  how often two people agree; nothing can beat this

The ceiling is the one that matters. Shown *dolphin, shark, helicopter* almost
everyone picks the helicopter; shown *knife, fork, scissors* people disagree.
Scoring a model against "the human's answer" scores it against one participant's
judgement on a question other participants would have answered differently. The
gap between the ceiling and 100 % is disagreement between people, not model
error — so the meaningful range runs from chance to the ceiling and is roughly a
third as wide as the nominal scale.
"""

from __future__ import annotations


def human_human_ceiling(triplets) -> float:
    """Estimate how often two participants agree on the same triplet.

    Use repeated triplets — the same three objects answered by more than one
    person — and measure how often the second answer matches the first. State
    in the docstring whichever estimator you settle on, since every percentage
    in the supplement is read against this number.
    """
    raise NotImplementedError


def free_embedding_accuracy() -> float:
    """hSPOSE's own held-out triplet accuracy, from stage 1's summary."""
    raise NotImplementedError


def run() -> None:
    """Write the three anchors to results/ for the reference-points table."""
    raise NotImplementedError
