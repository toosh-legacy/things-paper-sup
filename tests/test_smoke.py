"""
Cheap invariants worth asserting before trusting any table.

These are not unit tests of the maths — they are the assumptions the whole
pipeline rests on, and each one has a failure mode that would otherwise show up
as a plausible-looking wrong number in the supplement.
"""


def test_row_alignment():
    """Every feature file, hSPOSE and the published embedding have N_OBJECTS rows."""
    raise NotImplementedError


def test_valid_mask_is_shared():
    """The same objects are dropped for every model — no model gets an easier set."""
    raise NotImplementedError


def test_triplet_indices_in_range():
    """Every triplet index is a real object row, and no triplet repeats an object."""
    raise NotImplementedError


def test_oddoneout_chance_level():
    """Random vectors score near chance (1/3) — catches a flipped decision rule.

    A rule that returned the *most* similar object instead of the leftover one
    still produces sane-looking percentages, so test it against random input.
    """
    raise NotImplementedError


def test_ridge_folds_are_identical_across_models():
    """The fold assignment depends on the seed only, never on the model."""
    raise NotImplementedError
