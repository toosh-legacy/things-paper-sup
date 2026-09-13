"""
Loading, the valid-object mask, and saving.

One rule holds the whole project together: **row i is the same THINGS object in
every array** — every model's feature matrix, hSPOSE, the published SPoSE-49d
file, and the indices inside every triplet. Nothing here may reorder rows.

The second rule is the fair-comparison one: a small number of concepts have no
usable image, so their feature rows are NaN. Those objects, and every triplet
that mentions one of them, are dropped **once**, from the same mask, for every
model — so no model is scored on an easier set than another.
"""

from __future__ import annotations

import numpy as np


def load_features(model: str) -> np.ndarray:
    """Return the frozen feature matrix for one model, [N_OBJECTS, d].

    Reads config.FEATURES / f"{model}.npy". Assert the row count is N_OBJECTS
    and, if config.MODELS[model]["dim"] is set, that the width matches — a
    silent shape drift here corrupts every downstream table.
    """
    raise NotImplementedError


def load_triplets(split: str) -> np.ndarray:
    """Return the human odd-one-out triplets for "train" or "test", [n, 3] int64.

    Each row is [i, j, k] with k the object the participant called the odd one
    out — so (i, j) is the pair they treated as most similar, and it is the
    correct answer for every scoring function in the project.

    The test split is held out everywhere: it trains neither hSPOSE nor any
    ridge map, so it can score both without inflation.
    """
    raise NotImplementedError


def load_image_index() -> dict:
    """Return the object-name -> row-index mapping used to align everything."""
    raise NotImplementedError


def load_published_spose() -> np.ndarray:
    """Return Hebart et al.'s published 49-d embedding, [N_OBJECTS, 49].

    Used once, as the third reproduction check: does our refit describe the same
    similarity structure as theirs.
    """
    raise NotImplementedError


def load_hspose() -> np.ndarray:
    """Return the refit human embedding produced by stage 1, [N_OBJECTS, k]."""
    raise NotImplementedError


def valid_mask(models: list[str] | None = None) -> np.ndarray:
    """Boolean mask over objects, True where the object is usable, [N_OBJECTS].

    An object is valid when no model in `models` (default: all of MODEL_ORDER)
    has a NaN row for it. Computing it across all models rather than per model
    is what makes the object set identical everywhere.
    """
    raise NotImplementedError


def drop_invalid_triplets(triplets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Drop every triplet mentioning an invalid object.

    Keep a row only when mask is True for all three of its indices.
    """
    raise NotImplementedError


def ensure_dirs() -> None:
    """Create the results/ subdirectories the stages write into."""
    raise NotImplementedError


def save_array(path, array) -> None:
    """Save a float32 .npy, creating parent directories as needed."""
    raise NotImplementedError


def save_summary(path, rows: list[dict]) -> None:
    """Write a stage's headline numbers as a one-row-per-record CSV.

    These CSVs are what tables.py reads, so the supplement's tables are never
    transcribed by hand from a console log.
    """
    raise NotImplementedError
