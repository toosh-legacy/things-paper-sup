"""
The pieces more than one step needs: loading, the valid-object mask, cosine
geometry, the odd-one-out decision rule, and saving.

Two rules hold the whole project together and both live here.

1. **Row i is the same THINGS object in every array** — every model's feature
   matrix, hSPOSE, the published embedding, and every index inside a triplet.

2. **One shared valid-object mask.** A few concepts have no usable image, so
   their feature rows are NaN (things-sim: 31 of 1854, identical in all four
   model files, leaving 1823 objects). Those objects, and every triplet
   mentioning one of them, are dropped once for every model — so no model is
   ever scored on an easier set than another.

The odd-one-out rule lives here rather than in odd_one_out/ because the ridge, odd-one-out and extra sections
all score with it, and they must score identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import config


def load_features(model: str) -> np.ndarray:
    """Return one model's frozen feature matrix, [N_OBJECTS, d], NaN rows kept."""
    array = np.load(config.FEATURES / f"{model}.npy")          # written by the setup section, never recomputed later
    assert array.shape[0] == config.N_OBJECTS, f"{model}: {array.shape[0]} rows, expected {config.N_OBJECTS}"  # row drift corrupts every table
    assert array.shape[1] == config.MODELS[model]["dim"], f"{model}: width {array.shape[1]}, expected {config.MODELS[model]['dim']}"  # wrong layer or checkpoint
    return array


def load_triplets(split: str) -> np.ndarray:
    """Return the human odd-one-out triplets for "train" or "test", [n, 3] int64.

    A row is [i, j, k] with k the object the participant called the odd one out,
    so (i, j) is the pair they kept together and is the correct answer for every
    scoring function in the project.
    """
    triplets = np.load(config.TRIPLETS[split]).astype(np.int64)  # int64 so the array can index directly
    assert triplets.ndim == 2 and triplets.shape[1] == 3, f"{split}: expected [n, 3], got {triplets.shape}"
    return triplets


def load_hspose() -> np.ndarray:
    """Return the refit human embedding from the setup section, [N_OBJECTS, k]."""
    return np.load(config.HSPOSE).astype(float)                # float64 everywhere downstream, as in things-sim


def valid_mask(models: list[str] | None = None) -> np.ndarray:
    """Boolean mask over objects, [N_OBJECTS], True where the object is usable.

    An object is valid when no model has a NaN row for it. Computing it across
    all models at once is what makes the object set identical everywhere.
    """
    mask = np.ones(config.N_OBJECTS, dtype=bool)               # assume every concept is usable...
    for model in (models or config.MODEL_ORDER):               # ...then take away the ones any model failed on
        mask &= ~np.isnan(load_features(model)).any(axis=1)    # a NaN row means "this concept had no usable image"
    return mask


def drop_invalid_triplets(triplets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only the triplets whose three objects are all valid."""
    return triplets[mask[triplets].all(axis=1)]                # mask[triplets] is [n, 3] of booleans; all three must hold


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """[n, n] cosine similarity — the geometry every comparison in the project uses."""
    normalised = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # unit length; +1e-12 keeps an all-zero row finite
    return normalised @ normalised.T                           # dot products of unit vectors = cosines


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Each unordered object pair once, diagonal excluded (1823 objects -> 1,660,753 pairs)."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]       # k=1 drops the self-similarity diagonal


def odd_one_out_percent(vectors: np.ndarray, triplets: np.ndarray, rule: str = "cosine",
                        chunk: int = 200_000, tiebreak_seed: int | None = None) -> float:
    """Percentage of triplets where the model's odd one out matches the human's.

    `rule="cosine"` is the decision rule used for raw features, ridge-mapped
    vectors and trained read-outs alike; `rule="dot"` is the SPoSE rule (dot
    product on the non-negative embedding), comparable with hSPOSE itself.

    `tiebreak_seed` adds a tiny fixed-seed jitter before the argmax. It is needed
    only for an embedding that can collapse to all zeros (extra/behavioural_training's random-feature
    control): with all similarities equal, argmax silently returns index 0, which
    would score every triplet correct because the data format always stores the
    true odd one out in column 2 — a scoring artefact, not a signal.
    """
    if rule == "cosine":                                       # cosine: direction only, magnitude ignored
        vectors = vectors.astype(np.float64).copy()            # copy: the caller's array is not ours to modify
        vectors[np.isnan(vectors).any(axis=1)] = 0.0           # image-less objects become zero rows...
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0                                # ...and are left at zero instead of dividing by zero
        vectors = vectors / norms
    elif rule == "dot":                                        # SPoSE rule: dot product of the non-negative part
        vectors = np.clip(np.nan_to_num(vectors.astype(np.float64)), 0, None)
    else:
        raise ValueError(f"unknown rule {rule!r}")

    generator = np.random.default_rng(tiebreak_seed) if tiebreak_seed is not None else None
    correct = 0                                                # triplets answered the way the participant answered them
    for start in range(0, len(triplets), chunk):               # chunked: there are hundreds of thousands of triplets
        batch = triplets[start:start + chunk]
        a, b, c = batch[:, 0], batch[:, 1], batch[:, 2]        # the three objects the participant saw
        similarities = np.stack([                              # the three possible "most similar pair" outcomes
            (vectors[a] * vectors[b]).sum(1),                  # 0: (a, b) stay together -> odd one out is c
            (vectors[a] * vectors[c]).sum(1),                  # 1: (a, c) stay together -> odd one out is b
            (vectors[b] * vectors[c]).sum(1),                  # 2: (b, c) stay together -> odd one out is a
        ], axis=1)
        if generator is not None:
            similarities = similarities + generator.uniform(-1e-9, 1e-9, size=similarities.shape)  # break exact ties at random
        winner = similarities.argmax(1)                        # which pair the model keeps together
        odd = np.where(winner == 0, c, np.where(winner == 1, b, a))  # the leftover object is the model's answer
        correct += int((odd == batch[:, 2]).sum())             # column 2 is the human's answer
    return 100.0 * correct / len(triplets)


def odd_one_out_predictions(vectors: np.ndarray, triplets: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """The same cosine rule, returning reordered triplets instead of a percentage.

    Output rows are [kept, kept, odd one out], the same layout as the human rows,
    so a model's answers can be saved and compared row for row.
    """
    vectors = vectors.astype(np.float64).copy()                # same normalisation as odd_one_out_percent
    vectors[np.isnan(vectors).any(axis=1)] = 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = (vectors / norms).astype(np.float32)             # float32 is enough once normalised, and halves the memory

    out = np.empty_like(triplets)                              # same shape as the input triplets
    for start in range(0, len(triplets), chunk):
        batch = triplets[start:start + chunk]
        a, b, c = batch[:, 0], batch[:, 1], batch[:, 2]
        winner = np.stack([                                    # identical scoring to odd_one_out_percent
            (vectors[a] * vectors[b]).sum(1),
            (vectors[a] * vectors[c]).sum(1),
            (vectors[b] * vectors[c]).sum(1),
        ], axis=1).argmax(1)
        out[start:start + chunk, 0] = np.where(winner == 0, a, np.where(winner == 1, a, b))  # first kept object
        out[start:start + chunk, 1] = np.where(winner == 0, b, np.where(winner == 1, c, c))  # second kept object
        out[start:start + chunk, 2] = np.where(winner == 0, c, np.where(winner == 1, b, a))  # the odd one out
    return out


def save_array(path, array: np.ndarray) -> None:
    """Save a float32 .npy, creating the step's results directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)             # results/ is gitignored, so it may not exist yet
    np.save(path, array.astype(np.float32))                    # one dtype on disk everywhere


def save_table(path, rows) -> None:
    """Write a step's headline numbers as a CSV, one record per row.

    These CSVs are what the supplement's tables are read off, so no number in
    report/ is ever transcribed by hand from a console log.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)  # accept a frame or a list of dicts
    frame.to_csv(path, index=False)
    print(f"wrote {path}")
