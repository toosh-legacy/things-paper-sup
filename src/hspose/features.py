"""
Stage 0 — frozen feature extraction.

Each of the five models is shown photographs of the 1,854 THINGS concepts and
the representation it produces for each object is recorded. The models are
**frozen**: no weights are updated at any point in this project, here or later.

Output: config.FEATURES / "<model>.npy", [N_OBJECTS, d], float32, with an
all-NaN row for any concept that has no usable image. Row order is the image
index's order and must match across all five files.

Run once per model; every later stage reads these files and never the images.
"""

from __future__ import annotations

import numpy as np


def build_model(model: str):
    """Instantiate one frozen backbone plus its preprocessing transform.

    Dispatch on config.MODELS[model]["loader"]:
      timm         -> EVA02, DINOv3
      open_clip    -> CLIP image encoder only (the text tower is unused)
      torchvision  -> VGG-16
      local        -> IMIC-B checkpoint, the EVA02 architecture after person
                      re-identification training

    Return (model, transform) with the model in eval mode, requires_grad off,
    and on config device. Use each backbone's *own* preprocessing — a mismatched
    resize or normalisation quietly degrades that model's row in every table.
    """
    raise NotImplementedError


def image_paths() -> list:
    """Return one image path per THINGS concept, in image-index row order.

    Return None in the slot for a concept with no usable image; extract_model
    turns those into NaN rows rather than silently shifting the row order.
    """
    raise NotImplementedError


def pooled_embedding(model, batch, loader: str) -> np.ndarray:
    """Take one batch of preprocessed images to one vector per image.

    Which activation counts as "the representation" differs by loader — CLS
    token vs. pooled patch tokens vs. a named classifier-head layer for VGG-16.
    Decide it once per loader here so the choice is visible in one place.
    """
    raise NotImplementedError


def extract_model(model: str, batch_size: int = 32) -> np.ndarray:
    """Run one model over every concept image and return [N_OBJECTS, d].

    No gradients, fixed batch order, NaN rows for missing images. Assert the
    output width against config.MODELS[model]["dim"] before saving.
    """
    raise NotImplementedError


def run(models: list[str] | None = None) -> None:
    """Extract and save features for each requested model (default: all five)."""
    raise NotImplementedError
