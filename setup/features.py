"""
§ Setup -> The five models — frozen feature extraction.

Ported from things-sim `notebooks/0{1,2,3,5}_*_embeddings.ipynb`, which ran one
notebook per model with an identical skeleton and a different loader cell; the
loader cells are the `if` branches below.

Each model is shown the THINGS *reference* photograph of every concept and the
representation it produces is recorded. The models are frozen: eval mode, no
gradients, no weight updated anywhere in this repo.

Output: data/features/<model>.npy, [N_OBJECTS, d] float32, all-NaN row for any
concept with no usable image.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from PIL import Image

from common import config


def reference_image(object_name: str):
    """Locate one THINGS reference photograph for a concept, or None.

    The folder naming in THINGS is inconsistent — spaces become underscores,
    hyphens are sometimes kept and sometimes not, and homonyms are numbered
    (*bat* -> *bat1*, *bat2*) — so each spelling is tried in a fixed order. The
    file ending in "b" is the dataset's designated reference image; if a folder
    has none, its first exemplar is used instead.
    """
    base = object_name.replace(" ", "_")                         # spaces -> underscores (folders keep hyphens)
    folder = None
    for variant in dict.fromkeys([base, base.replace("-", "_"), base.replace("-", "")]):  # de-duped, tried in order
        for candidate in (config.IMAGES / variant, config.IMAGES / (variant + "1")):      # exact folder, then sense 1
            if candidate.is_dir():
                folder = candidate
                break
        if folder is None:                                       # otherwise the first numbered sense, e.g. bat2
            numbered = sorted(config.IMAGES.glob(variant + "[0-9]"))
            folder = numbered[0] if numbered else None
        if folder is not None:
            break
    if folder is None:                                           # no folder under any spelling -> this row stays NaN
        return None
    reference = sorted(folder.glob("*b.jpg"))                    # THINGS reference image ends in 'b'
    if reference:
        return reference[0]
    exemplars = sorted(folder.glob("*.jpg"))                     # fall back to the first exemplar
    return exemplars[0] if exemplars else None


def build_model(model_name: str):
    """Instantiate one frozen backbone and its own preprocessing.

    Returns (forward, transform), where `forward` maps a preprocessed batch to
    one pooled vector per image. Each backbone must use the preprocessing it was
    trained with: a mismatched resize or normalisation quietly degrades that
    model's row in every table downstream.
    """
    spec = config.MODELS[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"      # the pinned GPU if there is one

    if spec["loader"] == "timm":                                 # EVA02, DINOv3, CLIP image tower, VGG-16, ViT
        import timm
        from timm.data import create_transform, resolve_data_config
        model = timm.create_model(spec["weights"], pretrained=True, num_classes=0)  # num_classes=0 -> pooled features
        model.eval().to(device)                                  # eval mode + move to the device
        transform = create_transform(**resolve_data_config({}, model=model))        # the checkpoint's own preprocessing
        forward = model                                          # timm models already return the pooled embedding
        # NOTE for VGG-16: model.num_features is the 512 conv-channel count, not the output
        # width — num_classes=0 keeps the fc7 head, so model(x) is 4096-d (config asserts it).

    elif spec["loader"] == "imic":                               # IMIC: EVA02 after person re-identification training
        import torchvision.transforms as T
        sys.path.insert(0, str(config.IMIC_REPO))                # the repo that defines EchoBID
        from model_definitions import EchoBID, ResizeOntoBlackSquare
        state = torch.load(spec["weights"], map_location="cpu")  # the re-identification checkpoint
        state = state.get("model_state_dict", state)             # unwrap if it is nested
        state = {key.replace("module.", ""): value for key, value in state.items()}  # strip the DataParallel prefix
        model = EchoBID(model=config.IMIC_BACKBONE,              # EVA02-large backbone
                        n_training_ids=state["query_fc.weight"].shape[0],   # number of training identities
                        model_output_dim=state["head.weight"].shape[0])     # penultimate embedding width
        model.load_weights(spec["weights"], map_location=device)  # load the trained weights through EchoBID's own loader
        model.eval().to(device)
        transform = T.Compose([                                  # IMIC's own preprocessing pipeline
            ResizeOntoBlackSquare(target_size=448),              # pad + resize onto a 448x448 black square
            T.ToTensor(),                                        # HWC uint8 -> CHW float in [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406],              # ImageNet normalisation
                        std=[0.229, 0.224, 0.225]),
        ])
        forward = lambda batch: model(batch)[0]                  # EchoBID returns a tuple; [0] is the penultimate embedding

    else:
        raise ValueError(f"unknown loader {spec['loader']!r} for model {model_name!r}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)                          # belt and braces — nothing here is ever trained
    return forward, transform


def extract(model_name: str) -> np.ndarray:
    """Run one frozen model over every concept's reference image.

    Returns [N_OBJECTS, d] float32 in image-index row order, with an all-NaN row
    for any concept that has no usable image — a hole, never a shifted row.
    """
    spec = config.MODELS[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward, transform = build_model(model_name)                 # backbone + its matching preprocessing

    index = json.loads(config.IMAGE_INDEX.read_text())           # {object name: row index} — defines the row order
    paths = [None] * config.N_OBJECTS                            # one image path per row; None = no image
    for object_name, row in index.items():
        paths[row] = reference_image(object_name)
    todo = [(row, path) for row, path in enumerate(paths) if path is not None]  # the rows we can actually fill
    print(f"[{model_name}] rows: {config.N_OBJECTS} | with image: {len(todo)} | NaN rows: {config.N_OBJECTS - len(todo)}")

    features = np.full((config.N_OBJECTS, spec["dim"]), np.nan, dtype=np.float32)  # start all-NaN: missing rows stay NaN
    for start in range(0, len(todo), config.BATCH_IMAGES):
        chunk = todo[start:start + config.BATCH_IMAGES]          # a list of (row, path), fixed order for reproducibility
        images = [transform(Image.open(path).convert("RGB")) for _, path in chunk]  # RGB: some THINGS photos are greyscale
        with torch.no_grad():                                    # inference only; also keeps memory flat
            out = forward(torch.stack(images).to(device))        # [B, d] pooled representation
        out = out.float().cpu().numpy()                          # back to float32 numpy for storage
        assert out.shape[1] == spec["dim"], f"{model_name}: got width {out.shape[1]}, expected {spec['dim']}"  # wrong layer?
        for (row, _), vector in zip(chunk, out):
            features[row] = vector                               # scatter into the object's own row, never append
        print(f"[{model_name}] {start + len(chunk)}/{len(todo)}", end="\r")  # single-line progress for a slow step

    print(f"[{model_name}] done | NaN rows: {int(np.isnan(features).any(axis=1).sum())}")
    return features
