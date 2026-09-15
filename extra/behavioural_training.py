"""
extra — training each model toward human judgements.

Not one of the supplement's sections: it asks what § Ridge mapping leaves open,
whether training on human answers gets a model closer than a fitted map does.

Ported from things-sim `notebooks/hspose_alignment/behavioural_training/01-04_train_<model>.ipynb`.

Mahner et al. (2025) fit a sparse embedding to VGG-16's *own* odd-one-out choices
to get interpretable "DNN dimensions". Here the complementary question is asked
for every model: if we train directly on **real human** answers, how close to the
~64 % a free human embedding reaches can each model's frozen features get?

Three embeddings per model, all with the hSPOSE loss (dot-product softmax choice
model + L1 sparsity + non-negativity), differing in labels and in capacity:

  A  paper-faithful — generate the model's own choices from its raw features and
     fit a free [1854, 90] embedding to those. Reports held-out accuracy on the
     model's own choices and how often that embedding agrees with real humans.
  B  linear read-out — frozen features -> one trainable linear layer -> ReLU ->
     90-d embedding, trained on real human triplets.
  C  the same protocol with one hidden layer, so the read-out can be nonlinear.

**Why B and C use an object-level split.** An earlier version held out triplets
only, so train and test shared all 1823 objects. That measured nothing about the
representation: with >= 1024 input dimensions the linear map can use the features
as near-unique object identifiers and memorise a good vector per object, which
generalises fine to new triplets of the same objects. The control was
unambiguous — random Gaussian features scored 62.07 % against real ViT's 64.06 %
(chance 33.3 %), with all four models within 0.15 points of each other, a capacity
ceiling rather than a property of any model. Both branches now use the same
5-fold object-level CV as § Ridge mapping (identical folds), each fold training only on
triplets whose three objects are all training objects and emitting embeddings for
the held-out objects. Memorisation is then impossible: a held-out object's vector
must come from its features alone. The random-feature control is run through the
identical pipeline and should collapse toward chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import config
from setup.hspose import DEVICE, train_embedding, triplet_loss_and_acc
from ridge_mapping.ridge import folds


def relabel_with_model(features: np.ndarray, triplets: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Branch A's labels: replace each human answer with the model's own choice.

    Same cosine rule as everywhere else, but the output keeps the triplet layout
    [kept, kept, odd one out], so the relabelled array can be fed to the same
    choice model that the human triplets are fed to.
    """
    vectors = features.copy()                                     # do not touch the caller's feature matrix
    vectors[np.isnan(vectors).any(axis=1)] = 0.0                  # image-less objects contribute nothing
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)  # unit length -> dot product is cosine

    out = triplets.copy()
    for start in range(0, len(triplets), chunk):
        batch = triplets[start:start + chunk]
        a, b, c = batch[:, 0], batch[:, 1], batch[:, 2]
        winner = np.stack([                                       # which pair this model keeps together
            (vectors[a] * vectors[b]).sum(1),
            (vectors[a] * vectors[c]).sum(1),
            (vectors[b] * vectors[c]).sum(1),
        ], axis=1).argmax(1)
        out[start:start + chunk, 0] = np.where(winner == 0, a, np.where(winner == 1, a, b))  # kept pair, first object
        out[start:start + chunk, 1] = np.where(winner == 0, b, np.where(winner == 1, c, c))  # kept pair, second object
        out[start:start + chunk, 2] = np.where(winner == 0, c, np.where(winner == 1, b, a))  # the model's odd one out
    return out


def standardise(features: np.ndarray) -> torch.Tensor:
    """Z-score the features over the valid objects and put them on the device.

    Image-less rows become zeros rather than NaN so they can pass through the
    read-out harmlessly; no triplet ever indexes them.
    """
    missing = np.isnan(features).any(axis=1)
    mean, std = features[~missing].mean(axis=0), features[~missing].std(axis=0) + 1e-6  # statistics from real rows only
    standardised = np.where(missing[:, None], 0.0, (features - mean) / std)
    return torch.from_numpy(standardised.astype(np.float32)).to(DEVICE)


def triplets_within(triplets: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """Keep the triplets whose three objects all belong to a given object set."""
    return triplets[object_mask[triplets].all(axis=1)]            # a fold trains only on its own objects' triplets


def fresh_readout(d_in: int, hidden: int | None):
    """Initialise a read-out and its learnable softmax scale.

    `hidden=None` gives branch B's single linear layer; an integer gives branch
    C's one-hidden-layer MLP. The linear layer's init std is TARGET_EMB_STD /
    sqrt(d) so the softmax starts at the same temperature whatever the feature
    width — otherwise a 4096-d model and a 1024-d model would not start level.
    """
    torch.manual_seed(config.SEED)                                # same initialisation for every model and fold
    log_scale = torch.nn.Parameter(torch.zeros((), device=DEVICE))  # learnable temperature, exp()'d at use
    if hidden is None:
        W = torch.nn.Parameter(torch.normal(0.0, config.TARGET_EMB_STD / np.sqrt(d_in),
                                            size=(d_in, config.INIT_DIM), device=DEVICE))
        return [W, log_scale]
    W1 = torch.nn.Parameter(torch.empty(d_in, hidden, device=DEVICE))
    b1 = torch.nn.Parameter(torch.zeros(hidden, device=DEVICE))
    W2 = torch.nn.Parameter(torch.empty(hidden, config.INIT_DIM, device=DEVICE))
    b2 = torch.nn.Parameter(torch.zeros(config.INIT_DIM, device=DEVICE))
    torch.nn.init.kaiming_normal_(W1, nonlinearity="relu")        # standard init for the ReLU layer
    torch.nn.init.normal_(W2, mean=0.0, std=config.TARGET_EMB_STD / np.sqrt(hidden))  # same temperature argument as above
    return [W1, b1, W2, b2, log_scale]


def readout_embedding(features_t: torch.Tensor, params: list) -> torch.Tensor:
    """Apply a read-out to the standardised features: [N_OBJECTS, INIT_DIM].

    Branch B is one matrix multiply; branch C adds a ReLU hidden layer. The final
    ReLU is applied inside triplet_loss_and_acc, exactly as in the hSPOSE fit, so what is
    optimised is the non-negative embedding in both branches.
    """
    if len(params) == 2:                                          # branch B: [W, log_scale]
        W, _ = params
        return features_t @ W
    W1, b1, W2, b2, _ = params                                    # branch C: one hidden layer
    return F.relu(features_t @ W1 + b1) @ W2 + b2


def run_object_cv(features_t: torch.Tensor, human_train: np.ndarray, human_test: np.ndarray,
                  valid_idx: np.ndarray, hidden: int | None, lam: float, settings: dict,
                  tag: str, verbose: bool = False) -> np.ndarray:
    """Out-of-fold embedding: every object's vector comes from a fit that never saw it.

    For each fold: train on the triplets whose three objects are all training
    objects, early-stop on held-out triplets *among those same training objects*
    (so held-out objects never influence fitting), then keep only the held-out
    objects' rows of the resulting embedding.
    """
    out_of_fold = np.zeros((config.N_OBJECTS, config.INIT_DIM), dtype=np.float32)
    for fold, (train_positions, test_positions) in enumerate(folds(len(valid_idx))):  # the ridge-mapping folds, unchanged
        train_objects = np.zeros(config.N_OBJECTS, dtype=bool)
        train_objects[valid_idx[train_positions]] = True          # this fold's training objects
        train_t = torch.from_numpy(triplets_within(human_train, train_objects)).to(DEVICE)
        val_t = torch.from_numpy(triplets_within(human_test, train_objects)).to(DEVICE)  # training objects only

        params = fresh_readout(features_t.shape[1], hidden)       # a fresh read-out per fold, same seed
        train_embedding(lambda: (readout_embedding(features_t, params), params[-1].exp()), params,
                        train_t, val_t, lam, settings["lr"], settings["max_epochs"], settings["patience"],
                        f"{tag} f{fold}", verbose)
        with torch.no_grad():
            # Store the post-ReLU embedding: triplet_loss_and_acc applies ReLU internally, so the
            # embedding that was actually optimised is the ReLU'd one — the raw pre-activation
            # output has mixed signs and does not reflect what the read-out learned to produce.
            embedding = F.relu(readout_embedding(features_t, params)).cpu().numpy()
        out_of_fold[valid_idx[test_positions]] = embedding[valid_idx[test_positions]]  # held-out objects only
        print(f"  {tag} fold {fold}: trained on {len(train_t):,} triplets, wrote {len(test_positions)} held-out objects")
    return out_of_fold


def select_on_fold0(features_t: torch.Tensor, human_train: np.ndarray, human_test: np.ndarray,
                    valid_idx: np.ndarray, candidates: list, settings: dict, tag: str) -> tuple:
    """Pick a read-out's hyperparameters on fold 0 only, by validation NLL.

    Selection touches one fold, leaving the other four untouched. `candidates` is
    a list of {"hidden": int | None, "lam": float} — branch B searches lambda,
    branch C searches hidden size and lambda.

    Returns (best candidate, selection table).
    """
    train_positions, _ = folds(len(valid_idx))[0]
    train_objects = np.zeros(config.N_OBJECTS, dtype=bool)
    train_objects[valid_idx[train_positions]] = True
    train_t = torch.from_numpy(triplets_within(human_train, train_objects)).to(DEVICE)
    val_t = torch.from_numpy(triplets_within(human_test, train_objects)).to(DEVICE)

    rows = []
    for candidate in candidates:
        params = fresh_readout(features_t.shape[1], candidate["hidden"])
        _, nll = train_embedding(lambda: (readout_embedding(features_t, params), params[-1].exp()), params,
                                 train_t, val_t, candidate["lam"], settings["lr"], settings["max_epochs"],
                                 settings["patience"], f"sel {candidate}", False)
        rows.append({"hidden": candidate["hidden"], "lambda": candidate["lam"], "fold0_val_nll": round(nll, 4)})
        print(f"  {tag} {candidate}: fold-0 validation NLL {nll:.4f}")
    table = pd.DataFrame(rows)
    best = table.loc[table.fold0_val_nll.idxmin()]                # lowest held-out NLL on fold 0 wins
    return {"hidden": None if pd.isna(best["hidden"]) else int(best["hidden"]), "lam": float(best["lambda"])}, table


def branch_a(features: np.ndarray, human_train: np.ndarray, human_test: np.ndarray):
    """Branch A — a free embedding fit to the model's *own* choices.

    The paper's procedure, unchanged: relabel the human triplets with the model's
    answers, fit a free [1854, 90] embedding to those with the paper's sparsity,
    then report both how well it predicts the model's own choices and how often it
    agrees with real people.

    Returns (pruned embedding, accuracy on own choices, agreement with humans).
    """
    model_train = torch.from_numpy(relabel_with_model(features, human_train)).to(DEVICE)  # the model's own answers
    model_test = torch.from_numpy(relabel_with_model(features, human_test)).to(DEVICE)
    human_test_t = torch.from_numpy(human_test).to(DEVICE)        # the real answers, for the agreement number

    torch.manual_seed(config.SEED)
    embedding = torch.nn.Parameter(                               # free embedding, exactly as in the hSPOSE fit
        torch.normal(0.1, 0.01, size=(config.N_OBJECTS, config.INIT_DIM), device=DEVICE))
    train_embedding(lambda: (embedding, 1.0), [embedding], model_train, model_test,
                    config.BRANCH_A["lam"], config.BRANCH_A["lr"], config.BRANCH_A["max_epochs"],
                    config.BRANCH_A["patience"], "A/own-choices")

    with torch.no_grad():
        _, own = triplet_loss_and_acc(embedding, model_test)      # does it predict the model's own choices?
        _, human = triplet_loss_and_acc(embedding, human_test_t)  # does that make it human-like? (it does not)
    return embedding.detach().cpu().numpy(), float(own), float(human)
