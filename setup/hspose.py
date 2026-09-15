"""
§ Setup -> Building the yardstick — hSPOSE, the human reference.

Ported from things-sim `notebooks/hspose_alignment/human_spose/train_human_spose.ipynb`,
with the training loop factored out because extra/behavioural_training trains three more embeddings
with exactly the same loop and the same loss.

SPoSE (Hebart et al., 2020) turns millions of odd-one-out responses into a short
list of numbers per concept, every number non-negative and most of them zero.
Non-negativity makes the recovered dimensions readable as ordinary attributes
(*is food*, *is metallic*, *is animate*): a dimension can say an object has a
property strongly, weakly or not at all, never negatively. The L1 sparsity leaves
each object described by the few properties that genuinely apply to it.

The embedding is fitted to people's *choices*, not to similarity ratings, and we
refit rather than download the published embedding so the human reference comes
from the same objects and the same split as everything compared against it.
things-sim's run: 64.12 % held out (paper 64.60 %), 52 dimensions kept (paper 49),
RSM r = 0.963 against the published 49-d embedding.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"          # one device decision for the whole step


def triplet_loss_and_acc(embedding: torch.Tensor, triplets: torch.Tensor, scale: float = 1.0):
    """The SPoSE choice model on one batch of triplets: (mean NLL, accuracy).

    A triplet row is [i, j, k] with k the odd one out, so the pair (i, j) is the
    participant's "most similar" pair and is always the correct answer — column 0
    below.

    `scale` is the softmax temperature. A free embedding leaves it at 1.0, since
    it grows its own magnitude and magnitude *is* the temperature here. A linear
    read-out (extra/behavioural_training) cannot: becoming confident means scaling all of W uniformly,
    a direction gradient descent barely moves, so one learnable scalar is passed
    in instead. It adds no capacity — logits are bilinear in the embedding and
    ReLU is positively homogeneous, so scale s with weights W is exactly scale 1
    with weights sqrt(s)·W.
    """
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    xi, xj, xk = F.relu(embedding[i]), F.relu(embedding[j]), F.relu(embedding[k])  # the non-negative part is the embedding
    logits = torch.stack([                                       # one score per candidate pairing
        (xi * xj).sum(1),                                        # the pair the participant kept -> correct, column 0
        (xi * xk).sum(1),
        (xj * xk).sum(1),
    ], dim=1) * scale
    log_p = F.log_softmax(logits, dim=1)                         # softmax over the three pairings
    return -log_p[:, 0].mean(), (log_p.argmax(dim=1) == 0).float().mean()  # cross-entropy on column 0, and how often it wins


def sparsity_penalty(embedding: torch.Tensor, lam: float) -> torch.Tensor:
    """L1 sparsity plus a soft push toward non-negativity.

    lam = 0.008 is the paper's cross-validated value, used for every free
    embedding; the trained read-outs in extra/ cross-validate lam for their own
    parameterisation rather than inheriting the constant.
    """
    if lam == 0.0:                                               # branch B/C may select no sparsity at all
        return torch.zeros((), device=embedding.device)
    l1 = (lam / embedding.shape[0]) * embedding.abs().sum()      # scaled per object, so lam means the same at any N
    positive = 0.01 * F.relu(-embedding).sum()                   # ReLU is flat below zero, so negatives need their own push
    return l1 + positive


def train_embedding(get_state, params, train_t, val_t, lam, lr, max_epochs, patience, tag, verbose=True):
    """Adam + early stopping on held-out NLL; restores the best weights.

    `get_state()` returns (embedding, scale) — a free parameter for hSPOSE and
    branch A, a read-out of frozen features for branches B and C — so the same
    loop trains all four. `params` is what the optimiser owns.

    Returns (history, best held-out NLL).
    """
    optimiser = torch.optim.Adam(params, lr=lr)
    history = {"train_nll": [], "val_nll": [], "val_acc": []}
    best_nll, best_state, waited = float("inf"), None, 0         # early-stopping state
    n = len(train_t)

    for epoch in range(1, max_epochs + 1):
        order = torch.randperm(n, device=DEVICE)                 # reshuffle so batches are not correlated with response order
        total, batches = 0.0, 0
        for start in range(0, n, config.BATCH_SIZE):
            embedding, scale = get_state()                       # recomputed per batch: for a read-out it depends on W
            nll, _ = triplet_loss_and_acc(embedding, train_t[order[start:start + config.BATCH_SIZE]], scale)
            loss = nll + sparsity_penalty(embedding, lam)        # fit the choices, stay sparse, stay non-negative
            optimiser.zero_grad()                                # gradients do not accumulate across batches
            loss.backward()
            optimiser.step()
            total += nll.item()
            batches += 1

        with torch.no_grad():                                    # held-out pass: scoring only
            embedding, scale = get_state()
            val_nll, val_acc = triplet_loss_and_acc(embedding, val_t, scale)
        history["train_nll"].append(total / batches)
        history["val_nll"].append(val_nll.item())
        history["val_acc"].append(val_acc.item())
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  [{tag}] ep {epoch:3d}  train nll {total/batches:.4f}  val nll {val_nll.item():.4f}  val acc {val_acc.item():.4f}")

        if val_nll.item() < best_nll - 1e-4:                     # a real improvement, not optimiser noise
            best_nll = val_nll.item()
            best_state = [p.detach().clone() for p in params]    # keep the best parameters, not the last ones
            waited = 0
        else:
            waited += 1                                          # the sparsity penalty keeps shrinking weights,
            if waited >= patience:                               # so stopping late overprunes
                break

    with torch.no_grad():
        for parameter, best in zip(params, best_state):
            parameter.copy_(best)                                # restore the early-stopping winner before returning
    return history, best_nll


def prune(weights: np.ndarray) -> np.ndarray:
    """Drop the dimensions no object uses, then sort by total weight.

    A dimension survives when its largest object weight exceeds config.PRUNE_THR.
    The fit starts from a deliberately generous INIT_DIM, so the final count is
    discovered by the procedure rather than chosen — report it, don't target it.
    """
    keep = np.where(weights.max(axis=0) > config.PRUNE_THR)[0]   # dimensions some object genuinely uses
    order = np.argsort(-np.abs(weights[:, keep]).sum(axis=0))    # most-used dimension first
    return weights[:, keep][:, order]


def fit(train: np.ndarray, test: np.ndarray):
    """Fit the free human embedding on human triplets and prune it.

    Free means the parameters *are* the object weights, with no model features as
    input: this is a description of the human data, not of any network.

    Returns (hspose [N_OBJECTS, k], held-out accuracy, history).
    """
    train_t = torch.from_numpy(train).to(DEVICE)                 # whole split on device: it is only indices
    test_t = torch.from_numpy(test).to(DEVICE)                   # held out — it never enters the optimiser

    torch.manual_seed(config.SEED)                               # the only randomness: initialisation and batch order
    embedding = torch.nn.Parameter(                              # free [1854, 90] embedding, exactly as in the paper
        torch.normal(mean=0.1, std=0.01, size=(config.N_OBJECTS, config.INIT_DIM), device=DEVICE))

    history, _ = train_embedding(lambda: (embedding, 1.0), [embedding], train_t, test_t,
                                 config.LAMBDA_L1, config.LR, config.MAX_EPOCHS, config.PATIENCE, "hSPOSE")

    with torch.no_grad():                                        # final score from the restored best weights
        _, accuracy = triplet_loss_and_acc(embedding, test_t)
    print(f"held-out accuracy: {accuracy.item():.4f}   (chance {config.CHANCE:.4f}, "
          f"paper {config.PAPER_ACCURACY:.4f}, ceiling {config.NOISE_CEILING:.4f})")

    hspose = prune(embedding.detach().cpu().numpy())             # [1854, k] — k is what the pruning discovered
    print(f"kept {hspose.shape[1]} / {config.INIT_DIM} dimensions   (paper's human model: {config.PAPER_N_DIMS})")
    return hspose, float(accuracy), history
