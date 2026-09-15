"""
extra — can an MLP beat RidgeCV at mapping a model into hSPOSE?

Not one of the supplement's sections: it tests whether linearity is the limit
in § Ridge mapping.

Ported from things-sim `notebooks/hspose_alignment/ridge_mapping/09-12_ridge_nonlinear_<model>.ipynb`.

Exactly the linear protocol of ridge_map.py — the same folds (they depend only on
the number of objects and the seed), the same target, the same per-dimension and
odd-one-out scoring — with RidgeCV replaced by a small MLP (d -> hidden -> k,
ReLU, trained by MSE), whose hidden size and weight decay are selected on fold 0
only.

This asks a narrower question than behavioural training's branch C: there, a nonlinear read-out
was compared against a linear one under the strict human-triplet protocol; here
both versions regress onto the *same* hSPOSE target, which isolates "does
nonlinearity help this regression" from "does the strict protocol get in the way".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from common import config
from ridge_mapping.ridge import folds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLPRegressor(nn.Module):
    """d -> hidden -> k with a ReLU between.

    Trained by MSE against the unstandardised hSPOSE vector; only the input X is
    standardised, matching RidgeCV's own StandardScaler-on-X-only convention, so
    the two maps differ in exactly one respect.
    """

    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def inner_split(n_train: int):
    """A fixed-seed slice of a training fold's own objects, for early stopping only.

    RidgeCV does not need this — its alpha comes from built-in generalised CV —
    but a hand-trained MLP does, and the slice must come from the training fold so
    the held-out objects stay untouched.
    """
    generator = np.random.default_rng(config.SEED)
    positions = np.arange(n_train)
    generator.shuffle(positions)                                  # fixed seed: the same slice on every call
    n_val = max(1, int(n_train * config.NL_VAL_FRACTION))
    return positions[n_val:], positions[:n_val]                   # (inner train, inner validation)


def fit_mlp(X_train, Y_train, X_val, Y_val, hidden: int, weight_decay: float, tag: str = "", verbose: bool = False):
    """Train one MLP to convergence with early stopping on the inner split.

    Full-batch Adam: 1,400-odd objects fit comfortably on the device at once, and
    full-batch descent makes the run deterministic given the seed. Returns
    (model, best validation MSE).
    """
    torch.manual_seed(config.SEED)                                # same initialisation for every candidate and fold
    model = MLPRegressor(X_train.shape[1], hidden, Y_train.shape[1]).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.NL_LR, weight_decay=weight_decay)
    X_train_t = torch.from_numpy(X_train).to(DEVICE)              # moved once, reused every epoch
    Y_train_t = torch.from_numpy(Y_train).to(DEVICE)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    Y_val_t = torch.from_numpy(Y_val).to(DEVICE)

    best_mse, best_state, waited = float("inf"), None, 0
    for epoch in range(1, config.NL_MAX_EPOCHS + 1):
        model.train()
        optimiser.zero_grad()
        loss = F.mse_loss(model(X_train_t), Y_train_t)            # regression onto the human dimensions
        loss.backward()
        optimiser.step()
        model.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(model(X_val_t), Y_val_t).item()  # early-stopping criterion
        if verbose and (epoch % 40 == 0 or epoch == 1):
            print(f"  [{tag}] ep {epoch:4d}  train mse {loss.item():.5f}  val mse {val_mse:.5f}")
        if val_mse < best_mse - 1e-7:
            best_mse = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # keep the best, not the last
            waited = 0
        else:
            waited += 1
            if waited >= config.NL_PATIENCE:
                break
    model.load_state_dict(best_state)                             # restore the early-stopping winner
    return model, best_mse


def select_architecture(X: np.ndarray, Y: np.ndarray):
    """Choose hidden size and weight decay on fold 0 only.

    Selection touches one fold, so the other four remain untouched held-out
    objects — the same discipline branch C uses in extra/behavioural_training. Returns
    (hidden, weight_decay, selection table).
    """
    train, _ = folds(len(X))[0]                                   # fold 0's training objects
    scaler = StandardScaler().fit(X[train])                       # scaled like the real fits, on the training fold
    X_scaled = scaler.transform(X[train]).astype(np.float32)
    inner_train, inner_val = inner_split(len(train))

    rows = []
    for hidden in config.NL_HIDDEN_GRID:                          # the same grid the notebooks searched
        for weight_decay in config.NL_WEIGHT_DECAY_GRID:
            _, val_mse = fit_mlp(X_scaled[inner_train], Y[train][inner_train].astype(np.float32),
                                 X_scaled[inner_val], Y[train][inner_val].astype(np.float32),
                                 hidden, weight_decay, tag=f"h={hidden} wd={weight_decay:g}")
            rows.append({"hidden": hidden, "weight_decay": weight_decay, "val_mse": round(val_mse, 6)})
            print(f"  hidden {hidden:4d}  wd {weight_decay:g}: val MSE {val_mse:.6f}")
    table = pd.DataFrame(rows)                                    # the selection CSV saved next to the results
    best = table.loc[table.val_mse.idxmin()]                      # lowest validation MSE wins
    return int(best["hidden"]), float(best["weight_decay"]), table


def mlp_out_of_fold(X: np.ndarray, Y: np.ndarray, hidden: int, weight_decay: float) -> np.ndarray:
    """Out-of-fold MLP predictions, the same folds and guarantee as the ridge map."""
    Y_oof = np.zeros_like(Y)
    for fold, (train, test) in enumerate(folds(len(X))):
        scaler = StandardScaler().fit(X[train])                   # training-fold statistics only
        X_train = scaler.transform(X[train]).astype(np.float32)
        X_test = scaler.transform(X[test]).astype(np.float32)
        inner_train, inner_val = inner_split(len(train))          # early-stopping slice from inside the training fold
        model, val_mse = fit_mlp(X_train[inner_train], Y[train][inner_train].astype(np.float32),
                                 X_train[inner_val], Y[train][inner_val].astype(np.float32),
                                 hidden, weight_decay, tag=f"fold{fold}")
        model.eval()
        with torch.no_grad():
            Y_oof[test] = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()  # predict the held-out objects
        print(f"fold {fold}: val mse {val_mse:.6f}, wrote {len(test)} held-out objects")
    return Y_oof
