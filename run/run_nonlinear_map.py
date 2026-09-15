"""
extra — the same map, an MLP instead of ridge.

    python -m run.run_nonlinear_map

Optional: it answers "would a nonlinear map recover more?", not one of the
supplement's main tables. Same folds, same target, same scoring as the linear
version, so the two are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import config, shared
from extra.nonlinear_map import mlp_out_of_fold, select_architecture
from ridge_mapping.ridge import per_dimension_recovery
from raw_similarity.rsa import rsa


def main() -> None:
    """Select, fit, score and save the nonlinear map for every model."""
    mask = shared.valid_mask()
    hspose = shared.load_hspose()[mask].astype(np.float32)        # float32: the MLP trains in single precision
    rows = []
    for model in config.MODEL_ORDER:
        print(f"=== {model} ===")
        features = shared.load_features(model)[mask].astype(np.float32)

        hidden, weight_decay, selection = select_architecture(features, hspose)  # fold 0 only
        shared.save_table(config.RESULTS_RIDGE_MAPPING / f"{model}_nonlinear_selection.csv", selection)
        print(f"selected hidden={hidden}, weight_decay={weight_decay:g}")

        predicted = mlp_out_of_fold(features, hspose, hidden, weight_decay)  # the same five folds as the linear map
        per_dim = per_dimension_recovery(hspose, predicted)       # scored exactly like the linear map
        shared.save_table(config.RESULTS_RIDGE_MAPPING / f"{model}_nonlinear_per_dim_accuracy.csv", per_dim)

        linear = pd.read_csv(config.RESULTS_RIDGE_MAPPING / "ridge_recovery.csv")  # the linear numbers, for the comparison column
        linear_row = linear[linear.model == config.MODELS[model]["label"]].iloc[0]

        full = np.zeros((config.N_OBJECTS, hspose.shape[1]), dtype=np.float32)
        full[mask] = predicted                                    # full size, for the odd-one-out scoring
        shared.save_array(config.RESULTS_RIDGE_MAPPING / f"{model}_nonlinear_hspose_vector.npy", full)

        rows.append({
            "model": config.MODELS[model]["label"],
            "hidden": hidden, "weight_decay": weight_decay,
            "mean_pearson_r": round(float(per_dim.pearson_r.mean()), 4),
            "mean_r2": round(float(per_dim.r2.mean()), 4),
            "rsa_nonlinear_predicted": round(rsa(hspose, predicted), 4),
            "linear_mean_pearson_r": float(linear_row.mean_pearson_r),   # the question this step exists to answer:
            "linear_rsa": float(linear_row.rsa_ridge_predicted),         # does nonlinearity buy anything?
        })
        print(rows[-1])
    shared.save_table(config.RESULTS_RIDGE_MAPPING / "nonlinear_recovery.csv", rows)


if __name__ == "__main__":
    main()
