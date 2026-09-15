"""
§ Ridge mapping — the cross-validated linear map from each model into hSPOSE.

    python -m run.run_ridge_map

Writes, per model: the per-dimension recovery table, the out-of-fold predicted
hSPOSE (both at valid-object size and at full [1854, k] size, for the odd-one-out section), and a
row in the recovery summary.
"""

from __future__ import annotations

import numpy as np

from common import config, shared
from ridge_mapping.ridge import per_dimension_recovery, ridge_out_of_fold
from raw_similarity.rsa import rsa


def main() -> None:
    """Fit, score and save the mapped vectors for every model."""
    mask = shared.valid_mask()                                    # identical object set, therefore identical folds
    hspose = shared.load_hspose()[mask]                           # the targets: each object's human description
    rows = []
    for model in config.MODEL_ORDER:
        print(f"=== {model} ===")
        features = shared.load_features(model)[mask].astype(float)  # the inputs: this model's untouched representation
        predicted, alphas = ridge_out_of_fold(features, hspose)   # every row predicted by a map that never saw it

        per_dim = per_dimension_recovery(hspose, predicted)       # how well each human dimension survives the trip
        shared.save_table(config.RESULTS_RIDGE_MAPPING / f"{model}_per_dim_accuracy.csv", per_dim)
        print(f"mean r = {per_dim.pearson_r.mean():.4f}   mean R^2 = {per_dim.r2.mean():.4f}   "
              f"dims with R^2<=0: {(per_dim.r2 <= 0).sum()} / {hspose.shape[1]}")

        r_predicted = rsa(hspose, predicted)                      # does the map preserve overall structure too...
        r_raw = rsa(hspose, features)                             # ...and what did the raw geometry score? (§ Raw similarity's number)
        print(f"RSA true hSPOSE vs ridge-predicted: {r_predicted:.4f}   vs raw: {r_raw:.4f}")

        shared.save_array(config.RESULTS_RIDGE_MAPPING / f"{model}_predicted_hspose_oof.npy", predicted)  # valid-object rows
        full = np.zeros((config.N_OBJECTS, hspose.shape[1]), dtype=np.float32)  # back to full size...
        full[mask] = predicted                                    # ...so the odd-one-out section can index it with raw triplet indices
        shared.save_array(config.RESULTS_RIDGE_MAPPING / f"{model}_ridge_hspose_vector.npy", full)

        rows.append({
            "model": config.MODELS[model]["label"],
            "mean_pearson_r": round(float(per_dim.pearson_r.mean()), 4),  # the recovery table's headline number
            "mean_r2": round(float(per_dim.r2.mean()), 4),
            "dims_r2_le_0": int((per_dim.r2 <= 0).sum()),
            "rsa_ridge_predicted": round(r_predicted, 4),
            "rsa_raw": round(r_raw, 4),
            "alpha_min": min(alphas), "alpha_max": max(alphas),   # pinned at an end of the grid = the grid is too narrow
        })
    shared.save_table(config.RESULTS_RIDGE_MAPPING / "ridge_recovery.csv", rows)


if __name__ == "__main__":
    main()
