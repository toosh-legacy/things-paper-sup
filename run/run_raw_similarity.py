"""
§ Raw similarity — each model's geometry against hSPOSE's.

    python -m run.run_rsa

The raw column needs only the setup section; the ridge column needs § Ridge mapping;
the human-trained column needs extra/behavioural_training, so it is filled in
only once that has run —
re-run this script after run_behavioural.py to complete the table.
"""

from __future__ import annotations

import numpy as np

from common import config, shared
from raw_similarity.rsa import rsa


def main() -> None:
    """Correlate every version of every model's geometry against hSPOSE's."""
    mask = shared.valid_mask()                                    # one object set for all five models
    hspose = shared.load_hspose()[mask]                           # the human side of every comparison
    rows = []
    for model in config.MODEL_ORDER:
        row = {"model": config.MODELS[model]["label"]}
        row["raw_r"] = round(rsa(hspose, shared.load_features(model)[mask].astype(float)), 4)  # untouched geometry

        ridge_path = config.RESULTS_RIDGE_MAPPING / f"{model}_predicted_hspose_oof.npy"
        if ridge_path.exists():                                   # the ridge mapping's out-of-fold predictions, valid-object rows
            row["ridge_r"] = round(rsa(hspose, np.load(ridge_path)), 4)

        trained_path = config.RESULTS_EXTRA / f"{model}_human_trained_embedding.npy"
        if trained_path.exists():                                 # behavioural training's branch B, full [1854, k]
            row["humantrained_r"] = round(rsa(hspose, np.load(trained_path)[mask]), 4)

        rows.append(row)
        print(row)
    # The supplement's table is the raw column alone; the mapped columns are kept
    # beside it because the comparison is what makes the raw number interpretable.
    shared.save_table(config.RESULTS_RAW_SIMILARITY / "raw_similarity.csv",
                      [{"model": row["model"], "spearman_r": row["raw_r"]} for row in rows])
    shared.save_table(config.RESULTS_RAW_SIMILARITY / "rsa_before_after.csv", rows)


if __name__ == "__main__":
    main()
