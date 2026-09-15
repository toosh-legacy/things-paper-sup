"""
§ Setup -> The five models — extract the frozen features.

    python -m run.run_features

The slow step of the pipeline (GPU, minutes to hours per model). Everything else
caches on its output, so it is run once and then left alone.
"""

from __future__ import annotations

import numpy as np

from common import config, shared
from setup.features import extract


def main() -> None:
    """Extract and save one feature matrix per model, plus a summary row each."""
    rows = []
    for model in config.MODEL_ORDER:                              # the supplement's table order
        features = extract(model)                                 # [N_OBJECTS, d], NaN rows where there is no image
        shared.save_array(config.FEATURES / f"{model}.npy", features)  # the only thing later steps read
        rows.append({
            "model": config.MODELS[model]["label"],
            "dim": features.shape[1],                             # fills the "Dim" column of the five-models table
            "concepts_without_image": int(np.isnan(features).any(axis=1).sum()),  # things-sim saw 31 of 1854
        })
    shared.save_table(config.RESULTS_SETUP / "features.csv", rows)


if __name__ == "__main__":
    main()
