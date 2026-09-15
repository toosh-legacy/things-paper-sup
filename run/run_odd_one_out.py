"""
§ Odd-one-out — raw vs. mapped, on held-out human triplets.

    python -m run.run_odd_one_out

Scores every representation of a model on exactly the same triplets — the raw
features, the ridge-mapped vector, and the nonlinear-mapped vector when
run_nonlinear_map.py has produced one.
"""

from __future__ import annotations

import numpy as np

from common import config, shared
from odd_one_out.scoring import scorable_triplets


def main() -> None:
    """Score raw and mapped representations for every model and save the table."""
    mask = shared.valid_mask()
    test = shared.drop_invalid_triplets(shared.load_triplets("test"), mask)  # held out of every fit in the project
    rows = []
    for model in config.MODEL_ORDER:
        raw = shared.load_features(model)                         # full [1854, d], NaN rows for image-less objects
        ridge = np.load(config.RESULTS_RIDGE_MAPPING / f"{model}_ridge_hspose_vector.npy")  # full [1854, k], zeros outside the fit
        nonlinear_path = config.RESULTS_RIDGE_MAPPING / f"{model}_nonlinear_hspose_vector.npy"
        nonlinear = np.load(nonlinear_path) if nonlinear_path.exists() else None    # optional, from run_nonlinear_map.py

        representations = [raw, ridge] + ([nonlinear] if nonlinear is not None else [])
        scored = scorable_triplets(test, *representations)        # one triplet set for every column, or no gain column
        print(f"[{model}] scored test triplets: {len(scored):,} / {len(test):,}")

        raw_pct = shared.odd_one_out_percent(raw, scored)         # the model's own untouched representation
        ridge_pct = shared.odd_one_out_percent(ridge, scored)     # the same objects after the linear map
        row = {
            "model": config.MODELS[model]["label"],
            "n_test_scored": len(scored),
            "raw_pct_correct": round(raw_pct, 2),
            "ridge_mapped_pct_correct": round(ridge_pct, 2),
            "gain": round(ridge_pct - raw_pct, 2),                # the behavioural counterpart of the ridge recovery scores
            "chance_pct": round(100 * config.CHANCE, 2),          # the scale these percentages are read against
            "noise_ceiling_pct": round(100 * config.NOISE_CEILING, 2),
        }
        if nonlinear is not None:
            row["nonlinear_mapped_pct_correct"] = round(shared.odd_one_out_percent(nonlinear, scored), 2)
        rows.append(row)
        print(f"[{model}] raw {raw_pct:.2f} %   ridge {ridge_pct:.2f} %   gain {ridge_pct - raw_pct:+.2f}")

        predictions = config.RESULTS_ODD_ONE_OUT / f"{model}_ridge_predictions"  # the mapped model's own answers, saved
        predictions.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):                           # both splits, so the answers can be inspected anywhere
            np.save(predictions / f"{split}_triplets.npy",        # np.save, not shared.save_array: these are object
                    shared.odd_one_out_predictions(ridge, shared.load_triplets(split)))  # indices and must stay integers
    shared.save_table(config.RESULTS_ODD_ONE_OUT / "odd_one_out.csv", rows)


if __name__ == "__main__":
    main()
