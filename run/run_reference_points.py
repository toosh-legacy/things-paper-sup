"""
§ Reference points — the three anchors every percentage is read against.

    python -m run.run_reference_points

Chance is arithmetic, the free human embedding is what run_hspose.py just scored,
and the ceiling is Hebart et al.'s human-human agreement — re-measured from the
test split when that split contains repeated triples.
"""

from __future__ import annotations

import pandas as pd

from common import config, shared
from reference_points.anchors import measured_ceiling


def main() -> None:
    """Write the reference-points table."""
    mask = shared.valid_mask()                                    # the ceiling must describe the scored object set
    test = shared.drop_invalid_triplets(shared.load_triplets("test"), mask)
    measured = measured_ceiling(test)                             # None when no triple was answered twice
    print("measured human-human ceiling:", "not estimable from this split" if measured is None else f"{measured:.4f}")

    free = float(pd.read_csv(config.RESULTS_SETUP / "hspose.csv")["held_out_accuracy"].iloc[0])  # the number § Setup just measured

    shared.save_table(config.RESULTS_REFERENCE_POINTS / "reference_points.csv", [
        {"anchor": "chance", "percent_correct": round(100 * config.CHANCE, 2),
         "source": "random three-way guess"},
        {"anchor": "free human embedding", "percent_correct": round(100 * free, 2),
         "source": "our hSPOSE refit - what a purpose-built human model achieves on this data"},
        {"anchor": "human-human ceiling", "percent_correct": round(100 * config.NOISE_CEILING, 2),
         "source": "Hebart et al.; nothing can beat this"},
        {"anchor": "human-human ceiling (measured here)",
         "percent_correct": None if measured is None else round(100 * measured, 2),
         "source": "agreement between repeated triplets in our test split"},
    ])


if __name__ == "__main__":
    main()
