"""
Run the whole supplement, section by section, in the order the report presents it.

    python -m run.run_all                # every section, reusing cached features
    python -m run.run_all --features     # extract the features first (GPU, slow)
    python -m run.run_all --extra        # also run the ported work with no section

Sections, in report order:

    § Setup                 features   images -> one frozen feature matrix per model
                            hspose     human triplets -> the yardstick + three checks
    § Reference points      anchors    chance / free human embedding / ceiling
    § Raw similarity        rsa        each model's untouched geometry vs hSPOSE
    § Ridge mapping         ridge      cross-validated linear map model -> hSPOSE
    § Odd-one-out           scoring    the real task, raw vs mapped, held-out triplets

`--extra` appends the two analyses ported from things-sim that the supplement has
no section for — the nonlinear map and the behavioural training branches — and
then re-runs § Raw similarity, whose before/after table gains a column from each.
"""

from __future__ import annotations

import sys
import time

from run import (run_behavioural_training, run_features, run_hspose, run_nonlinear_map, run_odd_one_out,
                 run_raw_similarity, run_reference_points, run_ridge_mapping)


def main() -> None:
    """Call each section's runner in turn, reporting wall time for each."""
    sections = [run_hspose, run_reference_points, run_raw_similarity, run_ridge_mapping, run_odd_one_out]
    if "--features" in sys.argv:                                  # only when the features need rebuilding
        sections.insert(0, run_features)                          # everything else reads its output
    if "--extra" in sys.argv:                                     # the ported work outside the supplement's sections
        sections += [run_nonlinear_map, run_behavioural_training, run_raw_similarity]  # last pass completes the RSA table
    for section in sections:
        start = time.time()
        print(f"=== {section.__name__} ===")
        section.main()
        print(f"=== {section.__name__} done in {time.time() - start:.1f}s ===")


if __name__ == "__main__":
    main()
