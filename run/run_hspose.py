"""
§ Setup -> Building the yardstick — refit hSPOSE and check it three ways.

    python -m run.run_hspose

Needs the feature files: the valid-object mask is built from them, so the human
yardstick describes exactly the same concepts the models do.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from common import config, shared
from setup.hspose import fit


def main() -> None:
    """Fit, save, and write the reproduction summary the supplement reports."""
    mask = shared.valid_mask()                                    # objects every model produced a vector for
    train = shared.drop_invalid_triplets(shared.load_triplets("train"), mask)  # a triplet naming a dropped object is unscorable
    test = shared.drop_invalid_triplets(shared.load_triplets("test"), mask)    # held out of this fit and of every later one
    print(f"objects kept: {int(mask.sum())} / {config.N_OBJECTS}   "
          f"triplets: {len(train):,} train / {len(test):,} test")

    hspose, accuracy, history = fit(train, test)                  # the refit human embedding, pruned
    shared.save_array(config.HSPOSE, hspose)                      # the yardstick every later section reads
    np.savez(config.RESULTS_SETUP / "hspose_history.npz", **history)  # per-epoch curves, for the appendix

    nonneg = np.clip(hspose, 0, None)                             # hSPOSE's own similarity is the dot product
    shared.save_array(config.RESULTS_SETUP / "hspose_similarity.npy", nonneg @ nonneg.T)

    published = np.loadtxt(config.PUBLISHED_SPOSE)                # Hebart et al.'s released 49-d embedding, used once
    published_nonneg = np.clip(published, 0, None)
    r, p = pearsonr(                                              # third check: same structure, whatever the dimension order
        shared.upper_triangle((nonneg @ nonneg.T)[np.ix_(mask, mask)]),       # our similarity over the valid objects...
        shared.upper_triangle((published_nonneg @ published_nonneg.T)[np.ix_(mask, mask)]))  # ...against theirs
    print(f"RSM correlation with published SPoSE-49d: r = {r:.3f} (p = {p:.1e})")

    shared.save_table(config.RESULTS_SETUP / "hspose.csv", [{
        "objects_kept": int(mask.sum()),                          # the setup section's first table
        "n_train_triplets": len(train),
        "n_test_triplets": len(test),
        "held_out_accuracy": round(accuracy, 4),                  # check 1 — things-sim: 0.6412
        "n_dims_kept": int(hspose.shape[1]),                      # check 2 — things-sim: 52
        "rsm_r_vs_published_49d": round(float(r), 4),             # check 3 — things-sim: 0.963
        "paper_accuracy": config.PAPER_ACCURACY,                  # their figures, for the "Hebart et al." column
        "paper_n_dims": config.PAPER_N_DIMS,
        "noise_ceiling": config.NOISE_CEILING,
    }])


if __name__ == "__main__":
    main()
