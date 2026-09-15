"""
extra — train each model toward human judgements (branches A, B and C).

    python -m run.run_behavioural

The longest step after feature extraction: per model it fits a free embedding on
the model's own choices (A), then five object-level folds of a linear read-out
(B) and of an MLP read-out (C), each preceded by a fold-0 hyperparameter search,
plus the random-feature control that makes the numbers interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import config, shared
from setup.hspose import prune
from extra.behavioural_training import branch_a, run_object_cv, select_on_fold0, standardise


def main() -> None:
    """Run all three branches for every model and save the summary table."""
    mask = shared.valid_mask()                                    # the same objects as every other step
    valid_idx = np.where(mask)[0]                                 # positions, because the folds index into this array
    human_train = shared.drop_invalid_triplets(shared.load_triplets("train"), mask)
    human_test = shared.drop_invalid_triplets(shared.load_triplets("test"), mask)
    odd_one_out = shared.odd_one_out_percent                      # one decision rule for the whole project

    rows = []
    for model in config.MODEL_ORDER:
        print(f"=== {model} ===")
        features = shared.load_features(model).astype(np.float32)
        features_t = standardise(features)                        # z-scored, on the device, zeros for image-less rows

        # ---- branch A: the paper's procedure, on the model's own choices ----
        embedding_a, acc_own, acc_human = branch_a(features, human_train, human_test)
        shared.save_array(config.RESULTS_EXTRA / f"{model}_ownchoice_embedding.npy", prune(embedding_a))
        print(f"branch A: {acc_own*100:.2f} % on its own choices, {acc_human*100:.2f} % agreement with humans")

        # ---- branch B: linear read-out, object-level CV, lambda chosen on fold 0 ----
        best_b, selection_b = select_on_fold0(features_t, human_train, human_test, valid_idx,
                                              [{"hidden": None, "lam": lam} for lam in config.BRANCH_B_LAMBDAS],
                                              config.BRANCH_B, "B")
        shared.save_table(config.RESULTS_EXTRA / f"{model}_branchB_selection.csv", selection_b)
        oof_b = run_object_cv(features_t, human_train, human_test, valid_idx,
                              None, best_b["lam"], config.BRANCH_B, "B/real")
        shared.save_array(config.RESULTS_EXTRA / f"{model}_human_trained_embedding.npy", oof_b)  # § Raw similarity reads this

        # The control: identical pipeline on random features. If it scores near the real
        # thing, the protocol is measuring capacity rather than the representation.
        generator = np.random.default_rng(config.SEED)
        random_features = generator.standard_normal((config.N_OBJECTS, features.shape[1]))
        random_features[~mask] = np.nan                           # the same objects are missing, so the split is identical
        oof_b_random = run_object_cv(standardise(random_features.astype(np.float32)), human_train, human_test,
                                     valid_idx, None, best_b["lam"], config.BRANCH_B, "B/random")
        shared.save_array(config.RESULTS_EXTRA / f"{model}_random_control.npy", oof_b_random)

        # ---- branch C: the same protocol with a hidden layer ----
        best_c, selection_c = select_on_fold0(
            features_t, human_train, human_test, valid_idx,
            [{"hidden": hidden, "lam": lam} for hidden in config.BRANCH_C_HIDDEN for lam in config.BRANCH_C_LAMBDAS],
            config.BRANCH_C, "C")
        shared.save_table(config.RESULTS_EXTRA / f"{model}_branchC_selection.csv", selection_c)
        oof_c = run_object_cv(features_t, human_train, human_test, valid_idx,
                              best_c["hidden"], best_c["lam"], config.BRANCH_C, "C/real")
        shared.save_array(config.RESULTS_EXTRA / f"{model}_human_trained_embedding_mlp.npy", oof_c)

        # ---- score every branch on the real held-out human triplets ----
        # The random control is scored with a tiebreak seed: its embedding can legitimately
        # collapse to all zeros, and argmax's silent "first index" rule would then score every
        # triplet correct, because the true odd one out is always stored in column 2.
        ridge = pd.read_csv(config.RESULTS_ODD_ONE_OUT / "odd_one_out.csv")
        ridge_row = ridge[ridge.model == config.MODELS[model]["label"]].iloc[0]
        row = {
            "model": config.MODELS[model]["label"],
            "raw_pct": float(ridge_row.raw_pct_correct),                       # the odd-one-out numbers, for one table
            "ridge_mapped_pct": float(ridge_row.ridge_mapped_pct_correct),
            "branchA_on_own_choices": round(acc_own * 100, 2),
            "branchA_agreement_with_humans": round(acc_human * 100, 2),
            "branchB_lambda": best_b["lam"],
            "branchB_pct": round(odd_one_out(oof_b, human_test), 2),           # the headline branch B number
            "branchB_pct_dot_rule": round(odd_one_out(oof_b, human_test, rule="dot"), 2),  # comparable with hSPOSE itself
            "branchB_random_control_pct": round(odd_one_out(oof_b_random, human_test, tiebreak_seed=config.SEED), 2),
            "branchC_hidden": best_c["hidden"], "branchC_lambda": best_c["lam"],
            "branchC_pct": round(odd_one_out(oof_c, human_test, tiebreak_seed=config.SEED), 2),
            "free_human_embedding_pct": round(100 * config.FREE_HUMAN_ACC, 2),  # the scale, again
            "noise_ceiling_pct": round(100 * config.NOISE_CEILING, 2),
            "chance_pct": round(100 * config.CHANCE, 2),
        }
        row["branchB_feature_contribution"] = round(row["branchB_pct"] - row["branchB_random_control_pct"], 2)
        row["branchC_vs_branchB"] = round(row["branchC_pct"] - row["branchB_pct"], 2)
        rows.append(row)
        print(row)

    shared.save_table(config.RESULTS_EXTRA / "behavioural_training.csv", rows)
    print("re-run `python -m run.run_rsa` to fill the human-trained column of the RSA table")


if __name__ == "__main__":
    main()
