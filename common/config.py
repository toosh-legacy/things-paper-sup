"""
Every path and every constant, in one place.

All values are the ones the `things-sim` hspose_alignment notebooks hardcoded;
nothing is re-tuned here. Where a constant came from the paper rather than from
a run, the comment says so.

One folder per section of report/things_part3_supplement.md, in the order the
supplement presents them: setup -> reference points -> raw similarity -> ridge
mapping -> odd-one-out. Anything ported from things-sim that the supplement has
no section for lives in extra/.
"""

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# ROOT is the repo checkout (this file is <root>/common/config.py), so the
# pipeline runs from any working directory.

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"                                        # inputs; nothing here is committed
FEATURES = DATA / "features"                                # one <model>.npy per model, written by the setup section
IMAGES = DATA / "object_images"                             # THINGS photographs, one folder per concept
IMAGE_INDEX = DATA / "image_index.json"                     # object name -> row index; defines the row order
TRIPLETS = {                                                # human odd-one-out choices, [i, j, k] with k the odd one out
    "train": DATA / "human_train_triplets.npy",
    "test": DATA / "human_test_triplets.npy",               # held out of every fit in the project
}
PUBLISHED_SPOSE = DATA / "spose_embedding_49d_sorted.txt"   # Hebart et al.'s released 49-d embedding

RESULTS = ROOT / "results"                                  # one directory per section of the supplement
RESULTS_SETUP = RESULTS / "setup"                           # § Setup: the data, the models, and building a yardstick
RESULTS_REFERENCE_POINTS = RESULTS / "reference_points"     # § Reference points
RESULTS_RAW_SIMILARITY = RESULTS / "raw_similarity"         # § Computation and results -> Raw similarity
RESULTS_RIDGE_MAPPING = RESULTS / "ridge_mapping"           # § Computation and results -> Ridge mapping
RESULTS_ODD_ONE_OUT = RESULTS / "odd_one_out"               # § Computation and results -> Odd-one-out
RESULTS_EXTRA = RESULTS / "extra"                           # ported work the supplement has no section for

HSPOSE = RESULTS_SETUP / "hspose.npy"                       # the human yardstick every later step reads

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Keys are the short names used in filenames and table rows.
#
#   loader   which branch of setup/features.py builds it:
#            "timm"  -> timm.create_model(weights, pretrained=True, num_classes=0)
#            "imic"  -> the EchoBID re-identification checkpoint, loaded from IMIC_REPO
#   weights  timm model id, or the checkpoint path for the imic loader
#   dim      feature width, asserted after extraction
#   label    how the model is named in the supplement's tables
#
# EVA02 and IMIC-B share an architecture and a starting point on purpose: any
# difference between those two rows is attributable to the person-identity
# training and not to architecture or scale.
#
# CONFIRM BEFORE RUNNING: `eva02` and `dinov3` are new rows in this supplement —
# things-sim ran ViT / IMIC-A / CLIP / VGG-16 — so their timm ids and the IMIC-B
# checkpoint path below are the only values in this file that were not taken
# from a notebook that had already been run.

IMIC_REPO = Path("/mnt/Data/Tushaar/otoolelab_nn_repo")     # repo that defines EchoBID and ResizeOntoBlackSquare
IMIC_CKPT = Path("/mnt/Data/briar/models/imic/imic-a_eva02_triplet.pth")  # things-sim used IMIC-A; swap for the IMIC-B checkpoint
IMIC_BACKBONE = "eva02_large_patch14_448"                   # the EchoBID backbone argument, as in things-sim

MODELS = {
    "eva02":  {"loader": "timm", "weights": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", "dim": 1024, "label": "EVA02"},
    "imic_b": {"loader": "imic", "weights": str(IMIC_CKPT),                                   "dim": 1568, "label": "IMIC - B"},
    "vgg16":  {"loader": "timm", "weights": "vgg16.tv_in1k",                                  "dim": 4096, "label": "VGG-16"},
    "dinov3": {"loader": "timm", "weights": "vit_large_patch16_dinov3.lvd1689m",              "dim": 1024, "label": "DINOv3"},
    "clip":   {"loader": "timm", "weights": "vit_large_patch14_clip_224.metaclip_2pt5b",      "dim": 1024, "label": "CLIP"},
    # things-sim's fifth model, kept so its published numbers can be reproduced here.
    # It is not one of the supplement's five rows, so it is absent from MODEL_ORDER.
    "vit":    {"loader": "timm", "weights": "vit_large_patch16_224.augreg_in21k_ft_in1k",     "dim": 1024, "label": "ViT"},
}

# Row order used in every table in the supplement.
MODEL_ORDER = ["eva02", "imic_b", "vgg16", "dinov3", "clip"]

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

N_OBJECTS = 1854          # THINGS concepts; row i is the same object in every array
BATCH_IMAGES = 64         # images per forward pass during extraction

# --------------------------------------------------------------------------
# hSPOSE fit (Hebart et al. 2020 recipe — do not tune these)
# --------------------------------------------------------------------------

INIT_DIM = 90        # deliberately generous; pruning decides the final count
LAMBDA_L1 = 0.008    # the paper's cross-validated sparsity weight
PRUNE_THR = 0.1      # keep a dimension if its largest object weight exceeds this
BATCH_SIZE = 1024
LR = 1e-3
MAX_EPOCHS = 60
PATIENCE = 6         # early stopping on held-out NLL

# --------------------------------------------------------------------------
# Ridge mapping
# --------------------------------------------------------------------------

N_FOLDS = 5                      # object folds — the same split for every model and for extra/
ALPHAS = np.logspace(-2, 5, 20)  # RidgeCV regularisation grid

# Nonlinear counterpart: the same folds and the same target, an MLP instead of ridge.
NL_HIDDEN_GRID = [128, 256, 512]
NL_WEIGHT_DECAY_GRID = [0.0, 1e-4, 1e-3]
NL_LR = 1e-3
NL_MAX_EPOCHS = 400
NL_PATIENCE = 25
NL_VAL_FRACTION = 0.15           # inner slice of each fold's training objects, for early stopping only

# --------------------------------------------------------------------------
# Behavioural training (extra/ — no section in the supplement)
# --------------------------------------------------------------------------

# Branch A — the paper's free embedding fit to the model's own choices: paper settings, untouched.
BRANCH_A = {"lam": LAMBDA_L1, "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE}

# Branch B — a linear read-out of frozen features, under object-level CV.
TARGET_EMB_STD = 0.19                      # W init std = TARGET/sqrt(d), so the softmax starts d-independent
BRANCH_B_LAMBDAS = [0.0, 0.001, 0.008]     # 0.008 = the paper's value, included so it can win
BRANCH_B = {"lr": 1e-3, "max_epochs": 40, "patience": 6}

# Branch C — the same protocol with a one-hidden-layer read-out.
BRANCH_C_HIDDEN = [128, 256, 512]
BRANCH_C_LAMBDAS = [0.0, 0.001]
BRANCH_C = {"lr": 1e-3, "max_epochs": 60, "patience": 8}

# --------------------------------------------------------------------------
# Reference points — the scale every percentage is read against
# --------------------------------------------------------------------------

CHANCE = 1 / 3            # random three-way guess
NOISE_CEILING = 0.6722    # human-human agreement, from Hebart et al.; nothing can beat this
PAPER_ACCURACY = 0.6460   # their own held-out number, for the reproduction table
PAPER_N_DIMS = 49         # dimensions their published embedding kept
FREE_HUMAN_ACC = 0.6412   # things-sim's hSPOSE refit scored this; recomputed by run_hspose.py

SEED = 0                  # torch.manual_seed and KFold(random_state=)
