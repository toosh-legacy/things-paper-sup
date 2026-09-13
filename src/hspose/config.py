"""
Every path and every constant in the project, in one place.

Nothing else in the package hardcodes a path or a magic number — if a number
appears in the supplement, it should be traceable to a name defined here.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# ROOT is the repo checkout; everything else hangs off it so the pipeline is
# runnable from any working directory.

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"             # inputs: features, triplets, image index, published SPoSE
FEATURES = DATA / "features"     # one <model>.npy per model, written by stage 0
IMAGES = DATA / "images"         # THINGS photographs, one folder or file per concept
RESULTS = ROOT / "results"       # everything the pipeline produces
REPORT = ROOT / "report"         # markdown tables assembled by tables.py

# Per-stage output directories (created on demand by io.ensure_dirs).
RESULTS_SPOSE = RESULTS / "spose"
RESULTS_RSA = RESULTS / "rsa"
RESULTS_RIDGE = RESULTS / "ridge"
RESULTS_ODDONEOUT = RESULTS / "oddoneout"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Keys are the short names used in filenames, CLI arguments and table rows.
# Each entry carries what stage 0 needs to instantiate the frozen backbone and
# what every later stage needs to label it.
#
#   loader      which code path in features.py builds it (timm / open_clip / torchvision / local)
#   weights     checkpoint id or path
#   dim         expected feature width, asserted after extraction
#   label       how the model is named in the supplement's tables
#
# EVA02 and IMIC-B share an architecture and a starting point on purpose: any
# difference between those two rows is attributable to the person-identity
# training and not to architecture or scale.

MODELS = {
    "eva02":  {"loader": "timm",        "weights": "",  "dim": None, "label": "EVA02"},
    "imic_b": {"loader": "local",       "weights": "",  "dim": None, "label": "IMIC - B"},
    "vgg16":  {"loader": "torchvision", "weights": "",  "dim": None, "label": "VGG-16"},
    "dinov3": {"loader": "timm",        "weights": "",  "dim": None, "label": "DINOv3"},
    "clip":   {"loader": "open_clip",   "weights": "",  "dim": None, "label": "CLIP"},
}

# Row order used in every table in the supplement.
MODEL_ORDER = ["eva02", "imic_b", "vgg16", "dinov3", "clip"]

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

N_OBJECTS = 1854          # THINGS concepts; row i is the same object in every array
N_VALID_EXPECTED = None   # objects left after dropping the image-less ones — assert once known

# --------------------------------------------------------------------------
# hSPOSE fit (Hebart et al. 2020 recipe — do not tune these)
# --------------------------------------------------------------------------

INIT_DIM = 90        # deliberately generous; pruning decides the final count
LAMBDA_L1 = 0.008    # the paper's cross-validated sparsity weight
PRUNE_THR = 0.1      # keep a dimension if any object's weight exceeds this
BATCH_SIZE = 1024
LR = 1e-3
MAX_EPOCHS = 60
PATIENCE = 6         # early stopping on held-out NLL

# --------------------------------------------------------------------------
# Ridge mapping
# --------------------------------------------------------------------------

N_FOLDS = 5                       # object folds — the same split for every model
ALPHA_GRID = (1e-2, 1e5, 20)      # (low, high, n) for np.logspace

# --------------------------------------------------------------------------
# Anchors — the scale every percentage is read against
# --------------------------------------------------------------------------

CHANCE = 1 / 3                    # random three-way guess
NOISE_CEILING = None              # human-human agreement; computed by anchors.py
PAPER_ACCURACY = 0.6460           # Hebart et al.'s own held-out number, for comparison
PAPER_N_DIMS = 49                 # dimensions their published embedding kept

SEED = 0                          # torch.manual_seed and KFold(random_state=)
