"""
hspose — Part 3 supplement pipeline.

Five frozen vision models (EVA02, IMIC-B, CLIP, DINOv3, VGG-16) compared against
hSPOSE, a sparse non-negative embedding refit on real THINGS odd-one-out choices.

Stage order (each stage writes the inputs the next one reads):

    0  features    images   -> data/features/<model>.npy       [1854, d]
    1  spose       triplets -> results/spose/hspose.npy        [1854, k]
    2  rsa         raw model geometry vs hSPOSE geometry       -> Spearman r
    3  ridge       5-fold out-of-fold map model -> hSPOSE      -> mean Pearson r
    4  oddoneout   raw vs ridge-mapped on held-out triplets    -> % correct

Run a stage with `python -m hspose <stage>`; see cli.py.
"""

__all__ = [
    "config",
    "io",
    "features",
    "spose",
    "anchors",
    "rsa",
    "ridge",
    "oddoneout",
    "tables",
]
