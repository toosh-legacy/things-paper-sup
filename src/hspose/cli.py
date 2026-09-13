"""
Command-line entry point: `python -m hspose <stage> [--models ...]`.

Stages run in order and each reads what the previous one wrote:

    features    0  extract frozen features for the five models     (GPU, slow)
    spose       1  refit hSPOSE from human triplets                (GPU, minutes)
    anchors        chance / free embedding / human-human ceiling
    rsa         2  raw model geometry vs hSPOSE                    (fast)
    ridge       3  5-fold out-of-fold map into hSPOSE              (fast)
    oddoneout   4  raw vs mapped on held-out triplets              (fast)
    tables         rebuild the supplement's markdown tables
    all            every stage above, in this order

--models restricts the per-model stages to a subset, for iterating on one row
without re-running the other four.
"""

from __future__ import annotations


def build_parser():
    """Argument parser: positional stage, optional --models, --seed, --device."""
    raise NotImplementedError


def main(argv=None) -> int:
    """Dispatch to the chosen stage's run() and return an exit code."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
