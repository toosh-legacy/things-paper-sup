"""
Assemble the supplement's tables from the stages' summary CSVs.

The point is that no number in report/ is ever typed by hand. Each function here
emits one markdown table with the exact column headers and row order the
supplement uses, so a rerun of the pipeline updates the write-up mechanically.

Row order for every per-model table comes from config.MODEL_ORDER.
"""

from __future__ import annotations


def reproduction_table() -> str:
    """Ours vs. Hebart et al.: held-out accuracy, dimensions kept, RSM r."""
    raise NotImplementedError


def reference_points_table() -> str:
    """Chance, free human embedding, human-human ceiling."""
    raise NotImplementedError


def raw_similarity_table() -> str:
    """One Spearman r per model (stage 2)."""
    raise NotImplementedError


def ridge_recovery_table() -> str:
    """Mean Pearson r per model (stage 3)."""
    raise NotImplementedError


def oddoneout_table() -> str:
    """Raw %, ridge-mapped %, gain (stage 4)."""
    raise NotImplementedError


def run(out_path=None) -> None:
    """Write every table to report/tables.md (or out_path) in supplement order."""
    raise NotImplementedError
