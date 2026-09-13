"""
Run the whole pipeline end to end.

Thin wrapper over the CLI — kept because the stage order is part of the method
and belongs somewhere a reader can see it at a glance. Stage 0 is separate: the
features are expensive, rarely change, and are what everything else caches on.

    python scripts/run_all.py             # stages 1-4 + tables
    python scripts/run_all.py --features  # include feature extraction first
"""

# Order matters: spose -> anchors -> rsa -> ridge -> oddoneout -> tables.
STAGES = ["spose", "anchors", "rsa", "ridge", "oddoneout", "tables"]


def main():
    """Call each stage's run() in turn, logging wall time per stage."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
