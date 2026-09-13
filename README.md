# things-paper-sup

Code for the Part 3 supplement: five frozen vision models compared against a
model of human object similarity on THINGS.

All five models are used **frozen** — each is shown photographs of the 1,854
THINGS concepts, the representation it produces for each object is recorded, and
the model itself is never changed.

| model | what it was trained to do |
|---|---|
| **EVA02** | ordinary large-scale image training — the untouched starting point |
| **IMIC-B** | the same architecture after person re-identification training; the model this project is about |
| **CLIP** | matching photographs to captions — object knowledge arrived through language |
| **DINOv3** | images alone, no labels and no text |
| **VGG-16** | object classification; the model analysed by Mahner et al. (2025) |

EVA02 and IMIC-B share an architecture and a starting point, so any difference
between them is attributable to the person-identity training rather than to
architecture or scale.

## Layout

```
src/hspose/
├── config.py      every path and constant — nothing else hardcodes either
├── io.py          loading, the shared valid-object mask, saving
├── features.py    stage 0 — frozen feature extraction
├── spose.py       stage 1 — refit hSPOSE from human triplets
├── anchors.py     chance / free human embedding / human-human ceiling
├── rsa.py         stage 2 — raw model geometry vs hSPOSE geometry
├── ridge.py       stage 3 — cross-validated linear map into hSPOSE
├── oddoneout.py   stage 4 — raw vs mapped on held-out human triplets
├── tables.py      assembles the supplement's markdown tables from the CSVs
└── cli.py         python -m hspose <stage>
scripts/run_all.py the whole pipeline, in order
tests/             invariants worth asserting before trusting a table
data/              inputs (gitignored) — see data/README.md
results/           stage outputs (gitignored)
report/            the supplement plus the generated tables
```

Every module is currently a comment-only skeleton: docstrings state what each
function must do and why, and the bodies are yours to write.

## Running

```bash
pip install -e .
python -m hspose features     # stage 0, GPU, slow — cache it and forget it
python -m hspose spose        # stage 1, GPU, minutes
python -m hspose rsa          # stage 2
python -m hspose ridge        # stage 3
python -m hspose oddoneout    # stage 4
python -m hspose tables       # rebuild report/tables.md
# or: python scripts/run_all.py
```

`--models eva02 imic_b` restricts the per-model stages to a subset. The seed is
fixed in `config.SEED`; the ridge folds depend on it and nothing else, so every
model is scored on the same object split.

## Two rules the whole pipeline rests on

1. **Row `i` is the same object everywhere** — in every feature file, in hSPOSE,
   in the published embedding, and inside every triplet index. Nothing reorders
   rows.
2. **One shared valid-object mask.** Concepts with no usable image are dropped
   once, along with every triplet mentioning them, identically for all five
   models — so no model is scored on an easier set than another.

## References

Hebart, M. N., Zheng, C. Y., Pereira, F., & Baker, C. I. (2020). Revealing the
multidimensional mental representations of natural objects underlying human
similarity judgements. *Nature Human Behaviour*.

Mahner, F. P., et al. (2025). Dimensions underlying the representational
alignment of deep neural networks with humans. *Nature Machine Intelligence*.
