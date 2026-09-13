# Things Part 3 supplement paper

> Tables are left blank; they are filled in from the model runs.

## Setup: the data, the models, and building a yardstick

### THINGS

1,854 everyday object concepts. Some have no usable image and are all-NaN in every model's feature
file, so they are dropped everywhere, along with any triplet that mentions one of them. The dropped
set is identical across models, so none is scored on an easier set than another.

|  | count |
|---|---|
| concepts with no usable image | |
| objects kept | |
| human triplets, train / test | |

### The human judgements

Odd-one-out choices collected for THINGS: a participant was shown three object images and asked
which one did not belong. There is no right answer; the choice reveals how that person organises
objects. Each response says the two objects the participant did *not* choose were judged more
similar to each other than either was to the third, and millions of these statements together
describe the shape of human object space.

### The five models

All are used **frozen**: each is shown photographs of the kept concepts, we record the
representation it produces for each object, and the model itself is never changed. They differ in
what they were originally trained to do, and the set was chosen so the comparisons between them mean
something.

| Model | Dim | What it is |
|---|---|---|
| EVA02 | | large vision transformer, ordinary large-scale image training — the untouched starting point |
| IMIC-B | | same architecture after triplet-loss training for person re-identification — the model of interest |
| CLIP | | CLIP image encoder; language-supervised |
| DINOv3 | | self-supervised on images alone; no labels and no text |
| VGG-16 | | penultimate features; the model Mahner et al. (2025) used |

IMIC-B was never given object categories or labels, and shares its architecture and starting point
with EVA02 — so any difference between those two rows is attributable to the person-identity
training rather than to architecture or scale. CLIP and DINOv3 together ask whether human-like
object structure requires linguistic supervision or arises without it.

### Building the yardstick: reproduction of Hebart et al.

Everything here is measured against a model of human judgement rather than the raw responses, so we
first rebuilt that model and checked it was sound. We re-ran SPoSE (Hebart et al., 2020) on the human
odd-one-out responses following the published recipe exactly, including the same sparsity setting;
the procedure starts from a deliberately generous number of dimensions and prunes any that carry no
weight, so the final count is discovered rather than chosen. We call the result **hSPOSE**, and it is
the human yardstick for the rest of the paper. We re-fitted rather than downloading the published
embedding so the human reference comes from the same objects and the same data split as everything
it is compared against. It was then checked against the original three ways: held-out triplet
accuracy, the number of dimensions retained, and whether it describes the same similarity structure
as their published embedding.

|  | ours | Hebart et al. |
|---|---|---|
| held-out triplet accuracy | | |
| dimensions kept after pruning | | |
| similarity-matrix correlation with their published 49-d embedding | | — |

## Reference points

Percentages on this task cannot be read against 100 %, because the task has no correct answer. Three
anchors set the scale.

|  | % correct | |
|---|---|---|
| chance | | random three-way guess |
| free human embedding | | our hSPOSE re-fit — what a purpose-built human model achieves on this data |
| human–human ceiling | | how often two people agree. Nothing can beat this |

The ceiling matters most. Shown *dolphin, shark, helicopter*, almost everyone picks the helicopter;
shown *knife, fork, scissors*, people disagree. When we score a model against "the human's answer",
we score it against one participant's judgement on a question other participants would have answered
differently. The gap between the ceiling and 100 % is genuine disagreement between people, not model
error, so the meaningful range for every percentage below runs from chance up to the ceiling and is
roughly a third as wide as the nominal scale.

## Computation and results

### Raw similarity

The first question is whether a model's representation already resembles the human one before we do
anything to it. We cannot compare the two directly — they have different numbers of dimensions and
those dimensions mean different things — so we compared *patterns of similarity* instead. For each
model we worked out how similar it considers every possible pair of objects, did the same for
hSPOSE, and then asked how well the two orderings of pairs agree: does this model also treat
dolphin–shark as a closer pair than dolphin–helicopter? We used a rank-based correlation, because
only the ordering of pairs is comparable between two spaces on unrelated scales. Nothing is trained
or fitted at this stage.

| Model | Spearman r |
|---|---|
| EVA02 | |
| IMIC - B | |
| VGG-16 | |
| DINOv3 | |
| CLIP | |

A low score here does not by itself mean a model lacks human-relevant information. Comparing
similarity in a model's own coordinates treats all of its dimensions as equally important, so
structure that is present but spread across unhelpful directions will not show up. The next section
is what tells the two cases apart.

### Ridge mapping — how much can we recover from each model?

Here we allowed each model one piece of help: a single fitted mapping from its representation into
the human dimensions. A mapping of this kind can rotate the space, stretch some directions relative
to others, and combine existing dimensions together — but it cannot add information that was not
there to begin with. That is exactly why it is the right tool. If the mapping recovers a lot of
human structure, the structure must have been in the model's representation all along, just
oriented in a way the previous comparison could not read.

The mapping was fitted with ridge regression, which keeps a fit from latching onto noise when there
are far more input dimensions than objects, and its strength was chosen automatically rather than by
hand. The critical safeguard is how it was tested. We split the objects into five groups and
predicted each group using a mapping fitted only on the other four, so **no object's predicted human
description was ever informed by its own true one**. Without this, a mapping with thousands of free
parameters could score well simply by memorising, and the numbers would mean nothing. The same five
groups were used for every model, so the comparison across models is exact. We then asked how well
each human dimension was recovered, on average, from each model's representation.

| Model | Mean Pearson r |
|---|---|
| EVA02 | |
| IMIC - B | |
| VGG-16 | |
| DINOv3 | |
| CLIP | |

### Odd-one-out: raw representations vs. the mapped ones

The measures above are about overall structure. The last step asks the stricter question of whether
the model reproduces individual human decisions. For each held-out triplet we gave the model the same
three objects a participant saw, worked out which two it considers most similar, and took the
remaining object as its answer, scoring it against what the participant actually chose. We did this
twice for each model: once using the model's own untouched representation, and once using the mapped
version from the previous section. The triplets used for scoring were held out throughout — they were
not used to build hSPOSE, and the mapped predictions were produced by mappings that had never seen
those objects' human descriptions — so neither column can be inflated by having seen the answers.

| Model | raw % | Ridge mapped | gain |
|---|---|---|---|
| EVA02 | | | |
| IMIC - B | | | |
| VGG-16 | | | |
| DINOv3 | | | |
| CLIP | | | |

These percentages should be read against chance below and the human–human ceiling above. The gain
column is the behavioural counterpart of the recovery scores in the previous table: where a model
improves on both, it is not merely producing a similarity structure that correlates better in
aggregate, it is answering more of the actual questions the way a person answered them. The EVA02
and IMIC-B rows carry the central comparison, in both columns. If the two differ before mapping but
converge after it, training on person identity reorganised the model's object structure without
destroying it. If the difference survives the mapping, that training genuinely cost the model
information human object judgements depend on.

## References

Hebart, M. N., Zheng, C. Y., Pereira, F., & Baker, C. I. (2020). Revealing the multidimensional
mental representations of natural objects underlying human similarity judgements. *Nature Human
Behaviour*.

Mahner, F. P., et al. (2025). Dimensions underlying the representational alignment of deep neural
networks with humans. *Nature Machine Intelligence*.
