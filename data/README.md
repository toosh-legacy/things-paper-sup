# data/

Nothing in here is committed. Everything is either downloaded or produced by
stage 0, and every file shares the same row order: **row `i` is the same THINGS
object in all of them**.

| file | shape | what it is |
|---|---|---|
| `features/<model>.npy` | 1854 × d | one frozen feature vector per object; all-NaN row = no usable image |
| `human_train_triplets.npy` | n × 3 | human odd-one-out choices, `[i, j, k]` with `k` = the odd one out |
| `human_test_triplets.npy` | n × 3 | held-out choices — never used to fit hSPOSE or any ridge map |
| `image_index.json` | — | object name → row index; defines the row order above |
| `spose_embedding_49d_sorted.txt` | 1854 × 49 | Hebart et al.'s published embedding, used once as a sanity check |
| `images/` | — | THINGS photographs, input to stage 0 only |
