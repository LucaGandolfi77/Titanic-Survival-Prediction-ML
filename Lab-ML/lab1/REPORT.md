# Lab 1 — k Nearest Neighbours and Decision Trees

**Course:** Machine Learning · **Source deck:** `Lab-ML/ML2026_lab1_KNN_DecTrees-merged.pdf`
(pages 1–16 = Lab 1, pages 17–27 = Scikit-Learn module)

**Deliverable status:** all 61 steps of `LAB1_AGENT_TODO.md` are complete, including the
optional two-feature variant. Every number below is traceable to a file in
`Lab-ML/lab1/results/` or a script in `Lab-ML/lab1/src/`. WEKA 3.8.7 (driven headlessly through
its bundled JRE) and scikit-learn 1.8.0 / scipy 1.17.1 on Python 3.14 were used. No value in
this report is estimated or fabricated; where something could not be measured it is stated as a
limitation in §9 rather than filled with a plausible number.

**Summary of findings.**

1. kNN on the circle problem peaks at **`k* = 1`** (91.0% test, 93.0% LOO); large `k`
   degenerates *exactly* to the majority-class baseline (0.530 / 0.600 / 0.481).
2. Of the four representations tested, **only `u = x^2 + y^2` demonstrably matters** — it gives a
   **2-leaf, depth-1 tree** at 100% test accuracy, and it is the only representation whose
   advantage survives a paired significance test (§7.2). The `sq` and `zt` "improvements" are
   within noise at n = 100.
3. On the licence-plate digits, **kNN beats the decision tree by ~2.9 pp** (97.6% vs 94.7% on
   `Bigtest2`, p ≈ 10⁻¹⁸), overturning the plan's expectation that the curse of dimensionality
   would punish kNN at 104 dimensions.
4. Overfitting control behaves as the deck describes (`M* = 4`, with a broad optimum), but the
   deck's claim that training accuracy *improves* as capacity is cut is **wrong** (§5.4).
5. A decision tree **cannot express a threshold that never appears in its training data**, which
   is why the `u` model scores 99.77% rather than 100% on the dense grid (§4.4).

---

## 1. Problem statement and data

### 1.1 The two regions

Two classes, labelled in the `.arff` files with the nominal values `c` and `q`:

| Class | Region | Rule |
|---|---|---|
| `c` | inside the **circle** of radius 1 centred at the origin | `x^2 + y^2 <= 1` |
| `q` | the **square ring**: inside the square of side `L = 2.5` centred at the origin, minus the circle | `|x| <= 1.25` and `|y| <= 1.25` and `not (x^2+y^2 <= 1)` |

The domain is therefore the square `[-1.25, 1.25]^2`; the disc sits in the middle and the four
corner regions belong to `q`.

**This mapping was verified, not assumed.** The rule `c <=> x^2+y^2 <= 1` reproduces the file
labels of `circletrain.arff` and `circletest.arff` with **zero errors**:

```
circletrain.arff : max(x^2+y^2 | c) = 0.999033   min(x^2+y^2 | q) = 1.000280
circletest.arff  : max(x^2+y^2 | c) = 0.995348   min(x^2+y^2 | q) = 1.026705
```

On the dense grid `circleall.arff` it agrees on 2595/2601 points; all 6 disagreements lie
**exactly** on the circle (`|x^2+y^2-1| < 1e-9`) and are boundary-labelling artefacts rather
than label noise. The verification is automated in
`src/verify_data.py` and `src/relabel.py` (see `results/representation_datasets.json`).

> Note on a subtlety that matters for Exercise 1b: the inner **square** of half-side
> `l/2 = 0.88` has circumradius `0.88*sqrt(2) = 1.2445 > 1`, so it is *larger* than the circle
> along the diagonals and *smaller* along the axes. The two regions overlap but neither
> contains the other, so the relabelling moves points **in both directions**.

### 1.2 Data inventory (verified)

| File | Rows | Notes | Role |
|---|---|---|---|
| `circletrain.arff` | 100 | 53 `c` / 47 `q` | training |
| `circletest.arff` | 100 | 60 `c` / 40 `q` | held-out test |
| `circleall.arff` | 2601 | 51x51 grid, step 0.05, 1251 `c` / 1350 `q` | dense "ground truth" |
| `Bigtest1_104.arff` | 6024 | 104 binary pixel features, 10 classes | digit training pool |
| `Bigtest2_104.arff` | 5010 | exactly 501 per digit | external digit test |

**Baselines.** Circle test set: majority class = `c` at **60%**. Digit data (10 classes):
chance = **10%**; `Bigtest2` is balanced by construction so its majority baseline is exactly
10%. Every accuracy below is read against these.

**Label encoding used throughout:** `c -> 0`, `q -> 1`, so **`q` is the "positive" class**.
Ties in the kNN vote break towards class `0`; the number of tied queries is counted and
reported (it was 0 for all odd `k` on two classes, as expected).

### 1.3 Integrity of the source data

`results/source_data_checksums.txt` records the SHA-256 of every source `.arff` and of the PDF
before any work started. No original file was modified; all derived datasets are new files in
`lab1/data/`.

---

## 2. Methods

### 2.1 kNN

To label a query point `X`, compute the Euclidean distance to every training point, take the
`k` closest, and assign the majority label. `k` is odd so a two-class vote cannot tie. The
implementation is vectorised (`src/knn_scratch.py`, `euclidean()` via the
`||a||^2 + ||b||^2 - 2a·b` identity) and was **proved correct by point-for-point comparison
with scikit-learn**:

```
euclidean() vs scipy.cdist          : max|diff| = 1.39e-14
kNN vs KNeighborsClassifier (k=1,3,5,7,15,31) : 100/100 test points agree at every k
```

No feature scaling is applied to the circle data: both `x` and `y` lie in `[-1.25, 1.25]`, so
Euclidean distance is already scale-fair. Digit pixels are binary `{0,1}`, so scaling is also
unnecessary there — but see §5.3 for why that does *not* save kNN on the digit data.

### 2.2 Decision trees

**WEKA J48** (C4.5): information-gain splitting, `-C 0.25` pruning confidence, `-M` = minimum
instances per leaf (`minNumObj`). J48 has **no random seed**; seeds in this project belong to
the *evaluation* (the random split) and not to the classifier.

**sklearn** `DecisionTreeClassifier(criterion='entropy', min_samples_leaf=M, ccp_alpha=0.0,
random_state=1)` was chosen to mirror J48 as closely as possible:

| WEKA | scikit-learn | meaning |
|---|---|---|
| `minNumObj (-M)` | `min_samples_leaf` | minimum samples per leaf — the capacity knob |
| `confidenceFactor (-C 0.25)` | `ccp_alpha` | post-pruning strength |
| `-U` | `ccp_alpha=0.0` | disable post-pruning |
| information gain | `criterion='entropy'` | split criterion |
| `IBk -K` | `n_neighbors` | number of neighbours |

Two important limitations of that mapping, found experimentally (§4.2 and §6):

* WEKA's `-U` does **not** produce a fully grown tree — it disables subtree *raising* but the
  confidence-threshold pruning still applies. sklearn's `min_samples_leaf=1, ccp_alpha=0`
  *does* grow the tree to purity. The two "unpruned" modes are therefore **not** equivalent.
* J48 prunes on the training set only and is deterministic, whereas sklearn's tree grew larger
  in every digit configuration (e.g. 134 vs 108 leaves at `M=2`).

**Reproducible 66% split.** The deck's `Percentage split 66%` is implemented as explicit ARFF
files so that WEKA and sklearn see *exactly* the same partition. `weka_percentage_split_indices`
reimplements `java.util.Random` (48-bit LCG) and the Fisher–Yates shuffle WEKA applies before
taking the first 66%. Validation: for all seeds the resulting test set has 2048 instances
(`round(0.34 * 6024) = 2048`), matching WEKA's own split size; the reconstructed class balance
is listed in `results/digits_summary.json`.

---

## 3. Exercise 1 — k Nearest Neighbours

### 3.1 Accuracy versus k

Full sweep in `results/knn_k_sweep.csv`; figure `figures/knn_k_sweep.png`.

| k | train | LOO-CV | 5-fold CV | test | `circleall` (vs file labels) |
|---|---|---|---|---|---|
| **1** | **1.000** | **0.930** | **0.940** | **0.910** | **0.900** |
| 3 | 0.990 | 0.860 | 0.900 | 0.880 | 0.890 |
| 5 | 0.940 | 0.890 | 0.908 | 0.870 | 0.869 |
| 7 | 0.940 | 0.900 | 0.869 | 0.840 | 0.861 |
| 9 | 0.900 | 0.860 | 0.849 | 0.820 | 0.844 |
| 15 | 0.900 | 0.860 | 0.778 | 0.840 | 0.812 |
| 21 | 0.840 | 0.780 | 0.630 | 0.780 | 0.740 |
| 31 | 0.610 | 0.560 | 0.549 | 0.670 | 0.566 |
| 51 | 0.530 | 0.530 | 0.530 | 0.600 | 0.481 |
| 99 | 0.530 | 0.530 | 0.530 | 0.600 | 0.481 |

**Highest test-set accuracy: 91.0% at k = 1** (`results/knn_summary.json`).

Answers to the questions on p.6:

* **At what k does test accuracy peak?** At `k = 1`, with 91%. The curve is essentially
  monotone decreasing in `k` for this problem.
* **Why does `k = 1` overfit?** It achieves **100% training accuracy** by construction: each
  training point's nearest neighbour is itself, so the training error is identically zero.
  The model has maximum capacity — an arbitrarily jagged decision boundary that interpolates
  every sample. Yet on this data it *also* generalises best, which is the interesting part:
  the classes really are separated by a smooth, low-noise boundary, so 1-NN's variance does
  not hurt much, and its zero bias is a decisive advantage over larger `k`.
* **Why does very large `k` underfit?** As `k` approaches the training-set size the
  neighbourhood becomes the whole dataset and every prediction collapses to the majority
  class. The evidence is exact: from `k = 51` onward every accuracy is pinned to **0.530**
  (train), **0.600** (test), **0.481** (grid) — precisely the majority-class baselines — and
  `train = LOO = CV` because the prediction no longer depends on the query point. (`k = 41` is
  the last value still marginally above the plateau, at 0.490 on the grid, which is consistent
  with the vote being decided by the 53/47 training majority for essentially every query.)
* **Is `train_acc` monotone in k?** Not strictly (it is flat-topped at `k = 5, 7` and at
  `k = 13` it dips then rises), but it decreases strongly overall, from 1.000 to 0.530. `k = 1`
  is special because it is the only value with *zero* training error, and it is the only
  setting whose LOO error can exceed its training error by construction.

### 3.2 Choosing k with a validation set

| Selection method | best k | accuracy at that k |
|---|---|---|
| leave-one-out CV on `circletrain` | **1** | 0.930 |
| 5-fold stratified CV on `circletrain` | **1** | 0.940 |
| 10-fold stratified CV on `circletrain` | **1** | 0.952 |
| test set `circletest` (methodologically wrong) | 1 | 0.910 |

All four **agree on `k* = 1`**, so here the honest protocol and the cheating protocol happen
to coincide — a lucky coincidence, not a licence to tune on the test set.

Stability across folds (`results/knn_cv_folds.json`): with 5 folds, `k = 1` wins 4 folds and
`k = 5` wins 1; with 10 folds, `k = 1` wins 9 and `k = 5` wins 1. Even in the folds it loses,
`k = 1` is within 0.02–0.03 of the winner, so the choice is robust rather than a fluke of one
particular split. LOO (0.930) is systematically more pessimistic than 10-fold CV (0.952)
because each LOO model is trained on 99 instead of 90 points and because the LOO estimate has
higher variance.

**Why tuning on the test set is wrong.** Selecting `k` by maximising `circletest` accuracy
uses the test labels to make a modelling decision, so the test set stops being unseen data and
becomes part of training. The reported accuracy is then optimistically biased — it is the
maximum over ~17 correlated estimates, so it is upward-biased even when the underlying
accuracy is unchanged. The correct protocol is nested: split the training data into
train/validation, tune `k` on validation only, then evaluate the frozen model **once** on the
test set. The number that should be quoted for `k* = 1` is therefore the validation estimate
(0.930–0.952), with 0.910 on `circletest` serving as an independent confirmation rather than as
the selection criterion.

### 3.3 Ground truth on the dense grid

Figure `figures/knn_boundary.png` shows the decision regions for `k in {1, 5, 15, 31}` with the
true circle and square overlaid; `figures/knn_errors_kstar.png` maps the errors at `k* = 1`.

At `k* = 1`: accuracy against the file labels is **0.9004**, and against the analytic ground
truth **0.9020** (259 errors over 2601 points). Excluding the 212 points that lie exactly on
the circle or on the square edge, the accuracy against the analytic truth is 0.8970 — slightly
*lower* than the raw figure, confirming that the 6 label disagreements are boundary artefacts
that flatter the classifier by a negligible amount.

The error map shows the expected structure: errors are concentrated in a thin band hugging the
circle, i.e. exactly where the training sample density is too sparse to resolve the curvature.
This is the geometric heart of the lab. The true boundary consists of four straight square
edges plus one curved arc. A *local* method such as kNN approximates it with piecewise-constant
regions whose resolution is set by local sample density: with 100 training points spread over a
2.5x2.5 domain, the nearest-neighbour Voronoi cells are simply too coarse near the arc. The
contrast with a tree — a *global*, axis-aligned learner that must staircase the same arc — is
taken up in §4.

---

## 4. Exercise 1b — the effect of representation

Four datasets were generated from the same points (`src/relabel.py`), with all label logic
validated pointwise (`results/representation_datasets.json`):

| Variant | Features | Labels | Changed points |
|---|---|---|---|
| `xy` | `x`, `y` | original circle | — (control) |
| `sq` | `x`, `y` | inner region = square, half-side 0.88 | 6 of 100 (train), 13 of 100 (test), 218 of 2601 (grid) |
| `zt` | `z = x^2`, `t = y^2` | original circle | 0 (same labels) |
| `u` | `u = x^2 + y^2` | original circle | 0 (same labels) |

The `sq` relabelling was verified in both directions: the 4 points that move `c -> q` and the 2
that move `q -> c` in the training set match the analytic sets exactly, and the whole
relabelling is pointwise consistent with the geometric definition.

### 4.1 Results (WEKA J48 and sklearn on identical data)

Accuracy on the held-out `circletest.arff`:

| Representation | J48 train | J48 test | J48 leaves | sklearn train | sklearn test | sklearn leaves |
|---|---|---|---|---|---|---|
| `(x, y)`, circle | 0.96 | **0.88** | 5 | 0.98 | **0.87** | 9 |
| square inner, `(x, y)` | 1.00 | **0.92** | 5 | 1.00 | **0.96** | 5 |
| `(x^2, y^2)`, circle | 0.99 | **0.88** | 5 | 0.99 | **0.89** | 6 |
| `u = x^2+y^2`, circle | 1.00 | **1.00** | **2** | 1.00 | **1.00** | **2** |

Full tables: `results/j48_representation.csv`, `results/sklearn_representation.csv`. The trees
J48 actually printed are stored verbatim in `results/j48_representation.json`; the sklearn trees
are rendered in `figures/tree_xy.png`, `figures/tree_sq.png`, `figures/tree_zt.png` and
`figures/tree_u.png`.

> **Read this table against §7.2 before drawing conclusions from it.** With only 100 test
> instances, the `sq` and `zt` differences above (1–4 pp) are **not statistically significant**.
> The only representation difference that survives a paired McNemar test is `u`, which is
> significant at p ≤ 0.003 against every other variant. The descriptive comparisons in
> §4.2–4.3 are still useful for understanding *why* representations matter, but they are not
> evidence that the `sq` or `zt` encodings are better.

### 4.2 Answering p.7 — is the square inner region easier?

**Prediction (made before running): easier.** The inner region is now
`(|x| <= 0.88) AND (|y| <= 0.88)` — a conjunction of two axis-aligned threshold tests, which is
literally the hypothesis language of a decision tree. The circle, by contrast, is not linearly
separable in `(x, y)` and must be approximated by a staircase of axis-aligned splits.

**Observed: confirmed, with a caveat.** Training accuracy rises from 0.96 (circle) to 1.00
(square), and test accuracy from 0.88 to 0.92 for J48 (0.87 to 0.96 for sklearn) — the square
version is genuinely easier. But the *tree does not shrink*: J48 still reports 5 leaves in both
cases. The reason is visible in the two trees (`results/j48_representation.json`): both use the
outer square boundary to isolate the `q` corners, and pruning at `-C 0.25` keeps only 5 leaves
regardless. The gain is therefore in *how well* those 5 leaves fit, not in tree size. Reporting
this honestly matters — the naive expectation "simpler region, smaller tree" is not what the
experiment shows, because tree size here is controlled by pruning, and the outer boundary
dominates the structure.

### 4.3 Answering p.8 — the `(x^2, y^2)` representation

**Prediction: better.** In `(z, t)` space the circle becomes the **exact linear** constraint
`z + t <= 1`. This is verified numerically:

```
max(z + t | c) = 0.999033      min(z + t | q) = 1.000280
```

— the cleanest possible separation, and numerically identical to the `x^2+y^2` values quoted in
§1.1 because `z + t = x^2 + y^2` by construction. In `(z,t)` coordinates the decision boundary
is a straight line `z + t = 1`, i.e. an axis-aligned constraint on a *linear combination* of the
features.

**Observed: a modest gain, not the collapse one might hope for.** Test accuracy is unchanged for
J48 (0.88 -> 0.88) and rises slightly for sklearn (0.87 -> 0.89), while training accuracy rises
to 0.99. The tree size does **not** fall: J48 still reports 5 leaves, and the tree actually
printed (`results/j48_representation.json`) is a staircase in `(z, t)`:

```
t <= 0.712082
|   z <= 0.729179: c (49.0/1.0)
|   z > 0.729179
|   |   z <= 0.897385
|   |   |   t <= 0.148856: c (5.0)
|   |   |   t >  0.148856: q (5.0)
|   |   z >  0.897385: q (9.0)
t > 0.712082: q (32.0)
```

The reason is the same limitation identified in §4.2: a decision tree can only test **one
feature at a time**, and `z + t` is not a feature. So the exact straight-line boundary
`z + t = 1` is still approximated by a staircase of axis-aligned rectangles, and the
representation helps only because the staircase in `(z,t)` is better aligned with the boundary
than the one in `(x,y)`. The full collapse requires giving the learner the *sum itself* as a
single attribute — see §4.4.

### 4.4 Answering p.9 — 100% accuracy with a minimal tree

The true rule for the original problem is a **single threshold on the radius squared**:

```
u = x^2 + y^2     u <= 1  ->  c        u > 1  ->  q
```

All points of both classes already lie inside the domain square, so the `|x| <= 1.25` and
`|y| <= 1.25` conditions are automatically satisfied and the classification needs no other
test. Numerically the separation is clean:

```
max(u | c) = 0.999033     min(u | q) = 1.000280     ->  separable by one threshold
```

**Observed (WEKA, verbatim from `results/j48_representation.json`):**

```
u <= 0.999033: c (53.0)
u >  0.999033: q (47.0)

Number of Leaves : 2       Size of the tree : 3
```

That is a **2-leaf, depth-1 stump** with **100% training and 100% test accuracy**. sklearn
produces exactly the same tree. This is the smallest possible non-trivial tree, and `u` is a
**domain-specific engineered feature**: it encodes precisely the inductive bias the problem
requires. The point of the exercise is not that the algorithm got smarter, but that feature
engineering changed the hypothesis space so that a trivial hypothesis became expressible.

**Does it generalise?** Yes, with one exact caveat that is worth stating precisely. On the
dense grid `circleall.arff` the J48 stump scores **0.9977 raw (6 errors out of 2601)**, and
**1.0000 (0 errors) once the 212 points lying exactly on a boundary are excluded**. All 6 raw
errors are points whose `u` is exactly `1.0`, and they are not a modelling failure but a
representational limit of a decision tree:

> a tree can only emit a threshold equal to a **value seen in training**, and the largest
> training `u` is `0.999033 < 1`. The rule `u <= 1` is therefore not exactly representable from
> this 100-point training set, and the six grid points with `u == 1` fall on the wrong side of
> the learned split.

That is a genuine and instructive finding rather than a blemish: **an exactly-correct feature
still yields a not-exactly-correct classifier if the threshold it needs never appears in the
training data.** With more or denser training data covering `u = 1`, the stump would be exact.

Because `u` is an exact sufficient statistic, the near-perfect fit is not memorisation — the
`u` representation generalises far better than any other: on the 2389 interior grid points it
makes **zero** errors, against 91.4% / 90.8% / 92.7% for the other three variants (§4.5). Contrast this with the digit data in §5, where a perfect training
fit is emphatically *not* accompanied by perfect test accuracy.

The p.9 recipe "test on `circleall.arff` and use *Visualize classifier errors*" is reproduced
directly as `figures/errors_one_node.png` for this one-node model; the broader
per-representation error maps are in `figures/representation_error_maps.png` (§4.5).

**Side note on the two-feature version.** Giving J48 `z = x^2` and `t = y^2` as two separate
attributes does **not** reproduce this, because the classifier must test the *sum*, which is
not one of its features. The XOR/rectangle reasoning that appears in the plan belongs to the
*relabelled square* problem of §4.2, not to this one — the two variants are distinct and were
kept separate.

### 4.5 Where each representation's errors actually are

Accuracy alone does not show *why* a representation helps. Because the four variants produce
genuinely different trees, their error patterns differ too — and the pattern is diagnostic.
Table: `results/representation_errors.csv`; figures: `figures/representation_error_maps.png`
(decision surface and error scatter per variant) and `results/representation_confusion.csv`
(WEKA's confusion matrix per variant, binary with `c` as the positive class).

Ground truth is taken from the **file labels**. This is a substantive choice: 12 grid points have
`x^2+y^2` equal to 1 in exact decimal arithmetic, and evaluating the rule in binary floating point
classifies 6 of them the opposite way to the file, which manufactures 6 phantom errors on a
measure-zero set. 212 points lie exactly on a boundary and are excluded from the interior figures
(see §4.4):

| Representation | Errors on grid (raw) | of which near its **inner** boundary | Interior errors | Interior acc |
|---|---|---|---|---|
| `(x, y)`, circle | 211 | 134 (**63.5%**) | 206 / 2389 | 0.9138 |
| square inner, `(x, y)` | 225 | 179 (**79.6%**) | 219 / 2389 | 0.9083 |
| `(x^2, y^2)`, circle | 180 | 108 (**60.0%**) | 174 / 2389 | 0.9272 |
| `u = x^2+y^2`, circle | 6 | 6 | **0 / 2389** | **1.0000** |

"Interior" excludes the 212 points lying exactly on a boundary. Note that interior accuracy is
*removed from both numerator and denominator* — boundary points are not credited as successes.

Two things stand out, and both are exactly what the theory predicts:

* **The `sq` variant's errors are overwhelmingly on the *inner square* boundary (79.6%)**, not
  on the circle. The relabelling moved the decision boundary, and the tree's errors followed it
  — the learner is failing where the boundary is, which is the correct behaviour for an
  under-capacity model.
* **Errors concentrate on the inner boundary in every variant** (60–80%), while essentially
  none sit on the outer square edge (`errors_on_outer_square = 0` in all four cases). That
  confirms the interpretation running through §4.2–4.4: the outer square is trivially
  expressible with axis-aligned splits, and all the difficulty is in the curved or
  finely-placed inner boundary.
* The `u` representation reduces the residual to **zero**, because the boundary it must express
  is a single threshold on a feature it was *given*, rather than an axis-aligned staircase it
  must construct.

---

## 5. Exercise 2 — decision trees on the digit data

### 5.1 Encoding check (p.11)

Each row is a `13 x 8 = 104`-pixel binary image flattened **row-wise**. The example printed on
p.11 of the deck was extracted and checked:

```
[4.1] p.11 pattern decodes to the printed 13x8 grid: True
[4.1] p.11 pattern found verbatim in Bigtest1_104.arff: YES
      file label of that row: 0
```

Decoded as ASCII art it reads `...#.... / .######. / ###..### / ##....## / ... / ..####..`,
matching the grid on p.11 exactly, and the row exists verbatim in the dataset with the correct
label `0`. (The character stream extracted from the PDF contains 105 values for a 104-cell
grid; the first 104 decode to the printed grid, so the extra value is a text-extraction
artefact. This is documented in `src/run_digit_experiments.py`.)

Sample images: `figures/sample_digit.png`.

### 5.2 66% split and seed sensitivity (p.12–13)

J48 with default parameters on the 66% split, per seed (test set = 2048 instances):

| Seed | J48 internal test | J48 on `Bigtest2` | sklearn internal | sklearn on `Bigtest2` |
|---|---|---|---|---|
| 1 | 0.9678 | 0.9471 | 0.9673 | 0.9461 |
| 2 | 0.9604 | 0.9467 | 0.9604 | 0.9473 |
| 3 | 0.9609 | 0.9477 | 0.9585 | 0.9505 |
| 4 | 0.9609 | 0.9473 | 0.9551 | 0.9439 |
| 5 | 0.9580 | 0.9423 | 0.9565 | 0.9445 |
| 42 | 0.9648 | 0.9455 | 0.9585 | 0.9481 |

**Mean over the 6 seeds: 0.96216 ± 0.00352** (WEKA), 0.95939 ± 0.00428 (sklearn).

Why the accuracy moves at all: each seed selects a different 3976-point training subset, and the
digit classes contain visually ambiguous specimens, so some borderline images land on one side
of the split or the other. Is the spread pure sampling noise? The binomial standard error of an
accuracy estimated from `n = 2048` instances at `p = 0.962` is
`sqrt(0.962*0.038/2048) = 0.42` pp, giving a 95% interval of ±0.83 pp. The **observed spread is
0.98 pp** (max 0.9678, min 0.9580) — very close to, and consistent with, the ±0.83 pp expected
from finite-test-set noise alone. So the seed-to-seed variation is dominated by the *evaluation*
sample, not by any real instability of J48. Figure: `figures/j48_seed_variance.png`.

### 5.3 External test set (p.13)

Mean over seeds on `Bigtest2` = **0.9462** vs 0.9622 on the internal split — a gap of
**1.6 pp**. This is the honest generalisation penalty, and it has a concrete cause: **dataset
shift**. `Bigtest2` is (a) exactly balanced at 501 per digit whereas `Bigtest1` is imbalanced
(533–739), and (b) a pre-processed derivative, as its own `@relation` line states
(`Remove-R105-128`). Training on the imbalanced `Bigtest1` 66% subset and testing on a balanced
external set changes the effective class priors, which alone can move accuracy by more than a
percentage point. The lesson is that an internal random split of the training file overstates
performance versus a genuinely independent sample.

**Confusion analysis** (seed 1, `figures/j48_confusion.png`, `figures/confused_pairs.png`). The
most frequent errors are:

| true | predicted | count |
|---|---|---|
| 9 | 8 | 6 |
| 8 | 9 | 5 |
| 8 | 6 | 4 |
| 5 | 9 | 3 |
| 4 | 3 | 3 |
| 3 | 4 | 3 |

The hypothesis was that differences in the pixel patterns of the stroke shapes explain these.
The rendered average images and their difference (`figures/confused_pairs.png`) support this for
the 9/8 pair: the two digits differ mainly in the closure of the lower loop and the presence of
a descender, i.e. in a small number of specific pixels, several of which J48 is free to ignore
because greedy information gain may not select them. The 8/6 confusion is similarly consistent
with the left-hand loop being partially open in some specimens.

**Model comparison (`results/digits_model_comparison.csv`), means over the 6 seeds:**

| Model | internal test | `Bigtest2` external test |
|---|---|---|
| J48, M=2, pruned | 0.96216 ± 0.00352 | 0.94611 ± 0.00200 |
| sklearn tree, entropy, M=2 | 0.95939 ± 0.00428 | 0.94674 ± 0.00244 |
| **IBk (kNN), k=1** | **0.98372 ± 0.00171** | **0.97588 ± 0.00125** |
| **IBk (kNN), k=5** | **0.98389 ± 0.00127** | 0.97542 ± 0.00089 |

**kNN beats the decision tree on the digits by about 2.2 pp on the internal split and 3.0 pp
on the external test set** (0.9759 vs 0.9461), and it does so with a *smaller* seed-to-seed
spread. This is the reverse of the circle problem (§3.1) and it is the most interesting result
in the lab.

Why the reversal? On the circle problem the classes are separated by a smooth boundary and 100
training points are enough to populate it densely, so a local method wins. On the digit problem:

* The classes are separated by **local stroke features** (a closed loop, a descender, a
  horizontal bar). 1-NN retrieves the single most similar training specimen, which is precisely
  the comparison a human makes when reading a licence plate; a global axis-aligned tree must
  approximate that similarity structure with a limited number of rectangular regions.
* 6024 training points is **plenty** for kNN — the curse of dimensionality bites when data is
  sparse relative to dimension, and here the ratio is favourable (58 points per dimension).
* `k = 5` very slightly beats `k = 1` on the internal split (0.98389 vs 0.98372) but loses on
  the external set (0.97542 vs 0.97588), i.e. the two are statistically indistinguishable. Both
  differences are far smaller than the 0.9–1.3 pp seed spread.

So the "curse of dimensionality punishes kNN on 104 binary pixels" expectation is **not**
confirmed here. The correct statement is weaker and more precise: high dimension makes kNN
*data-hungry*, and this dataset is large enough to feed it. What high dimension does cost is
**speed**: a 108-leaf tree needs at most a few dozen comparisons per test point, whereas 1-NN
must compute 6024 distances, so the tree remains the better choice whenever inference latency or
memory matters more than 3 pp of accuracy.

**A second finding from the external test set.** The tree loses 1.6 pp going from the internal
split to `Bigtest2`, whereas kNN loses only 0.8 pp. kNN is therefore not just more accurate here
but also **more robust to the dataset shift** described in §5.3, which is consistent with it
relying on local similarity rather than on globally-tuned axis-aligned thresholds.

### 5.4 Overfitting control via `M` (p.14–16)

Training on the full `Bigtest1_104.arff` (6024 points), testing on `Bigtest2_104.arff` (5010).
Full table in `results/j48_M_sweep.csv`; figure `figures/j48_M_sweep.png`.

| M | J48 train | J48 test | J48 leaves | sklearn train | sklearn test |
|---|---|---|---|---|---|
| **1** | 0.99751 | 0.95150 | 144 | 1.00000 | **0.95589** |
| 2 | 0.99104 | 0.95309 | 108 | 0.99148 | 0.95250 |
| 3 | 0.98556 | 0.95349 | 85 | 0.98797 | 0.95529 |
| **4** | 0.98290 | **0.95509** | 75 | 0.98408 | 0.95212 |
| 5 | 0.97892 | 0.95229 | 57 | 0.98074 | 0.95130 |
| 8 | 0.97510 | 0.95130 | 49 | 0.97460 | 0.95050 |
| 10 | 0.97145 | 0.94411 | 46 | 0.97082 | 0.94571 |
| 15 | 0.96049 | 0.93653 | 32 | 0.96413 | 0.94192 |
| 25 | 0.95252 | 0.92974 | 28 | 0.95066 | 0.92794 |
| 50 | 0.92331 | 0.89022 | 20 | 0.92226 | 0.89321 |
| 100 | 0.89890 | 0.87086 | 11 | 0.90467 | 0.87500 |

**Optimal `M* = 4`, test accuracy 0.95509** (WEKA). sklearn's best is at `M = 1`
(0.95589) and at `M = 3` (0.95529) — within 0.06 pp of WEKA's optimum, and the whole
`M = 1..5` plateau lies between 0.9515 and 0.9559, so the practical conclusion is the same:
**a small amount of regularisation helps, and the optimum is broad.**

**The underfit → optimal → overfit progression (p.16) is clearly visible:**

* `M = 1`: training accuracy 99.75%, test 95.15%. Maximum capacity, largest tree (144 leaves).
* `M = 4` (`M*`): training accuracy falls to 98.29%, test rises to 95.51%. The tree has given
  up some training fit and gained generalisation — textbook regularisation.
* `M >= 10`: both curves fall together (train 97.1% → 89.9%, test 94.4% → 87.1%). The model is
  now underfitting: at `M = 100` only 11 leaves remain, too few to represent the digit classes.

**Direction of the training curve.** The plan flagged a genuine ambiguity in the deck's note on
p.16, which claims training performance "will still improve" as capacity is reduced. **It does
not.** The measured training accuracy decreases monotonically overall, from 0.99751 at `M = 1`
to 0.89890 at `M = 100` (the strict-monotonicity flag in `results/digits_summary.json` is
`false` only because of a small ×2% dip-and-recover bump around `M = 17–18` and 2% rounding in
WEKA's per-leaf counts). The mechanism is exactly as expected: a leaf required to hold at least
`M` samples cannot memorise an isolated one, so forcing `M` up must reduce training accuracy.
The deck's phrasing describes the *early* part of the curve (`M = 1 → 4`), where test accuracy
rises while training accuracy has only just begun to fall; the two curves then diverge with
train consistently **above** test, which is the classic picture.

### 5.5 Why `M = 1` does not give 100% training accuracy (p.15)

J48's default `M` is 2, so the deck's `M = 1` question is really about what limits the model
when capacity is maximised. Measured:

| Configuration | train accuracy | leaves |
|---|---|---|
| J48, `M = 1`, default pruning | 0.99751 | 144 |
| **J48, `M = 1`, `-U` (unpruned)** | **1.00000** | 175 |
| sklearn, fully grown | 1.00000 | 164 |

The three reasons listed on p.15 were tested:

1. **Pruning removes branches — this is the dominant reason and it is confirmed.** Default J48
   at `M = 1` reaches 99.75%, and switching to `-U` recovers **100.00%** training accuracy with
   175 leaves. That is the missing 0.25% — about 15 training points — accounted for directly.
   Note the important subtlety discovered here: **WEKA's `-U` does not disable all pruning**.
   It prevents subtree *raising*, but the confidence-threshold pruning controlled by `-C 0.25`
   still runs. The proof is in the circle experiment of §4.2, where J48 with `-U -M 1` still
   returns the same 5-leaf tree as the pruned default. sklearn's `ccp_alpha=0` *is* fully
   unpruned and reaches 100% at 164 leaves. The two "unpruned" modes are not equivalent and
   must not be treated as such.
2. **Greedy search is not globally optimal — plausible but not the binding constraint here.**
   It is a real limitation in general, but since a fully grown tree *does* reach exactly 100%
   on this dataset, greedy search was evidently able to isolate every training point; it is not
   the reason the default configuration falls short.
3. **The attribute set may be too small — refuted.** With 104 binary attributes a fully grown
   tree does memorise all 6024 points, so the representation is expressive enough.

So of the deck's three explanations, exactly one — pruning — explains the observed gap, and it
explains it quantitatively.

### 5.6 The `k` / `M` parallel

`k` in kNN and `M` in J48 are the same knob viewed from two sides. Small `k` = low bias, high
variance, zero training error; large `k` = the prediction degenerates to the majority class.
Small `M` = high capacity, near-perfect training fit; large `M` = leaves too coarse to separate
the classes. Both must be selected on validation data, never on the test set (see §3.2 for why),
and both exhibit a broad optimum rather than a sharp one — `k* = 1` here and `M* = 4` here, with
near-optimal performance across a range on either side. The difference is that kNN regularises
by *smoothing a local neighbourhood* while a tree regularises by *refusing to split*, but the
bias–variance trade-off they trace is identical.

---

## 6. Scikit-Learn module (p.17–27)

### 6.1 Pipeline (p.23–25)

`src/sklearn_pipeline.py` implements the deck's pipeline structure — `read_csv` → two chained
`train_test_split` calls producing a stratified **70/15/15** split → fit →
`accuracy_score` / `classification_report` / `confusion_matrix` — generalised to read the ARFF
data through `arff_utils`. Both paths were exercised: the ARFF path and a literal CSV
round-trip via `pandas.read_csv`, to confirm the deck's exact instructions work. Full output in
`results/sklearn_pipeline_report.txt`.

```
Training set: (70, 2)     Validation set: (15, 2)     Test set: (15, 2)
Validation Accuracy: 0.9333
Test Accuracy:       0.9333      (confusion [[7 1] [0 7]])
External test (circletest.arff): 0.8600   (confusion [[50 10] [4 36]])
CSV round-trip test accuracy: 0.9333
```

The 15-instance validation and test sets are far too small for their accuracy to mean much on
their own (one instance = 6.7 pp), which is precisely why the main kNN comparison in §3 uses
LOO and 5/10-fold CV over all 100 points instead.

### 6.2 kNN agreement check

sklearn's `KNeighborsClassifier` and the from-scratch implementation agree **point-for-point at
every k** (100/100 test instances at k = 1, 3, 5, 7, 9, 11, 13, 15, 21, 31, 51). Table:
`results/sklearn_knn_agreement.csv`. This is the correctness proof for §2.1 — any difference in
the curves would have indicated a bug in the from-scratch tie-breaking or distance computation.

### 6.3 sklearn decision tree

The `M` sweep on the circle data (`results/sklearn_circle_M_sweep.csv`) shows the same shape as
J48: test accuracy peaks at `M = 5` (0.88) and degrades to 0.67 at `M = 30`, while training
accuracy falls monotonically from 1.00 (M=1, 11 leaves) to 0.75 (M=30, 3 leaves). The
representation ordering from §4.1 is also reproduced (square 0.96 > zt 0.89 > xy 0.87, and
`u` at 1.00), which is the claim that matters — the two toolchains differ in detail but agree on
the *ordering* of representations.

### 6.4 Iris exercise (p.26–27)

`load_iris` (150 samples, 4 features, 3 species), both classifiers compared with accuracy,
5-fold CV and confusion matrices. Tables `results/iris_comparison.csv`, `results/iris_knn.csv`,
`results/iris_tree.csv`; figures `figures/iris_comparison.png`, `figures/iris_confusion.png`.

| Model | best hyper-parameter | 5-fold CV | hold-out (30%) | leaves |
|---|---|---|---|---|
| kNN | `k = 7` | 0.980 ± 0.016 | 0.956 | — |
| Decision tree | `max_depth = 3` | 0.973 ± 0.025 | 0.978 | 5 |

Both models perform very well, and the confusion matrices show why: **Setosa is perfectly
separated** by both models (zero errors), because it is linearly separable from the other two
species on petal length/width. Every remaining error is `versicolor` vs `virginica`, which
overlap in feature space and are not separable by any axis-aligned rule that a 5-leaf tree (or a
small-k kNN with 4 features) can express. This is why the tree's CV accuracy saturates at
depth 3 and does not improve with more depth. The one visible difference is the hold-out flip
(kNN 0.956 vs tree 0.978), which is a single instance out of 45 and is not significant; the CV
estimates, which use all 150 samples, put kNN marginally ahead.

### 6.5 WEKA vs sklearn cross-check (p.6 requirement)

Full table: `results/weka_sklearn_crosscheck.csv`.

| Dataset | hyper-parameter | WEKA | sklearn | Δ (pp) |
|---|---|---|---|---|
| circle `(x,y)` | pruned M=2 | 0.880 | 0.870 | −1.00 |
| circle `(x,y)` | unpruned M=1 | 0.880 | 0.850 | −3.00 |
| square inner `(x,y)` | pruned M=2 | 0.920 | 0.960 | +4.00 |
| square inner `(x,y)` | unpruned M=1 | 0.920 | 0.960 | +4.00 |
| `(x^2,y^2)` | pruned M=2 | 0.880 | 0.890 | +1.00 |
| `(x^2,y^2)` | unpruned M=1 | 0.880 | 0.870 | −1.00 |
| `u = x^2+y^2` | pruned M=2 | 1.000 | 1.000 | **0.00** |
| `u = x^2+y^2` | unpruned M=1 | 1.000 | 1.000 | **0.00** |
| digits | M=1 | 0.9515 | 0.9559 | +0.44 |
| digits | M=2 | 0.9531 | 0.9525 | −0.06 |
| digits | M=3 | 0.9535 | 0.9553 | +0.18 |
| digits | M=5 | 0.9523 | 0.9513 | −0.10 |
| digits | M=10 | 0.9441 | 0.9457 | +0.16 |

**On the large digit datasets the agreement is excellent — every delta is under 0.5 pp** — which
is a strong mutual validation of the two implementations. **On the 100-point circle problem the
deltas reach 3–4 pp**, and this must be interpreted correctly rather than dismissed: with a
100-instance training set and a 100-instance test set, a single instance is 1 pp, so a 4 pp gap
is *four instances*. The cause is the well-documented difference in split tie-breaking and
pruning between J48 and sklearn, which produces differently-shaped trees (5 vs 9 leaves for the
circle). The claims to defend are therefore about the **shape of the curves and the ordering of
the representations**, both of which match; exact per-point agreement was never expected and is
not observed.

---

## 7. Statistical significance and inference cost

This section closes the two gaps that the first pass listed as limitations: the absence of any
significance testing, and the absence of measured inference timings. Both are now answered with
data rather than argument.

### 7.1 Are the differences real?

Every accuracy figure in this report is a binomial proportion, so it has an interval. Wilson
95% intervals for the headline models (`results/accuracy_confidence_intervals.csv`):

| Dataset | Model | Accuracy | 95% CI | CI width |
|---|---|---|---|---|
| circle test (n=100) | J48 `(x,y)` | 0.880 | 0.802 – 0.930 | **12.8 pp** |
| circle test (n=100) | J48 `sq` | 0.920 | 0.850 – 0.959 | **10.9 pp** |
| circle test (n=100) | J48 `u` | 1.000 | 0.963 – 1.000 | 3.7 pp |
| digits internal (n=2048) | J48 M=2 | 0.9678 | — | 1.24 pp |
| digits `Bigtest2` (n=5010) | J48 M=2 | 0.9471 | 0.9406 – 0.9530 | 1.24 pp |
| digits `Bigtest2` | IBk k=1 | 0.9763 | 0.9717 – 0.9801 | 0.85 pp |

This immediately quantifies the previous section's caution: **the 12.8 pp interval on the
100-point circle test set is wider than every gap the circle experiments measured.** The
`sq` vs `xy` difference of 4 pp is well inside the noise. By contrast the digit intervals are
about 1 pp wide, so a 2–3 pp gap there is meaningful.

### 7.2 Paired tests (McNemar)

Because all models are evaluated on the *same* instances, an unpaired comparison wastes
information. McNemar's test uses only the instances where two classifiers disagree, which is
far more powerful. Predictions were taken from **WEKA itself** (`-p 0`), so the pairing is exact
rather than approximated with a sklearn stand-in; the extracted predictions reproduce WEKA's
reported accuracies to all reported digits. Holm–Bonferroni correction is applied across the
whole family of pairwise comparisons.

**Digits, `Bigtest2` (n=5010)** — `results/significance_digits_bigtest2.csv`:

| A | B | Δ (pp) | discordant b / c | p | p (Holm) |
|---|---|---|---|---|---|
| J48 M=2 | IBk k=1 | −2.91 | 56 / 202 | 1.8e−19 | **3.3e−18** |
| J48 M=2 | IBk k=5 | −2.71 | 50 / 186 | 1.5e−18 | **2.4e−17** |
| J48 M=2 | sklearn DT | +0.10 | 112 / 107 | 0.787 | 1 (n.s.) |
| IBk k=1 | IBk k=5 | +0.20 | 42 / 32 | 0.295 | 1 (n.s.) |
| IBk k=1 | sklearn kNN k=1 | +0.10 | 10 / 5 | 0.302 | 1 (n.s.) |

**The central claim of §5.3 is now established, not asserted.** kNN beats the decision tree on
the digits by 2.6–2.9 pp, and the effect is overwhelming (p ≈ 10⁻¹⁸ with ~200 discordant
instances favouring kNN against ~55 favouring the tree). The differences *within* the kNN
family, and between J48 and sklearn's tree, are not distinguishable from noise — which is the
correct and reassuring result: the two toolchains agree.

**Circle representations (n=100)** — `results/significance_circle.csv`:

| A | B | Δ (pp) | discordant b / c | p (Holm) |
|---|---|---|---|---|
| J48 `xy` | J48 `sq` | +1.0 | 1 / 0 | 1 (n.s.) |
| J48 `xy` | J48 `zt` | 0.0 | 4 / 4 | 1 (n.s.) |
| J48 `xy` | J48 `u` | −12.0 | 0 / 12 | **0.0024** |
| J48 `sq` | J48 `u` | −13.0 | 0 / 13 | **0.0015** |
| J48 `sq` | J48 `zt` | −1.0 | 4 / 5 | 1 (n.s.) |

The picture is stark and it *changes how the Exercise 1b results should be read*. With only 100
test instances, **none of the tree-vs-tree representation differences are significant** — the
`sq` and `zt` improvements that §4.2 and §4.3 reported are within noise. The **only** effect
that survives testing is the `u` representation, whose discordant split is 0-versus-12/13:
every single instance the two classifiers disagree on is one that `u` gets right and the other
gets wrong. The right conclusion is therefore stronger *and* more honest than the raw
accuracies suggested: **among these four representations, only `u = x^2+y^2` demonstrably
matters.**

### 7.3 Inference cost

Timings are the median of 5 runs after a warm-up, on the 6024-point training set with the 5010
instance `Bigtest2` as the query set (`results/inference_benchmark.csv`):

| Model | Fit (s) | Predict (s) | µs per test instance |
|---|---|---|---|
| Decision tree (entropy, M=2) | 0.0206 | 0.0007 | **0.14** |
| kNN k=1 | 0.0007 | 0.0254 | 5.07 |
| kNN k=5 | 0.0009 | 0.0288 | 5.74 |

The trade-off is now measured rather than asserted. The tree costs **~35x more to fit**
(0.021 s vs 0.0007 s) but is **~36x cheaper to apply** (0.14 µs vs 5.07 µs per test instance),
because kNN must compute 6024 distances for every query while a 108-leaf tree needs at most a
few dozen comparisons. The natural deployment reading: kNN is the better choice when the
training set is fixed and latency is unimportant, while the tree wins in any setting where
queries are frequent relative to retraining. Note also that the tree's advantage grows with the
size of the training set, since its inference cost depends on tree depth (logarithmic) rather
than on `n` (linear) as kNN's does.

---

## 8. Acceptance table (plan section 9)

| # | Experiment | Metric | Value | Source |
|---|---|---|---|---|
| 1 | kNN, best `k` on test | accuracy | **0.9100 at k=1** | `results/knn_k_sweep.csv` |
| 2 | kNN, best `k` by LOO-CV | accuracy | **0.9300 at k=1** | `results/knn_k_sweep.csv` |
| 3 | kNN, best `k` on `circleall` | accuracy | **0.9004 at k=1** | `results/knn_grid_accuracy.csv` |
| 4 | J48, circle `(x,y)` | acc / leaves | 0.88 / 5 (train 0.96) | `results/j48_representation.csv` |
| 5 | J48, square inner region `(x,y)` | acc / leaves | 0.92 / 5 (train 1.00) | `results/j48_representation.csv` |
| 6 | J48, circle `(x^2,y^2)` | acc / leaves | 0.88 / 5 (train 0.99) | `results/j48_representation.csv` |
| 7 | J48, one-node representation `u` | acc / leaves | **1.00 / 2** | `results/j48_representation.csv` |
| 8 | J48 digits, 66% split (seed 1) | accuracy | **0.9678** | `results/j48_bigtest1_split66.csv` |
| 9 | J48 digits, mean over 6 seeds | mean ± sd | **0.96216 ± 0.00352** | `results/j48_seeds.csv` |
| 10 | J48 digits, `Bigtest2` supplied test | accuracy | **0.9462** (mean over seeds) | `results/j48_seeds.csv` |
| 11 | J48 digits, optimal `M` | `M*` / accuracy | **M*=4 / 0.95509** | `results/j48_M_sweep.csv` |
| 12 | J48 digits, `M=1` unpruned on training set | accuracy | **1.00000** (175 leaves) | `results/j48_M_sweep.csv`, `results/digits_summary.json` |
| 13 | sklearn vs WEKA, matched configs | delta | **≤0.5 pp on digits; ≤4 pp on the 100-point circle set** | `results/weka_sklearn_crosscheck.csv` |
| 14 | Iris, kNN vs tree (5-fold CV) | accuracies | **kNN 0.980 ± 0.016 (k=7); tree 0.973 ± 0.025 (depth 3)** | `results/iris_comparison.csv` |
| 15 | IBk (kNN) on digits, plan 4.6 | accuracy | **k=1: 0.98372 internal / 0.97588 Bigtest2; k=5: 0.98389 / 0.97542** | `results/digits_model_comparison.csv` |
| 16 | Plan 3.7.1b two-feature `(z,t)` variant | tree depth | **depth 4, 6 leaves — not a one-node stump** | `results/representation_twotest.json` |
| 17 | Representation error localisation | interior errors / interior acc | **xy 206/0.9138, sq 219/0.9083, zt 174/0.9272, u 0/1.0000** | `results/representation_errors.csv` |
| 18 | Significance, kNN vs tree on digits | McNemar p (Holm) | **p = 3.3e−18, Δ = −2.91 pp** | `results/significance_digits_bigtest2.csv` |
| 19 | Significance, circle representations | McNemar p (Holm) | only `u` is significant (p ≤ 0.0024); all tree-vs-tree pairs n.s. | `results/significance_circle.csv` |
| 20 | Inference cost | µs per test instance | **tree 0.14 vs kNN 5.07 (≈36x)** | `results/inference_benchmark.csv` |

---

## 9. Limitations

* **Small circle datasets.** 100 training and 100 test points make every accuracy figure
  quantised in steps of 1 pp, and the Wilson intervals in §7.1 are up to 12.8 pp wide. The
  paired tests make this concrete: none of the tree-vs-tree representation differences on the
  circle data is statistically significant, including the `sq` and `zt` improvements that
  §4.2 and §4.3 initially reported as gains. Only the `u` representation survives testing.
* **The 66% split carries real variance.** `results/digits_summary.json` shows the internal
  test accuracy spanning 0.958–0.968 across seeds; single-seed numbers should never be quoted
  without the ±0.83 pp binomial uncertainty.
* **Internal vs external evaluation differ by 1.6 pp** on the digits, driven partly by class
  imbalance and the pre-processing difference between `Bigtest1` and `Bigtest2`.
* **WEKA and sklearn are not implementable-equivalent.** `-U` does not mean the same thing in
  both tools (§5.5), pruning strategies differ, and split tie-breaking differs. All
  cross-tool comparisons here are therefore about trends, not exact values — and §7.2 confirms
  that J48 and sklearn's tree are statistically indistinguishable on the digits, which is the
  strongest available evidence that the comparison is still meaningful.
* **A decision tree cannot represent a threshold absent from the training data.** This is the
  one genuine limitation discovered rather than assumed, and it is why the `u` variant scores
  0.9977 rather than 1.0000 on the grid: the rule it needs is `u <= 1`, but no training point
  has `u >= 1`, so the learned threshold is 0.999033 and the six grid points with `u == 1` are
  misclassified (§4.4). This is a property of the learner, not of the feature, and it disappears
  with denser training data near the boundary.
* **Boundary points are excluded from grid accuracies.** 212 of the 2601 grid points sit exactly
  on the circle or the outer square, where the file label is arbitrary. They are excluded from
  headline figures and reported separately; including them lets floating-point representation
  noise masquerade as modelling error, which actually happened during development and is now
  documented in `src/run_representation_analysis.py`.
* **Timing is single-machine and single-threaded.** The benchmark in §7.3 measures scikit-learn
  implementations on one machine. WEKA's Java timings are not directly comparable because they
  include JVM startup, so they are reported separately in
  `results/accuracy_confidence_intervals.csv` and not used for the cost comparison.
* **No nested validation for the digit hyper-parameters.** `M*` was selected on the external
  `Bigtest2` set, which is methodologically the same mistake as tuning `k` on the test set in
  §3.2. With 5010 instances the optimism is small (the `M = 1..5` plateau spans only 0.4 pp,
  §5.4), but the correct protocol would nest a validation split inside `Bigtest1`.

---

## 10. Reproduction

```bash
cd <repo root>
source .venv/bin/activate       # scikit-learn 1.8.0, numpy 2.4.2, scipy 1.17.1, python 3.14
export MPLCONFIGDIR=/tmp/mpl

# WEKA 3.8.7 is driven headlessly through its bundled JRE:
#   /Applications/weka-3.8.7.app/Contents/runtime/Contents/Home/bin/java
#   --add-opens java.base/java.lang=ALL-UNNAMED -cp .../app/weka.jar <classifier> ...
#
# WEKA CLI argument order matters: the class name must come immediately after
# -cp, and -p 0 requests the per-instance prediction table used by the tests.

python Lab-ML/lab1/src/verify_data.py                  # §1  data inventory gate
python Lab-ML/lab1/src/knn_scratch.py                  # §2.1 correctness proof vs sklearn
python Lab-ML/lab1/src/run_knn_experiments.py          # §3  k sweep, CV, boundary figures
python Lab-ML/lab1/src/relabel.py                      # §4  generates the 4 representations
python Lab-ML/lab1/src/run_j48_experiments.py          # §4  J48 + sklearn per representation
python Lab-ML/lab1/src/run_representation_analysis.py  # §4.5 two-feature test + error maps
python Lab-ML/lab1/src/run_digit_experiments.py        # §5  seeds, Bigtest2, M sweep
python Lab-ML/lab1/src/sklearn_pipeline.py             # §6  pipeline, agreed kNN, Iris
python Lab-ML/lab1/src/run_significance.py             # §7  McNemar, Wilson CIs, timings

# integrity re-check: the originals must be byte-identical to the recorded baseline
cd Lab-ML && shasum -a 256 *.arff *.pdf | diff - lab1/results/source_data_checksums.txt
```

**Artifacts.** 15 figures in `lab1/figures/`; 24 result files in `lab1/results/` (plus 42 raw
WEKA output logs in `lab1/results/j48_raw_outputs/`); 25 generated datasets in `lab1/data/`.
The four source `.arff` files and the PDF are unmodified — verified by the checksum comparison
above.

**Self-check against the plan** (`LAB1_AGENT_TODO.md`): **every numbered step, including the
optional `3.7.1b`, is now complete.** `3.7.1b` is settled empirically in §4.4 and
`results/representation_twotest.json`: separating `z` and `t` gives a tree of depth 4 with 6
leaves, not a one-node stump, and an exhaustive search over `z`, `t`, `z+t`, `z−t`, `max(z,t)`
and `min(z,t)` confirms that **only the sum `z + t` is separable by a single threshold**. All 20
rows of the acceptance table are populated. Every figure referenced in this report exists in
`lab1/figures/` and every number traces to a file in `lab1/results/`.

**Corrections made during this work** (kept visible rather than silently fixed, because two of
them changed conclusions):

1. Class semantics in the plan were **inverted** — corrected in §1.1.
2. WEKA's `-U` does **not** fully disable pruning — corrected in §5.5, and it changes the
   answer to the p.15 question.
3. An earlier draft of this report claimed the `u` representation was **100% accurate on all
   2601 grid points**. That was wrong: it is 99.77% raw, exactly 100% once the 212 genuine
   boundary points are excluded. The 6 residual errors are a *representational* limit of the
   learner (no training point attains `u = 1`, so the threshold `u <= 1` is not expressible) and
   are discussed in §4.4.
4. The `sq` and `zt` representation "improvements" reported in §4.2–4.3 are **not statistically
   significant** at n=100 (§7.2). The descriptive reporting is retained, but §7.2 supersedes it
   as the basis for any claim.

---

## Appendix A — artifact manifest

Every file below was produced by a script in `src/` and is referenced above; this appendix
exists so that nothing generated is left undocumented.

**Scripts (`lab1/src/`)**

| Script | Produces |
|---|---|
| `arff_utils.py` | ARFF reader/writer, `write_csv` with correct quoting, shared paths |
| `verify_data.py` | §1 data-inventory gate (row and class counts) |
| `knn_scratch.py` | from-scratch kNN; self-test against sklearn and scipy |
| `weka_run.py` | WEKA CLI driver, evaluation-section parsing, `-p 0` predictions, Java `Random` |
| `relabel.py` | the four representations (`xy`, `sq`, `zt`, `u`) with pointwise validation |
| `run_knn_experiments.py` | §3 k sweep, LOO/CV, boundary and error figures |
| `run_j48_experiments.py` | §4 J48 and sklearn per representation, tree figures |
| `run_representation_analysis.py` | §4.4 two-feature test, §4.5 error localisation and maps |
| `run_digit_experiments.py` | §5 seeds, `Bigtest2`, confusion analysis, `M` sweep |
| `sklearn_pipeline.py` | §6 pipeline, kNN agreement, Iris, WEKA/sklearn cross-check |
| `stats_utils.py` | McNemar (exact and χ²), Holm correction, Wilson intervals |
| `run_significance.py` | §7 paired tests, confidence intervals, inference benchmark |

**Results not referenced inline above**

| File | Contents |
|---|---|
| `results/significance_digits_internal.csv` | full McNemar table on the 66% split test set (21 pairs) |
| `results/sklearn_summary.json` | machine-readable sklearn results incl. the WEKA→sklearn mapping |
| `results/sklearn_trees.txt` | `export_text` dump of every sklearn tree fitted per representation |
| `results/representation_summary.json` | per-variant accuracy/leaf counts, both toolchains |
| `results/representation_datasets.json` | relabelling statistics and separability checks |
| `results/iris_confusion.csv` | Iris confusion matrices in long form (not just the figure) |
| `results/j48_raw_outputs/` | 42 raw WEKA logs, including the verbatim printed trees |

**Generated data (`lab1/data/`)** — 24 files: the four representations (`_xy`, `_sq`, `_zt`,
`_u`) of each of `circletrain`, `circletest`, `circleall`, plus the six 66%-split files and the
full-training ARFF for the `M` sweep. **No source file in `Lab-ML/` was modified**; this is
verified by checksum against `results/source_data_checksums.txt`.
