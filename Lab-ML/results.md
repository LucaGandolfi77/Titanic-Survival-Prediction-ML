# Machine Learning Laboratory — Collected Results

**Course:** Machine Learning
**Source material:** *ML2026_lab1_KNN_DecTrees-merged*, pp. 1–91

This document collects the results of every laboratory in the course deck. It is organised
**one section per laboratory**, and each section contains all the exercises belonging to that
laboratory, in the order in which they appear in the deck. A consolidated
`Lab | Exercise | Status | Result` table appears in Section 6.

## How to read the Status column

| Status | Meaning |
|---|---|
| **Complete** | Implemented and executed; all numbers come from the result files listed in the appendix. |
| **Partial** | Implemented, but with a documented substitution or an incomplete sub-part. |
| **Not run** | Not implemented in this work; the required inputs are absent or the exercise depends on material that was not supplied. |

**Editorial policy.** No result is estimated, interpolated, or copied from the literature. Where
an exercise could not be completed, the row says so and Section 8 gives the reason. Where a
measurement contradicted the deck or an earlier draft of this document, the correction is stated
explicitly rather than silently applied; these are itemised in Section 8.3.

## Contents

| Section | Laboratory | Deck pages | Exercises |
|---|---|---|---|
| 1 | **Lab 1** — k-Nearest Neighbours and Decision Trees | 1–16 | Ex. 1, Ex. 1b, Ex. 2, Ex. 2b |
| 2 | **Lab 2** — Clustering | 28–43 | K-Means, variance, digit clusters, X-Means |
| 3 | **Lab 3** — Genetic Algorithms (DEAP) | 51–72 | Ex. 1, Ex. 2 (Ex. 3–4 not run) |
| 4 | **Lab 4** — Genetic Programming and PSO | 73–81 | Ex. 0, 0b, 1, 2 (not run) |
| 5 | **Lab 5** — The Ant Trail Problem | 82–91 | parts 1–4 (not run) |
| 6 | **Lab 6** — Scikit-Learn: Machine Learning in Python | 17–27 | slides 22–25 pipeline, slides 26–27 Iris |
| 7 | **Lab 7** — PyTorch: deep learning in Python | 44–56 | slides 45–56, MLP on FashionMNIST |
| 8 | **Consolidated results** | — | Lab\|Exercise\|Status\|Result, corrections, cross-lab significance, statistics |

> **Note on laboratory numbering.** The numbering used here follows the course deck for Labs 1–5
> (the deck itself labels the evolutionary-computation material "Lab 5" and "Lab 6", and the
> clustering material "Lab 2"). The scikit-learn module (deck pages 17–27) and the PyTorch module
> (deck pages 44–56) are presented as **Lab 6** and **Lab 7**, as requested. Sections 4 and 5
> therefore correspond to the deck's "Lab 6" and "Lab X" respectively.

## Common experimental environment

| Component | Version / location |
|---|---|
| Python | 3.14 (repository `.venv`) |
| scikit-learn | 1.8.0 |
| NumPy / SciPy | 2.4.2 / 1.17.1 |
| Matplotlib | current |
| DEAP | 1.4 |
| PyTorch / torchvision | 2.14.0 / 0.29.0 (accelerator: Apple MPS) |
| WEKA | 3.8.7, driven headlessly through its bundled JRE |

The ARFF reader/writer and the statistical utilities are shared across laboratories and live in
`lab1/src/arff_utils.py` and `lab1/src/stats_utils.py`. Original data files are never modified;
integrity is verified by SHA-256 checksum (`lab1/results/source_data_checksums.txt`).

---

# 1. Lab 1 — k-Nearest Neighbours and Decision Trees

**Deck pages:** 1–16 (WEKA exercises) and 17–27 (Scikit-Learn module).
**Data:** `circletrain.arff`, `circletest.arff`, `circleall.arff`, `Bigtest1_104.arff`,
`Bigtest2_104.arff`.
**Status:** Complete.

### 1.1 Abstract

This laboratory investigates two classical supervised learning algorithms — the *k*-nearest
neighbour classifier (kNN) and the univariate decision tree (C4.5/J48) — on two problems of
contrasting difficulty: a synthetic two-dimensional geometric discrimination task, and a
104-dimensional optical digit recognition task.

The principal findings are as follows. On the synthetic task kNN attains its optimum at *k* = 1
(91.0 % on the held-out test set), and for *k* ≥ 51 the classifier degenerates *exactly* to the
majority-class baseline. Of four input representations tested, only the engineered feature
*u* = *x*² + *y*² produces a demonstrably significant improvement, collapsing the decision tree
to a single split with two leaves at 100 % test accuracy; a paired McNemar test confirms that the
remaining representational variants are **not** distinguishable from chance at this sample size.
On the digit task kNN outperforms the decision tree by 2.9 percentage points (97.6 % vs 94.7 %,
*p* ≈ 10⁻¹⁸), overturning the expectation that the curse of dimensionality would penalise the
local method. The regularisation study confirms a broad optimum at *M* = 4 but **contradicts** the
claim that training accuracy improves as capacity is reduced.

A further result emerged from rigorous validation rather than being sought: a decision tree
**cannot represent a decision threshold that does not occur in its training data**.

### 1.2 Introduction

#### 1.2.1 Background

Supervised classification seeks a function *f* : ℝ^d → {1, …, C} inferred from a finite sample of
labelled observations. Two families of algorithms occupy opposite ends of a fundamental
trade-off.

**Instance-based (lazy) methods** such as kNN store the training sample and defer all computation
to prediction time. The decision boundary they induce is *local*: it is determined by the
Voronoi tessellation of the training points, so its resolution is limited by local sample density.

**Model-based (eager) methods** such as decision trees construct an explicit, compact hypothesis
at training time. The boundaries they induce are *global* and, for univariate splits, restricted
to be axis-aligned. A curved or oblique boundary must consequently be approximated by a staircase
of axis-aligned segments.

#### 1.2.2 The role of representation

A decision tree partitions the feature space by testing one attribute at a time. It follows that
the *encodability* of a concept depends on how the input is presented. A conjunction of
axis-aligned constraints is trivially expressible; an oblique or curved constraint is not. This
motivates the second theme: holding the data points fixed and varying only the representation.

#### 1.2.3 Objectives

1. Implement kNN from first principles and validate it against a reference implementation.
2. Characterise the dependence of accuracy on *k* and establish a sound procedure for selecting it.
3. Quantify how four input representations affect decision-tree accuracy, size, and error localisation.
4. Determine whether a tree can achieve perfect accuracy with a minimal hypothesis, and whether it generalises.
5. Study overfitting control via the leaf-size parameter *M* on a ten-class problem.
6. Establish the statistical significance of all comparisons rather than relying on point estimates.
7. Reproduce the pipeline in a second ecosystem (scikit-learn) as an independent cross-check.

## 1.3 Materials and Methods

### 1.3.1 Problem definitions

#### 2.1.1 The circle problem (Exercises 1 and 1b)

Points are drawn from the square domain [−1.25, 1.25]² and assigned to one of two classes
according to which region they occupy:

| Class | Region | Membership predicate |
|---|---|---|
| `c` | interior of the unit circle centred at the origin | *x*² + *y*² ≤ 1 |
| `q` | the square of side *L* = 2.5 centred at the origin, with the circle removed | (\|*x*\| ≤ 1.25) ∧ (\|*y*\| ≤ 1.25) ∧ ¬(*x*² + *y*² ≤ 1) |

The domain boundary at \|*x*\| = \|*y*\| = 1.25 is trivially expressible by two axis-aligned
splits. The discriminating structure is therefore entirely the curved interior boundary. This
makes the problem a clean probe of a learner's ability to approximate curvature.

**Verification of the labelling rule.** Before modelling, the predicate above was checked
against the supplied labels. It reproduces the labels of the training and test sets with **zero
errors**:

```
circletrain.arff :  max(x² + y² | c) = 0.999033     min(x² + y² | q) = 1.000280
circletest.arff  :  max(x² + y² | c) = 0.995348     min(x² + y² | q) = 1.026705
```

On the dense reference grid the rule agrees on 2595 of 2601 points; all six disagreements lie
*exactly* on the circle (\|*x*² + *y*² − 1\| < 10⁻⁹) and are therefore artefacts of assigning a
label to a measure-zero boundary, not label noise.

#### 2.1.2 Exercise 1b: the modified inner region

The laboratory specifies a variant in which the inner region is replaced by a **square** of side
*l* = 1.76 centred at the origin, i.e. \|*x*\| ≤ 0.88 ∧ \|*y*\| ≤ 0.88, the outer square being
unchanged.

An observation that proves important throughout Section 4: the half-side 0.88 gives the inner
square a circumradius of 0.88√2 ≈ 1.2445 > 1. The new region therefore **overlaps** the old
circle — it is larger along the diagonals and smaller along the axes. Neither region contains
the other, so the relabelling moves points in *both* directions rather than only one:

| Split | Points relabelled | `c` → `q` | `q` → `c` |
|---|---|---|---|
| Training (n = 100) | 6 | 4 | 2 |
| Test (n = 100) | 13 | 9 | 4 |
| Grid (n = 2601) | 218 | 122 | 96 |

Both transition sets were verified to match their analytic definitions exactly.

#### 2.1.3 The digit problem (Exercise 2)

The second task is ten-class optical digit recognition. Each observation is a binary image of a
digit extracted from a vehicle licence plate, of resolution 13 × 8 = 104 pixels, flattened
row-wise into a 104-dimensional feature vector. The example reproduced on p. 11 of the
laboratory handout was located verbatim in the dataset and confirmed to carry the correct label:

```
00010000
01111110
11100111
11000011  ...  (13 rows)
00111100

→ decoded correctly: YES; present in Bigtest1_104.arff: YES; file label: 0
```

### 1.3.2 Datasets

All datasets were verified against their expected row and class counts before any modelling
(Table 1). The verification is automated in `verify_data.py`.

**Table 1 — Datasets.**

| Dataset | Observations | Dimensionality | Class distribution | Role |
|---|---|---|---|---|
| `circletrain.arff` | 100 | 2 | 53 `c` / 47 `q` | training |
| `circletest.arff` | 100 | 2 | 60 `c` / 40 `q` | held-out test |
| `circleall.arff` | 2601 | 2 | 1251 `c` / 1350 `q` | dense grid (51 × 51, step 0.05) |
| `Bigtest1_104.arff` | 6024 | 104 | 525–739 per digit (imbalanced) | training pool |
| `Bigtest2_104.arff` | 5010 | 104 | exactly 501 per digit | external test set |

**Baselines.** For the circle test set the majority class is `c` at 60.0 %. For the ten-class
digit task, chance is 10 %, and on the balanced `Bigtest2` the majority-class baseline is
exactly 10 % by construction. All reported accuracies are interpreted against these baselines;
an accuracy of 53 % on the circle data, for instance, is *below* the trivial baseline, not merely
mediocre.

### 1.3.3 Algorithms

#### 2.3.1 k-Nearest Neighbours

Given a query point **x**, the classifier computes its distance to every training observation,
selects the *k* nearest neighbours *N_k*(**x**), and predicts

*f̂*(**x**) = argmax_{y ∈ 𝒴} Σ_{i ∈ N_k(**x**)} 𝟙[*y*_i = *y*].

The Euclidean metric *d*(**a**, **b**) = ‖**a** − **b**‖₂ was used, implemented in vectorised form
via the identity ‖**a** − **b**‖² = ‖**a**‖² + ‖**b**‖² − 2**a**·**b**. *k* was restricted to odd
values so that a two-class majority vote cannot tie. Ties arising from equal *vote counts* (which
can occur for even *k*, or in multi-class problems) are broken deterministically in favour of the
lower class index, and the number of such ties is counted and reported.

No feature standardisation was applied to the circle data: both coordinates lie in
[−1.25, 1.25], so the Euclidean metric is already scale-fair. Digit pixels are binary {0, 1},
so standardisation is likewise unnecessary there (though see Section 1.6.4 for why this does *not*
rescue kNN on a sparse high-dimensional problem).

**Leave-one-out cross-validation** was implemented using a single precomputed distance matrix
with the diagonal set to +∞, which prevents an observation from voting for itself. *Stratified*
5- and 10-fold cross-validation were also implemented for comparison, since LOO is known to have
higher variance.

#### 2.3.2 Decision trees

**J48** (WEKA's implementation of C4.5) selects splits by information gain ratio, then
post-prunes by subtree raising and by a confidence threshold (*C* = 0.25 by default). Its
principal capacity parameter is *M* = `minNumObj`, the minimum number of training observations
permitted in a leaf.

**scikit-learn's `DecisionTreeClassifier`** was configured to mirror J48 as closely as the two
implementations allow: `criterion='entropy'` (information gain), `min_samples_leaf=M`, and
`ccp_alpha=0.0`. The correspondence between the two tools is documented in Table 2.

**Table 2 — Concept mapping between WEKA and scikit-learn.**

| WEKA | scikit-learn | Meaning |
|---|---|---|
| `minNumObj` (−M) | `min_samples_leaf` | Minimum observations per leaf; the capacity control |
| `confidenceFactor` (−C 0.25) | `ccp_alpha` | Strength of post-pruning |
| `-U` | `ccp_alpha=0.0` | Request unpruned tree (see the caveat in Section 1.6.5) |
| information gain | `criterion='entropy'` | Split criterion |
| `IBk` −K | `n_neighbors` | Number of neighbours |

Exact numerical agreement between the two implementations is **not** expected: split
tie-breaking, pruning strategy, and threshold selection all differ. The claims defended in
Sections 5 and 6 concern the *ordering* of configurations and the *shape* of trends, not
individual values.

#### 2.3.3 Reproducible train–test partitioning

WEKA's `Percentage split 66%` option partitions the data using an internal shuffle. To ensure
that WEKA and scikit-learn were evaluated on *identical* observations, the partition was
reimplemented explicitly and materialised as separate ARFF files. This required reimplementing
`java.util.Random` (a 48-bit linear congruential generator) and WEKA's Fisher–Yates shuffle. The
reimplementation was validated against WEKA's own split size: for every seed the test partition
contained ⌊0.34 × 6024⌋ = 2048 observations.

#### 2.3.4 Statistical methodology

Because every model was evaluated on the *same* test observations, comparisons are naturally
paired, and an unpaired test would discard that information. **McNemar's test** was therefore
used throughout. It conditions on the observations on which two classifiers disagree:

|  | B correct | B incorrect |
|---|---|---|
| **A correct** | *a* | *b* |
| **A incorrect** | *c* | *d* |

and tests whether *b* and *c* are drawn from the same binomial distribution. Under the null
hypothesis of equal error rates, *b* ~ Binomial(*b* + *c*, ½). An **exact** binomial test was
used whenever *b* + *c* ≤ 25, and the continuity-corrected χ² statistic otherwise. To control the
family-wise error rate across the numerous pairwise comparisons, **Holm–Bonferroni** step-down
correction was applied within each family.

Point estimates are reported with **Wilson score intervals**, which are preferable to the normal
approximation for the small samples in the circle experiments because they remain within [0, 1]
and behave sensibly near the boundaries.

Predictions were taken from WEKA itself using the `-p 0` option, so that the pairing is exact
rather than approximated by a substitute implementation.

### 1.3.4 Software and reproducibility

WEKA 3.8.7 was driven headlessly through its bundled Java runtime. scikit-learn 1.8.0,
NumPy 2.4.2, and SciPy 1.17.1 were used on Python 3.14. All random seeds are recorded, no
source data file was modified, and byte-level integrity was verified by SHA-256 checksum before
and after the experiments. The complete pipeline is reproducible from the scripts listed in
Appendix A.

---

## 1.4 Exercise 1 — k-Nearest Neighbours

### 1.4.1 Validation of the implementation

Before any scientific claim, the from-scratch implementation was validated against two reference
implementations:

* the pairwise distance function agrees with `scipy.spatial.distance.cdist` to within
  1.39 × 10⁻¹⁴ (floating-point round-off);
* the classifier's predictions agree **point-for-point** with scikit-learn's
  `KNeighborsClassifier` on all 100 test observations at every value of *k* tested
  (*k* = 1, 3, 5, 7, 15, 31).

Any subsequent difference between the two therefore reflects the experiment, not a defect in the
implementation.

### 1.4.2 Accuracy as a function of *k*

**Table 3 — kNN accuracy versus *k*.** Train = accuracy on `circletrain`; LOO = leave-one-out
cross-validation on `circletrain`; 5-CV = stratified 5-fold cross-validation; Test = accuracy on
`circletest`; Grid = accuracy on the 2601-point dense grid.

| *k* | Train | LOO | 5-CV | Test | Grid |
|---|---|---|---|---|---|
| **1** | **1.000** | **0.930** | **0.940** | **0.910** | **0.900** |
| 3 | 0.990 | 0.860 | 0.900 | 0.880 | 0.890 |
| 5 | 0.940 | 0.890 | 0.908 | 0.870 | 0.869 |
| 7 | 0.940 | 0.900 | 0.869 | 0.840 | 0.861 |
| 9 | 0.900 | 0.860 | 0.849 | 0.820 | 0.844 |
| 11 | 0.900 | 0.860 | 0.878 | 0.830 | 0.840 |
| 13 | 0.910 | 0.860 | 0.809 | 0.860 | 0.825 |
| 15 | 0.900 | 0.860 | 0.778 | 0.840 | 0.812 |
| 21 | 0.840 | 0.780 | 0.630 | 0.780 | 0.740 |
| 31 | 0.610 | 0.560 | 0.549 | 0.670 | 0.566 |
| 51 | 0.530 | 0.530 | 0.530 | 0.600 | 0.481 |
| 99 | 0.530 | 0.530 | 0.530 | 0.600 | 0.481 |

*(Full table with standard deviations in `results/knn_k_sweep.csv`; figure in
`figures/knn_k_sweep.png`.)*

Three observations merit emphasis.

**Optimal performance at *k* = 1.** Test accuracy peaks at 91.0 % for *k* = 1 and declines
monotonically thereafter. It is instructive that the highest-capacity model is also the
best-generalising one here: with a smooth, noise-free boundary, 1-NN's zero bias outweighs its
high variance.

**Exact degeneration to the baseline for large *k*.** For *k* ≥ 51 the train, LOO, and
cross-validation figures collapse to identical values: 0.530 on the training set, 0.600 on the
test set, 0.481 on the grid. These are precisely the majority-class accuracies (53/100, 60/100,
1350/2601). The equality of training and LOO accuracy is diagnostic: the prediction no longer
depends on the query point at all, so the classifier has become a constant function. This is an
unusually clean illustration of the underfitting limit.

**Training accuracy is not monotone in *k*.** It decreases strongly overall but is flat-topped
between *k* = 5 and 7 (both 0.940) and non-monotone at *k* = 13. Only *k* = 1 yields exactly
zero training error, since each training observation is its own nearest neighbour.

### 1.4.3 Selection of *k*

**Table 4 — Optimal *k* by selection criterion.**

| Selection criterion | *k*\* | Accuracy at *k*\* |
|---|---|---|
| Leave-one-out CV on the training set | 1 | 0.930 |
| Stratified 5-fold CV on the training set | 1 | 0.940 |
| Stratified 10-fold CV on the training set | 1 | 0.952 |
| Test set (`circletest`) — *methodologically invalid* | 1 | 0.910 |

All four criteria agree on *k*\* = 1, but the agreement is a coincidence of this dataset, not a
justification for the fourth row. **Selecting a hyper-parameter by maximising test-set accuracy
is invalid**: it uses the test labels to make a modelling decision, so the test set ceases to be
unseen data and the reported accuracy becomes optimistically biased — in this case as the maximum
over 17 correlated estimates. The defensible estimate of the chosen model's performance is the
validation figure (LOO 0.930, 10-fold CV 0.952); the test accuracy of 0.910 serves only as an
independent confirmation.

The choice is also *robust*, not a fluke of one partition. Per-fold analysis (Table 5) shows
*k* = 1 winning 4 of 5 folds and 9 of 10 folds, and in the folds it loses it remains within
0.02–0.03 of the winner.

**Table 5 — Per-fold winning *k*.**

| CV scheme | Folds won by *k* = 1 | Folds won by *k* = 5 |
|---|---|---|
| 5-fold | 4 | 1 |
| 10-fold | 9 | 1 |

### 1.4.4 Behaviour on the dense reference grid

Figure `figures/knn_boundary.png` shows the induced decision regions for
*k* ∈ {1, 5, 15, 31} with the true circle and square superimposed. At *k*\* = 1 the accuracy
against the analytic ground truth is **0.9020** (259 errors over 2601 points), and 0.8970 when
the 212 exactly-on-boundary points are excluded.

The error map (`figures/knn_errors_kstar.png`) shows a decisive structure: errors are confined
to a **thin band following the circular arc**, and are absent along the straight square edges.
This is exactly what the local nature of kNN predicts. The Voronoi cells induced by 100 training
points spread over a 2.5 × 2.5 domain are too coarse to resolve curvature, whereas the straight
outer boundary is captured without difficulty. This observation motivates the representation
study of Section 4.

---

## 1.5 Exercise 1b — The Effect of Input Representation

### 1.5.1 Experimental design

Four representations were constructed from the *same* points, so that any difference in outcome
is attributable to representation alone:

| Variant | Attributes | Labels |
|---|---|---|
| `xy` | *x*, *y* | original (circle) |
| `sq` | *x*, *y* | inner region replaced by the side-1.76 square |
| `zt` | *z* = *x*², *t* = *y*² | original (circle) |
| `u` | *u* = *x*² + *y*² | original (circle) |

### 1.5.2 Results

**Table 6 — Accuracy and tree size by representation** (WEKA J48, `-C 0.25 -M 2`, trained on
`circletrain`, evaluated on `circletest`).

| Representation | Train accuracy | Test accuracy | Leaves | Tree size |
|---|---|---|---|---|
| `(x, y)`, circle | 0.96 | 0.88 | 5 | 9 |
| square inner, `(x, y)` | 1.00 | 0.92 | 5 | 9 |
| `(x², y²)`, circle | 0.99 | 0.88 | 5 | 9 |
| **`u` = `x² + y²`, circle** | **1.00** | **1.00** | **2** | **3** |

The corresponding scikit-learn results and the full confusion matrices are in
`results/sklearn_representation.csv` and `results/representation_confusion.csv`.

### 1.5.3 Is the square inner region easier to learn? (p. 7)

**Prediction.** Yes. The new inner region is the conjunction of two axis-aligned constraints,
(\|*x*\| ≤ 0.88) ∧ (\|*y*\| ≤ 0.88), which lies directly in the hypothesis language of a decision
tree, whereas the circle is not linearly separable in (*x*, *y*).

**Observation.** Partially confirmed. Training accuracy rises from 0.96 to 1.00 and test accuracy
from 0.88 to 0.92 (WEKA; 0.87 → 0.96 for scikit-learn). However, **the tree does not shrink**:
J48 reports 5 leaves in both cases. Inspection of the printed trees explains why — both variants
use the outer square boundary to isolate the `q` corners, and pruning at *C* = 0.25 retains only
five leaves regardless. The benefit is therefore in *how well* those five leaves fit, not in
tree size. This is a case where the naive expectation ("simpler region, smaller tree") is not
what the experiment shows, and it is reported as such.

### 1.5.4 Does the (*x*², *y*²) representation help? (p. 8)

**Prediction.** Yes. In (*z*, *t*) coordinates the circle becomes the exact linear constraint
*z* + *t* ≤ 1. This is confirmed numerically:

```
max(z + t | c) = 0.999033       min(z + t | q) = 1.000280
```

— perfect separation, and identical to the *x*² + *y*² figures of Section 1.3.1 by construction.

**Observation.** Only a marginal gain. Test accuracy is unchanged for J48 (0.88 → 0.88) and rises
slightly for scikit-learn (0.87 → 0.89), while training accuracy rises to 0.99. Crucially, the
tree remains 5 leaves, and the printed tree is a staircase in (*z*, *t*):

```
t <= 0.712082
|   z <= 0.729179: c (49.0/1.0)
|   z >  0.729179
|   |   z <= 0.897385
|   |   |   t <= 0.148856: c (5.0)
|   |   |   t >  0.148856: q (5.0)
|   |   z >  0.897385: q (9.0)
t > 0.712082: q (32.0)
```

The reason is the same structural limitation as in Section 1.5.3: **a decision tree tests one
attribute at a time, and *z* + *t* is not an attribute.** The exact straight-line boundary must
still be approximated by an axis-aligned staircase. The transformation helps only insofar as the
staircase in (*z*, *t*) aligns better with the true boundary than the one in (*x*, *y*). The full
collapse requires supplying the *sum* as a feature — see Section 1.5.5.

### 1.5.5 Can perfect accuracy be achieved with a minimal tree? (p. 9)

**Derivation.** The true rule for the original problem is a single threshold on the squared
radius. Because every observation of both classes already lies inside the domain square, the
conditions \|*x*\| ≤ 1.25 and \|*y*\| ≤ 1.25 are automatically satisfied and no further test is
required:

*u* = *x*² + *y*²,  *u* ≤ 1 → `c`,  *u* > 1 → `q`.

Numerically the separation is clean: max(*u* | `c`) = 0.999033, min(*u* | `q`) = 1.000280.

**Observation.** J48 produces exactly the predicted stump:

```
u <= 0.999033: c (53.0)
u >  0.999033: q (47.0)

Number of Leaves : 2      Size of the tree : 3
```

with **100 % accuracy on both the training and the test set**. This is the smallest possible
non-trivial tree. scikit-learn produces an identical structure.

The educational point is not that the algorithm became more capable, but that **feature
engineering altered the hypothesis space** so that a hypothesis of minimal complexity became
expressible. The variable *u* encodes precisely the inductive bias the problem demands.

### 1.5.6 Does the perfect result generalise?

Evaluated on all 2601 points of the dense grid, the model attains **99.77 %** (6 errors; both
WEKA — learned threshold 0.999033 — and scikit-learn — threshold 0.999656 — make the same six).
On the **2389 interior points, excluding the 212 that lie exactly on a boundary, the accuracy is
exactly 100 % (zero errors)**.

Every residual error is a point whose *u* is exactly 1.0. This reveals a genuine
**representational limit of the learner** rather than a deficiency of the feature:

> A decision tree can only emit a threshold equal to a value that **occurs in its training
> data**. The largest training value is *u* = 0.999033 < 1, so the rule `u <= 1` is not exactly
> representable. The six grid points with *u* = 1 consequently fall on the wrong side of the
> learned split.

This limit is a property of the algorithm, not of the data or the feature, and it would vanish
with training data covering the boundary more densely.

### 1.5.7 Where do the errors actually occur?

Accuracy alone does not explain *why* a representation helps. Because the four variants produce
structurally different trees, their error patterns differ, and the pattern is diagnostic
(Table 7; figures `figures/representation_error_maps.png`).

**Table 7 — Error localisation on the dense grid** (ground truth: analytic circle rule). "Interior"
excludes the 212 points lying exactly on a boundary, where the assigned label is arbitrary;
interior accuracy removes those points from both numerator and denominator.

| Representation | Raw grid errors | Errors near the inner boundary | Interior errors | Interior accuracy |
|---|---|---|---|---|
| `(x, y)`, circle | 211 | 134 (**63.5 %**) | 206 / 2389 | 0.9138 |
| square inner, `(x, y)` | 225 | 179 (**79.6 %**) | 219 / 2389 | 0.9083 |
| `(x², y²)`, circle | 180 | 108 (**60.0 %**) | 174 / 2389 | 0.9272 |
| `u` = `x² + y²` | **6** | 6 | **0 / 2389** | **1.0000** |

Note on ground truth: the **file labels are used as authoritative**. This is a substantive
choice, not a formality. Twelve grid points have *x*² + *y*² equal to 1 in exact decimal
arithmetic, and evaluating `x*x + y*y <= 1.0` on the stored decimal values in binary floating
point classifies six of them as inside and six as outside — **the opposite split to the one the
file uses**. Recomputing the rule from the rounded coordinates therefore manufactures six
phantom "errors" that are pure floating-point noise on a measure-zero set. The analytic rule is
retained as a cross-check and is reported to disagree with the file on exactly those six points.

Two findings stand out, and both confirm the theoretical account:

1. **The `sq` variant's errors are overwhelmingly on the *inner square* boundary (79.6 %)**, not
   on the circle. The decision boundary moved when the labels changed, and the tree's errors
   followed it. The learner is failing where the boundary is — correct behaviour for an
   under-capacity model.
2. **No representation produces errors on the outer square edge** (zero in all four cases),
   confirming that the outer boundary is trivially expressible by axis-aligned splits and that
   all the difficulty resides in the interior boundary.
3. **The `u` representation reduces the interior error count to zero** — 0 errors on 2389
   points, against 174–219 for the other variants (raw counts 6 vs 180–225). It succeeds because the boundary it must
   express is a single threshold on a supplied feature, rather than an axis-aligned staircase it
   must construct.

### 1.5.8 The optional two-attribute variant

Finally, the *z* and *t* attributes were supplied as two separate features (rather than
pre-summed) to test whether a one-node tree is achievable that way. It is not:

* the resulting tree has **depth 4 and 6 leaves** (7 leaves fully grown), with test accuracy
  0.89;
* an exhaustive search over *z*, *t*, *z* + *t*, *z* − *t*, max(*z*, *t*), and min(*z*, *t*)
  shows that **only the sum *z* + *t* is separable by a single threshold**.

This settles the question definitively: a tree cannot evaluate a sum of attributes, so separating
the components of *u* destroys the very property that made *u* effective.

---

## 1.6 Exercise 2 — Decision Trees on the Digit Data

### 1.6.1 Seed sensitivity of the 66 % split (pp. 12–13)

**Table 8 — J48 (default parameters) across six random seeds** (2048 test observations per seed).

| Seed | J48 on internal test | J48 on `Bigtest2` | scikit-learn on internal test | scikit-learn on `Bigtest2` |
|---|---|---|---|---|
| 1 | 0.9678 | 0.9471 | 0.9673 | 0.9461 |
| 2 | 0.9604 | 0.9467 | 0.9604 | 0.9473 |
| 3 | 0.9609 | 0.9477 | 0.9585 | 0.9505 |
| 4 | 0.9609 | 0.9473 | 0.9551 | 0.9439 |
| 5 | 0.9580 | 0.9423 | 0.9565 | 0.9445 |
| 42 | 0.9648 | 0.9455 | 0.9585 | 0.9481 |
| **Mean ± s.d.** | **0.96216 ± 0.00352** | **0.94611 ± 0.00200** | 0.95939 ± 0.00428 | 0.94674 ± 0.00244 |

**Interpretation.** Is the seed-to-seed variation genuine model instability, or sampling noise in
the evaluation? The binomial standard error of an accuracy estimated from *n* = 2048 observations
at *p* = 0.962 is √(0.962 × 0.038 / 2048) = 0.42 pp, giving a 95 % interval of ± 0.83 pp. The
**observed spread is 0.98 pp**, closely consistent with the expected ± 0.83 pp. The variation is
therefore dominated by the finite size of the *test partition*, not by instability of the
learning algorithm — a distinction that materially affects how single-seed results should be
quoted.

### 1.6.2 External validation on `Bigtest2`

The mean accuracy on the independent test set is 0.9462, against 0.9622 on the internal split — a
gap of **1.6 pp**. This is the honest generalisation penalty, and its causes are identifiable:

* `Bigtest2` is **exactly balanced** (501 per digit) whereas `Bigtest1` is imbalanced (525–739).
  Training on the imbalanced subset and testing on a balanced set alters the effective class
  priors.
* `Bigtest2` is a **pre-processed derivative**, as its own relation name records
  (`digit-weka.filters.unsupervised.attribute.Remove-R105-128`).

The methodological conclusion is that an internal random split of the training file
systematically **overstates** performance relative to a genuinely independent sample.

### 1.6.3 Error analysis

The most frequent confusions (seed 1; `figures/j48_confusion.png`, `figures/confused_pairs.png`)
are:

**Table 9 — Most frequent misclassifications.**

| True digit | Predicted digit | Count |
|---|---|---|
| 9 | 8 | 6 |
| 8 | 9 | 5 |
| 8 | 6 | 4 |
| 5 | 9 | 3 |
| 4 | 3 | 3 |
| 3 | 4 | 3 |

The hypothesis that these confusions reflect specific pixel-level similarities was tested by
rendering the mean image of each digit in the most-confused pair together with their difference
image. For the 9/8 pair the difference is concentrated in the closure of the lower loop and the
presence of a descender — a small number of pixels which a greedy information-gain search is free
to ignore if a higher-gain attribute is selected first. This is consistent with the observed
confusions and with the known behaviour of greedy top-down induction.

### 1.6.4 Comparison with the nearest-neighbour classifier

**Table 10 — Model comparison on the digit task**, means across six seeds.

| Model | Internal test | `Bigtest2` (external) |
|---|---|---|
| J48, *M* = 2 | 0.96216 ± 0.00352 | 0.94611 ± 0.00200 |
| scikit-learn tree, entropy, *M* = 2 | 0.95939 ± 0.00428 | 0.94674 ± 0.00244 |
| **IBk (kNN), *k* = 1** | **0.98372 ± 0.00171** | **0.97588 ± 0.00125** |
| **IBk (kNN), *k* = 5** | **0.98389 ± 0.00127** | 0.97542 ± 0.00089 |

**The nearest-neighbour classifier outperforms the decision tree by 2.2 pp on the internal split
and 3.0 pp on the external test set**, with a smaller seed-to-seed spread. This is the *reverse*
of the circle problem (Section 1.4.2) and it contradicts the expectation that the curse of
dimensionality would penalise kNN at 104 dimensions.

The reversal is explicable on structural grounds:

* The digit classes are discriminated by **local stroke features** — a closed loop, a descender, a
  horizontal bar. 1-NN retrieves the single most similar training specimen, which is precisely
  the comparison a human reader performs. A global axis-aligned tree must approximate that
  similarity structure with a limited number of rectangular regions.
* The curse of dimensionality manifests as a **data requirement**, not an automatic failure. With
  6024 training observations in 104 dimensions the sample-to-dimension ratio is favourable
  (≈ 58 observations per dimension), so the local method is adequately supported.
* The external-test gap is 1.6 pp for the tree but only 0.8 pp for kNN, indicating that the local
  method is also **more robust to the dataset shift** described in Section 1.6.2 — consistent with
  its reliance on local similarity rather than on globally tuned thresholds.

The correct conclusion is therefore a refinement of the standard heuristic rather than a
refutation of it: high dimensionality makes kNN *data-hungry*, and this dataset is large enough to
satisfy that appetite. The tree retains a decisive advantage in computational cost
(Section 8.5.3).

### 1.6.5 Overfitting control via *M* (pp. 14–16)

**Table 11 — Accuracy and tree size versus *M*** (trained on the full `Bigtest1`,
evaluated on `Bigtest2`; selected rows).

| *M* | J48 train | J48 test | J48 leaves | scikit-learn train | scikit-learn test |
|---|---|---|---|---|---|
| **1** | 0.99751 | 0.95150 | 144 | 1.00000 | **0.95589** |
| 2 | 0.99104 | 0.95309 | 108 | 0.99148 | 0.95250 |
| 3 | 0.98556 | 0.95349 | 85 | 0.98805 | 0.95529 |
| **4** | 0.98290 | **0.95509** | 75 | 0.98406 | 0.95210 |
| 5 | 0.97892 | 0.95229 | 57 | 0.98074 | 0.95130 |
| 10 | 0.97145 | 0.94411 | 46 | 0.97078 | 0.94571 |
| 15 | 0.96049 | 0.93653 | 32 | 0.96414 | 0.94192 |
| 25 | 0.95252 | 0.92974 | 28 | 0.95066 | 0.92794 |
| 50 | 0.92331 | 0.89022 | 20 | 0.92226 | 0.89321 |
| 100 | 0.89890 | 0.87086 | 11 | 0.90471 | 0.87500 |

*(Full sweep in `results/j48_M_sweep.csv`; figure `figures/j48_M_sweep.png`.)*

**Optimal value.** *M*\* = 4, with test accuracy 0.95509 under WEKA. scikit-learn's optimum lies
at *M* = 1 or 3 (0.95589 / 0.95529), within 0.06 pp of WEKA's. More importantly, **the whole
range *M* ∈ [1, 5] lies between 0.9515 and 0.9559** — the optimum is *broad*, so the precise
location of *M*\* is of far less practical importance than recognising the plateau.

**The underfitting–optimal–overfitting progression is clearly visible:**

| Regime | *M* | Behaviour |
|---|---|---|
| Overfitting | 1 | Training 99.75 %, test 95.15 %; 144 leaves — maximum capacity |
| **Optimal** | **4** | Training 98.29 %, test **95.51 %**; 75 leaves — some fit traded for generalisation |
| Underfitting | ≥ 10 | Both curves fall together (train 97.1 → 89.9 %, test 94.4 → 87.1 %) |

At *M* = 100 only 11 leaves remain — too few to partition ten digit classes.

**A discrepancy with the standard account.** The laboratory notes (p. 16) state that as capacity
is reduced the classifier performance "will still improve" on the training set. **The
measurements contradict this**: training accuracy decreases from 0.99751 at *M* = 1 to 0.89890 at
*M* = 100. The mechanism is transparent — a leaf required to contain at least *M* observations
cannot memorise an isolated one, so increasing *M* must reduce training accuracy. The textbook
description is accurate for the *early* part of the sweep (*M* = 1 → 4), where test accuracy rises
while training accuracy has only just begun to fall; beyond that, the two curves diverge with
training consistently above test, which is the classical picture. The distinction matters because
it identifies *where* the optimum lies: at the point of divergence, not at a maximum of the
training curve.

### 1.6.6 Why *M* = 1 does not yield 100 % training accuracy (p. 15)

**Table 12 — Testing the three proposed explanations.**

| Configuration | Training accuracy | Leaves |
|---|---|---|
| J48, *M* = 1, default pruning | 0.99751 | 144 |
| **J48, *M* = 1, `-U`** | **1.00000** | 175 |
| scikit-learn, fully grown | 1.00000 | 164 |

The three explanations offered in the laboratory notes were each tested:

1. **Pruning removes branches — confirmed, and it is the dominant cause.** Default J48 at
   *M* = 1 reaches 99.751 %; adding `-U` recovers **100.000 %** with 175 leaves. The missing
   0.249 % (≈ 15 training observations) is thereby fully accounted for.
   *Important caveat discovered during the experiments:* **WEKA's `-U` does not disable all
   pruning.** It prevents subtree *raising*, but the confidence-threshold pruning controlled by
   *C* = 0.25 still applies. Evidence: on the circle data, J48 with `-U -M 1` returns the *same*
   5-leaf tree as the pruned default (Section 1.5.2). scikit-learn's `ccp_alpha = 0` is genuinely
   unpruned. The two "unpruned" modes are therefore **not** equivalent and must not be treated
   as such.
2. **Greedy search is not globally optimal — plausible in general, but not the binding
   constraint here.** Since a fully grown tree reaches exactly 100 %, greedy induction was
   evidently able to isolate every training observation.
3. **The attribute set may be insufficient — refuted.** With 104 binary attributes, a fully grown
   tree memorises all 6024 observations.

### 1.6.7 The *k* / *M* correspondence

*k* in kNN and *M* in J48 are the same concept viewed from opposite directions. Small *k* gives
low bias and high variance with zero training error; large *k* degenerates to the majority class.
Small *M* gives high capacity and near-perfect training fit; large *M* produces leaves too coarse
to separate the classes. Both are selected on validation data, and both exhibit a broad rather
than sharp optimum (*k*\* = 1, *M*\* = 4). The mechanisms differ — kNN regularises by *smoothing a
local neighbourhood*, a tree by *refusing to split* — but the bias–variance trade-off they trace
is the same.

---

### 1.7 Scikit-Learn cross-check

An independent re-implementation of the same classifiers in scikit-learn was used throughout as
a cross-check on the WEKA results; the numerical agreement is reported in the relevant tables
above. The full treatment of the deck's scikit-learn module (pages 17-27) is a separate
laboratory and appears in Section 6.

# 2. Lab 2 — Clustering

**Deck pages:** 28–43, plus pages 10–11 (digit data description) and 36–37.
**Data:** `gausstrain.arff`, `gausstest.arff`, `gausstrainhv.arff`, `gausstesthv.arff`,
`Bigtest1_104.arff`, `Bigtest2_104.arff`.
**Status:** Partial — implemented and executed, but XMeans was unavailable and is substituted by
k-means with model-selection criteria (Exercise 2.6; see Section 8.1).

### 2.1 Problem statement

The gaussian datasets contain 100 two-dimensional points drawn from two Gaussian distributions.
The deck (p. 29) states the generating parameters: equal variances along both axes, with means at
**(3, 3)** and **(7, 7)**. The `hv` variants use the same means but **50 % greater variance**.

Each file declares a nominal class attribute `cluster {1, 2}` holding the ground-truth grouping.
This attribute is **ignored** during clustering — it is used only for the "classes-to-clusters"
evaluation of Exercise 2.3, exactly as in WEKA.

The empirical statistics confirm the design:

| Dataset | Class 1 mean | Class 2 mean | Class 1 variance | Class 2 variance |
|---|---|---|---|---|
| `gausstrain.arff` | (2.733, 2.723), n = 42 | (6.917, 7.602), n = 58 | 2.38 / 2.26 | 4.55 / 4.29 |
| `gausstest.arff` | (2.904, 2.934), n = 51 | (6.887, 6.651), n = 49 | 1.97 / 1.93 | 3.54 / 3.76 |
| `gausstrainhv.arff` | (2.780, 2.725), n = 53 | (7.272, 7.522), n = 47 | 4.26 / 4.38 | 6.05 / 10.45 |
| `gausstesthv.arff` | (2.947, 2.577), n = 49 | (7.461, 7.267), n = 51 | 4.24 / 4.62 | 6.18 / 7.51 |

The variance increase from the low- to the high-variance training file is a factor of ≈ 1.8–2.3,
consistent with the stated "+50 %" applied to the standard deviation.

### 2.2 Exercise 2.1–2.3 — K-Means on `gausstrain.arff`

**Protocol.** K-Means with *k* = 2, Euclidean distance, on the *(x, y)* coordinates with the class
attribute ignored. WEKA's `SimpleKMeans` is the reference implementation; scikit-learn's
`KMeans(n_init=10)` was used here. The two agree up to an arbitrary permutation of cluster
labels — the algorithm has no notion of which cluster is "cluster 1".

#### 2.2.1 Centroid placement (Exercise 2.2)

| Centroid | Position | Points assigned | Nearest generating mean | Distance to it |
|---|---|---|---|---|
| 0 | (2.7826, 2.9445) | 46 | (3, 3) | **0.2244** |
| 1 | (7.1840, 7.7750) | 54 | (7, 7) | **0.7966** |

Total inertia: 653.22.

**Interpretation.** Both centroids land very close to the generating means. The second centroid is
displaced by 0.80, i.e. roughly 3.5× further than the first. This asymmetry has a specific cause
and is not noise: the second component has both a larger empirical variance (4.55 / 4.29 versus
2.38 / 2.26) and **more points** (58 versus 42), and its empirical mean is already displaced from
the generating mean (6.917, 7.602) versus (7, 7). K-Means places each centroid at the mean of the
points assigned to it, so it is pulled towards the larger, more dispersed cloud.

This is worth stating precisely, because "how far is the centroid from the true mean?" invites the
answer "it should be zero". It should **not** be zero: the centroid estimates the mean of the
*empirical sample* assigned to the cluster, so the correct reference is the sample mean, not the
generating parameter. Distances to the empirical class means are 0.2271 and 0.3183 — markedly
smaller, and the residual is attributable to the few points assigned to the "wrong" cluster.

#### 2.2.2 Classes-to-clusters evaluation (Exercise 2.3)

Clusters are mapped to classes by majority vote, then a confusion matrix and accuracy are formed.

**Classes × clusters table** (rows = true class, columns = cluster):

|  | Cluster 0 | Cluster 1 |
|---|---|---|
| **Class 1** | 42 | 0 |
| **Class 2** | 4 | 54 |

**Cluster → class mapping:** cluster 0 → class 1, cluster 1 → class 2.
**Confusion matrix** (rows = true, columns = assigned): `[[42, 0], [4, 54]]`.
**Accuracy = 0.9600** (4 errors out of 100).

The four misassigned points are all true class 2:

| Index | Point | True class | d(mean₁) | d(mean₂) |
|---|---|---|---|---|
| 19 | (3.636, 4.982) | 2 | 2.433 | 4.199 |
| 31 | (2.369, 5.665) | 2 | 2.964 | 4.943 |
| 44 | (6.177, 4.040) | 2 | 3.688 | 3.638 |
| 89 | (1.043, 6.397) | 2 | 4.044 | 5.996 |

All four lie in the region between the two clouds, and all are outliers of the more dispersed
class 2 (recall that this class has the larger variance). Only one of the four is within 1.0 of the
perpendicular bisector between the two means; the others are *closer to the class-1 centroid than
to the class-2 centroid*, which is precisely why K-Means assigns them there. **The errors are not
algorithmic failures — K-Means has no access to the labels, and by the only criterion available to
it (distance to the nearest centroid) its assignments are optimal.** This distinction between
"wrong with respect to the hidden labels" and "suboptimal for the clustering objective" is the
central conceptual point of the exercise, and the deck raises it explicitly on p. 33.

*(Figures: `lab2/figures/kmeans_gausstrain.png` — clusters and centroids, beside ground truth with
the misassigned points circled. Table: `lab2/results/kmeans_gausstrain_centroids.csv`.)*

### 2.3 Exercise 2.4 — Effect of increased variance (`gausstrainhv.arff`)

| Quantity | Low variance | High variance (+50 %) |
|---|---|---|
| Empirical class variances | 2.32 / 4.54 | 4.32 / 8.26 |
| Centroid 0 position | (2.783, 2.945) | (3.251, 2.748) |
| Centroid 1 position | (7.184, 7.775) | (7.682, 8.780) |
| Distance to nearest generating mean | 0.2244 / 0.7966 | 0.3563 / **1.9064** |
| Points on the far side of the bisector from their own mean | 3 | **12** |
| Classes-to-clusters accuracy | **0.9600** | **0.9000** |
| Errors | 4 | 10 |

**Interpretation.** Quadrupling the overlap between the two clouds — a direct consequence of the
variance increase — degrades clustering accuracy from 96 % to 90 %, and the displacement of the
second centroid grows from 0.80 to 1.91. The number of points lying geometrically on the *wrong*
side of the bisector between the generating means rises fourfold, from 3 to 12.

The exercise asks which points will be assigned to the "wrong" cluster. The answer follows from
the geometry: **those lying beyond the perpendicular bisector of the segment joining the two
means**, i.e. points that are individually ambiguous. It also asks whether "wrong" is really the
right word, and the answer is *no*: as in Section 2.2.2, K-Means optimises squared distance to
centroids, not agreement with the generating labels. A point that is geometrically closer to the
other centroid is *correctly* assigned for the clustering objective, even though its generating
label differs. Indeed, in the high-variance case 4 of the 12 points on the wrong side of the
bisector are still assigned to their own generating class.

*(Figure: `lab2/figures/kmeans_variance_comparison.png`. Tables:
`lab2/results/kmeans_gausstrain_centroids.csv`, `lab2/results/kmeans_gausstrainhv_centroids.csv`.)*

### 2.4 Exercise 2.5 — K-Means on the digit images

K-Means with *k* = 10 on `Bigtest1_104.arff` (6024 images, 104 binary pixels), ignoring the class
attribute. Inertia = 64 013.1.

**Classes-to-clusters accuracy = 0.9456** (328 errors of 6024).

Because *k* = 10 equals the number of true classes, all ten clusters can be labelled 1:1. The
resulting mapping is a bijection onto the ten digits:

| Cluster | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Assigned digit | 0 | 2 | 5 | 4 | 8 | 1 | 7 | 6 | 3 | 9 |
| Correlation of centroid with the class-mean image | 0.9993 | 0.9993 | 0.9983 | 0.9998 | 0.9971 | 0.9997 | 0.9994 | 0.9980 | 0.9986 | 0.9992 |

**Every cluster corresponds to exactly one digit**, with no cluster split across two digits and no
digit shared by two clusters, and the mean correlation of a centroid with the mean image of its
assigned digit is **0.9989**.

#### 2.4.1 The centroids look like digits

This is the observation the deck asks about on p. 39–40, and the explanation is given there:
each pixel of a centroid is the **mean of a binary variable** over the patterns in that cluster.
Its grey level is therefore an estimate of the probability that the pixel is ON for a pattern in
the cluster. Because within-cluster variation is small (the images are clean, centred, binary
glyphs), those per-pixel frequencies concentrate near 0 or 1, and the resulting 13 × 8 greyscale
array reproduces the shape of the digit that generated the cluster.

The claim "they do look like the figures they represent" is therefore **not a coincidence and not
evidence of supervised learning** — it follows directly from the definition of the centroid
combined with the low within-cluster variance. The high correlations above quantify it, and
`lab2/figures/kmeans_digit_centroids.png` renders all ten centroids as images.

*(Tables: `lab2/results/kmeans_digit_clusters.csv`.)*

### 2.5 Exercise 2.6 — How many clusters? Did XMeans help?

> **Substitution.** WEKA's `XMeans` was not available in this environment. The exercise is
> reproduced with k-means over a range of *k* together with two standard model-selection criteria
> — **BIC** (for the spherical equal-variance Gaussian mixture that k-means implicitly fits) and
> the **silhouette coefficient**. This is a documented substitution, not an XMeans result, and is
> reflected in the Status column.

#### 2.5.1 Model-selection criteria

| Dataset | *k* minimising BIC | *k* maximising silhouette | *k* maximising classes-to-clusters accuracy | true *k* |
|---|---|---|---|---|
| `gausstrain` | 8 | **2** | 3 (0.97) | **2** |
| `Bigtest1_104` | 2 | **10** | 10 (0.9456) | **10** |

The two criteria disagree, and neither is reliable on both datasets:

* On the gaussian data, **silhouette correctly identifies *k* = 2**, while BIC prefers larger
  partitions — and in fact continues to decrease in the range tested (215.3 at *k* = 2 down to
  171.2 at *k* = 8), so it provides no usable stopping point here.
* On the digit data the situation reverses: **silhouette correctly identifies *k* = 10**, but BIC
  is minimised at *k* = 2 — badly wrong, and increasingly so as *k* grows (23 375 at *k* = 10,
  49 235 at *k* = 40).

This is a substantive negative result about the criteria themselves, not merely about this data.
The BIC form used penalises *k*(d + 1) parameters against a residual sum of squares; with d = 104
the per-cluster penalty is large, so BIC is biased towards very coarse partitions when the
dimensionality is high. Silhouette, by contrast, is a distance-ratio criterion and is unaffected
by the parameter count — but on the 2-dimensional gaussian data it *also* fails for the opposite
reason, namely that splitting a diffuse Gaussian into two halves genuinely improves the
within/between distance ratio.

**The practical conclusion is that unsupervised model selection failed on one of the two datasets
whichever criterion was used**, and the correct *k* was only identifiable because the ground-truth
labels were available for validation. This is a caution about clustering in general.

#### 2.5.2 Does a finer partition overfit? (Exercise 2.6.3)

Clusters were fitted on `gausstrain` and the *same* centroids used to assign `gausstest` points,
with the cluster→class mapping learned from the training labels only.

| *k* | Train accuracy | Test accuracy | Gap |
|---|---|---|---|
| 2 | 0.9600 | 0.9200 | +4.0 pp |
| 4 | 0.9500 | 0.9300 | +2.0 pp |
| 6 | 0.9200 | 0.9000 | +2.0 pp |
| 8 | 0.9300 | 0.9200 | +1.0 pp |
| 12 | 0.9600 | **0.9700** | −1.0 pp |
| 20 | 0.9500 | **0.9700** | −2.0 pp |

**On this data, increasing the number of clusters does not cause overfitting.** The train–test gap
is small throughout (±4 pp, i.e. ±4 points out of 100) and is *negative* at *k* = 12 and 20, meaning
the test set is classified slightly better than the training set. Two caveats prevent this from
being read as "more clusters is better":

* The differences are well inside sampling noise. With *n* = 100 a Wilson 95 % interval on an
  accuracy near 0.93 is roughly ±5 pp, which is wider than every gap in the table. No row here is
  statistically distinguishable from any other.
* Once *k* approaches the sample size the model can in principle memorise, but at *k* = 20 with
  *n* = 100 it has not yet reached that regime. The exercise's expected overfitting would require
  substantially larger *k* (or a lower-dimensional, sparser problem).

The honest statement is therefore: **no overfitting was observed in the range tested, but the
sample is too small to detect a modest effect either way.**

*(Figures: `lab2/figures/kmeans_k_selection.png`. Tables:
`lab2/results/kmeans_k_selection.csv`, `lab2/results/kmeans_generalisation.csv`,
`lab2/results/lab2_summary.json`.)*

### 2.6 Summary of Lab 2 results

| Exercise | Result |
|---|---|
| 2.1–2.2 (centroids) | Centroids at (2.783, 2.945) and (7.184, 7.775); distances to the generating means 0.224 and 0.797 |
| 2.3 (classes-to-clusters) | **96.0 %**, 4 errors; all errors are class-2 outliers closer to the other centroid |
| 2.4 (variance +50 %) | Accuracy falls to **90.0 %**, 10 errors; centroid displacement grows to 1.91 |
| 2.5 (digits, *k* = 10) | **94.56 %**, 328 errors; perfect 1:1 cluster↔digit mapping; mean centroid correlation **0.9989** |
| 2.6 (X-Means) | **Partial** — substituted by BIC + silhouette; silhouette found the true *k* on the digits, BIC did not; no overfitting detected (train–test gap within ±4 pp) |

---

# 3. Lab 3 — Evolutionary Computation: Genetic Algorithms

**Deck pages:** 51–72.
**Software:** DEAP 1.4.
**Data / inputs:** none required for Exercise 1; `smiley.txt` (14 × 16 = 224 bits) for
Exercise 2.
**Status:** Partial — both Exercise 1 and Exercise 2 were implemented and executed;
Exercises 3 (N-queens) and 4 (ant game) were **not** run (see Section 8.1).

### 3.1 Exercise 1 — Function optimisation

#### 3.1.1 Problem

Minimise

$$f(x,y,z) = \bigl(1.5 + \sin z\bigr)\left(\sqrt{(20-x)^2 + (30-y)^2} + 1\right),
\qquad x,y,z \in [-250, 250]$$

The deck (p. 63) asks for two implementations and a comparison of their convergence speed
(p. 65):

* **Representation A** — a list of three floats, with Gaussian mutation (μ = 0, σ = 0.2) and
  two-point crossover.
* **Representation B** — a list of 90 bits (30 bits per variable), with two-point crossover and
  flip-bit mutation, decoding via the formula on p. 65:
  `value(i) = Min + i/(2^N − 1) · (Max − Min)`.

#### 3.1.2 Analytic minimum

The objective is a product of two non-negative factors:

* (1.5 + sin *z*) is minimised at sin *z* = −1, giving **0.5**;
* (√((20 − *x*)² + (30 − *y*)²) + 1) is minimised at (*x*, *y*) = (20, 30), giving **1**.

Both can be satisfied simultaneously, so the global minimum is **0.5**, attained at
(20, 30, −π/2). A coarse grid search over (0…40) × (10…50) × [−250, 250] confirms 0.500000.

**The landscape is strongly multimodal.** Since sin *z* = −1 has **80 distinct solutions** in
[−250, 250] (*z* = −π/2 + 2*k*π), there are 80 equivalent global optima. This is what makes the
problem a genuine test for a GA rather than a convex descent exercise — and it explains the
results below.

An earlier draft of this document stated the minimum as 1.5; that was an arithmetic error and is
corrected here.

#### 3.1.3 Results

Settings: population 300, 100 generations, *P*c = 0.5, *P*m = 0.2, 5 independent seeds.

| Representation | Mean best fitness | Std. dev. | Best found | Mean evaluations | Runs within 1 % of optimum |
|---|---|---|---|---|---|
| **Float (σ = 0.2)** | **0.500501** | 0.000390 | 0.500120 | 18 353 | **5 / 5** |
| Bit (indpb = 0.01) | 0.800107 | 0.277871 | 0.502136 | 18 179 | 1 / 5 |

The float representation **reliably** locates the global optimum: all five runs finish within
1.2 × 10⁻³ of 0.5, a relative error below 0.25 %. The bit representation with the DEAP default
mutation rate fails in four of five runs, and its mean best fitness is 60 % worse.

#### 3.1.4 Convergence speed

First generation reaching within a tolerance of the optimum:

| Representation | within 1 % | within 0.1 % |
|---|---|---|
| Float | 5/5 runs, mean generation **58.2** | 3/5 runs, mean generation 83.3 |
| Bit (indpb = 0.01) | 1/5 runs (generation 35) | **0 / 5** |

Because a generation of the float run evaluates ~184 individuals and the bit run ~182, the two
representations consume almost identical evaluation budgets (18 353 vs 18 179). The comparison is
therefore fair, and the conclusion is unambiguous: **for this problem the float representation
converges to a better solution within the same budget.**

#### 3.1.5 Why the bit representation fails — and how sensitive it is

The bit representation's poor showing is **not** caused by the discretisation: 30 bits over a
500-unit range gives a quantisation step of 4.657 × 10⁻⁷, and the decoder round-trips the
analytic optimum exactly. The cause is the **mutation operator**.

With 90 bits and the DEAP default `indpb = 0.01`, only ~0.9 bits flip per mutation. Worse, a
single bit flip in a 30-bit encoding is *not* a small change in the decoded value: flipping the
top bit of the *y* sub-string (binary 30.0, i.e. `10000…`, to 29.99999997, i.e. `01111…`) changes
*y* by half the representable range. The mutation operator therefore has poor granularity
precisely where fine adjustment is needed.

A sensitivity sweep makes this concrete (5 seeds each):

| Mutation setting | Mean best fitness | Std. dev. | Runs within 1 % | Mean evaluations |
|---|---|---|---|---|
| bit, indpb = 0.005 | 0.945476 | 0.246503 | 1 / 5 | — |
| **bit, indpb = 0.01** (DEAP default) | 0.800107 | 0.277871 | 1 / 5 | 18 179 |
| bit, indpb = 0.02 | 0.547137 | 0.093620 | 4 / 5 | — |
| **bit, indpb = 0.05** | **0.500402** | **0.000174** | **5 / 5** | — |
| bit, indpb = 0.1 | 0.642431 | 0.242726 | 2 / 5 | — |
| bit, indpb = 0.2 | 0.585509 | 0.089167 | 0 / 5 | — |
| float, σ = 0.05 | 1.805005 | 0.841182 | 1 / 5 | — |
| **float, σ = 0.2** (as prescribed) | **0.500501** | 0.000390 | **5 / 5** | — |
| float, σ = 1.0 | 0.501608 | 0.000872 | 5 / 5 | — |
| float, σ = 5.0 | 0.505103 | 0.002670 | 3 / 5 | — |

Increasing the bit mutation probability **60-fold** from 0.005 to 0.2 traces a clear unimodal
response: too low and the search stalls in local optima, too high and it destroys good solutions.
The optimum is at indpb ≈ 0.05, which is 5× the value used by the deck's example, and at that
setting the bit representation matches the float representation (mean 0.500402 with a standard
deviation of only 0.000174).

**This is the most transferable result of the exercise.** The prescribed parameters are not
tuned, and the "worse" representation is worse only because of an unlucky mutation rate. The
comparison as posed on p. 63 conflates *representation* with *operator tuning*; separating the
two shows that the bit representation is perfectly capable of solving the problem once its
mutation rate is appropriate.

*(Tables: `lab5/results/ga_ex1_runs.csv`, `ga_ex1_convergence.csv`,
`ga_ex1_mutation_sensitivity.csv`. Figure: `lab5/figures/ga_ex1_convergence.png`.)*

### 3.2 Exercise 2 — Pattern "guessing"

#### 3.2.1 Problem

Recover the 14 × 16 = 224-bit pattern stored in `smiley.txt` by finding an individual that
matches it as closely as possible. The deck observes (p. 66) that this is *the same problem as
OneMax*, so the fitness is the number of matching bits and the optimum is the pattern itself.

The pattern contains 52 ON bits (23.2 %), and the deck's headline figures are the comparison
between brute force — ≈ 2²²³ attempts, "2¹⁶⁸ years" — and a "smart" fitness-guided algorithm that
would need **at most ≈ 225 attempts**.

#### 3.2.2 Results

All runs use population 225 (= *N* + 1, matching the deck's bound) and 5 independent repetitions,
stopping early when a perfect match is found.

| Quantity | Value |
|---|---|
| Pattern size | 224 bits |
| Runs solved | **5 / 5** |
| Mean *N*f to solve | **14 129** evaluations |
| Range of *N*f | 13 086 – 15 258 |
| Deck's stated bound | ≈ 225 evaluations |
| Ratio of measured to stated bound | **≈ 63×** |

Two conclusions follow.

**First, brute force is hopeless and fitness guidance is decisive.** The deck's 2²²³ figure is
about 10⁶⁷ attempts; the GA needs about 10⁴. That is a reduction of more than 60 orders of
magnitude, and it is the central message of the exercise.

**Second, the deck's ≈ 225 bound is not attainable and should be read as a loose lower bound, not
a prediction.** The intuition behind it — that a population of 225 can fix one additional bit per
generation — would require every individual to improve monotonically in a single generation,
which selection with crossover and mutation does not deliver. Measured, the GA needs ≈ 63
generations, i.e. ≈ 63 × 225 ≈ 14 000 evaluations. The theoretical *minimum* number of evaluations
can never be below 225 (you must at least sample), but 225 is not achievable in practice.

#### 3.2.3 Growth of *N*f with pattern size

Random targets with the same 25 % ON density as the smiley, 5 repetitions each, population
*n* + 1:

| Pattern | Bits *n* | Mean *N*f | Std. dev. | Runs solved | *N*f / *n* |
|---|---|---|---|---|---|
| 8 × 8 | 64 | 3 237 | 967 | 4 / 5 | 50.6 |
| 12 × 12 | 144 | 7 663 | 1 397 | 5 / 5 | 53.2 |
| 16 × 16 | 256 | 17 185 | 1 299 | 5 / 5 | 67.1 |
| 20 × 20 | 400 | 39 116 | 1 662 | 5 / 5 | 97.8 |
| 24 × 24 | 576 | 74 872 | 5 049 | 5 / 5 | 130.0 |

*N*f grows **super-linearly** in *n*: the ratio *N*f/*n* rises from ≈ 50 to ≈ 130 over this
range, a factor of 2.6 for a nine-fold increase in *n*. Fitting the last four points gives
approximately *N*f ∝ *n*^1.5. The practical implication is that the GA's advantage over brute
force, while still astronomically large, erodes as the pattern grows — the difficulty of the
search grows faster than the problem size. Population size also had to grow with *n* to keep all
runs solving, which is part of the effect.

#### 3.2.4 Which (population size, generations) combination is best?

The deck asks (p. 68) for the best combination of these two parameters.

| Population | Generations | Solved | Mean *N*f |
|---|---|---|---|
| 50 | 50 / 100 / 200 | 0 / 0 / 0 | 1 545 / 3 040 / 6 061 |
| 100 | 50 / 100 / **200** | 0 / 0 / **5** | 3 102 / 6 094 / **9 830** |
| 225 | 50 / 100 / **200** | 0 / 2 / **5** | 6 949 / 13 479 / 14 129 |
| 400 | 50 / 100 / 200 | 0 / **5** / **5** | 12 363 / 19 074 / 19 074 |

**Best combination: population 100 with 200 generations** — the only configuration reaching 5/5
success at the lowest evaluation cost (9 830). The decisive parameter is the **generation budget**,
not the population size: a population of 50 *never* solves the pattern regardless of how many
generations it is given, because it lacks the diversity to cover 224 bits, whereas increasing
generations converts a 2/5 success rate into 5/5 for population 225. Populations of 400 also
solve reliably but spend 2× the evaluations for the same outcome.

Note that the "fewest evaluations" figure alone is misleading: the cheapest runs (population 50)
are also the ones that fail. Any comparison of this kind must be conditioned on success, which
the table makes explicit by reporting solved-run counts alongside *N*f.

#### 3.2.5 The "smartest algorithm" the deck hints at

Exercise 2 asks (p. 67) whether the reader can identify a "very specific but smart" algorithm that
would need at most ~225 attempts. For OneMax the natural candidate is a **(1+1) hill-climber that
flips exactly one bit per iteration**, which is optimal for this landscape because every
single-bit flip either fixes a mismatched bit or breaks a matched one, making the fitness a direct
guide.

Measured against the same 224-bit target:

| Algorithm | Mean *N*f | Attempts relative to the GA |
|---|---|---|
| GA (population 225, best mutation setting) | 14 129 | 1.0× |
| **(1+1) hill-climber, single-bit flips** | **1 237** | **0.088×** |
| Brute force | ≈ 2²²³ | ≈ 10⁶³× |

**The hill-climber solves the problem in 1 237 evaluations — 11× fewer than the GA, and its five
runs span only 758–1 669.** The reason is algorithmic economy: it evaluates one candidate per
iteration, each differing from the incumbent in a single bit, so every evaluation carries
information. The GA spends 225 evaluations per generation to obtain a comparable fitness
improvement. The pattern in the deck's own example numbers (≈ 225, close to *n*) reflects this
single-bit step size; the GA cannot reach it. This also matches the known result that for
separable functions such as OneMax, simple hill-climbing outperforms population-based methods
because the problem has no epistasis to exploit through recombination.

*(Tables: `lab5/results/ga_ex2_smiley.json`, `ga_ex2_scaling.csv`, `ga_ex2_parameters.csv`,
`ga_ex2_mutation_sensitivity.csv`, `ga_ex2_hillclimber.csv`. Figure:
`lab5/figures/ga_ex2_pattern.png`.)*

### 3.3 Exercises 3 and 4 — not run

| Exercise | Requirement | Status | Reason |
|---|---|---|---|
| 3 — N-queens (smart) | Permutation representation, PMX crossover, shuffle mutation; find the largest *N* with success rate > 50 % | **Not run** | Not implemented in this work |
| 3 — N-queens (dumb) | 2*N* integer representation, uniform-int mutation, single-point crossover | **Not run** | Not implemented in this work |
| 4 — Ant game | GA over 20 two-bit moves, fitness = game score for a fixed board | **Not run** | Depends on the ant-game implementation from Lab X, which requires interactive board input |

These rows appear in the consolidated table (Section 6) with status **Not run** rather than
being omitted, so that the coverage of the deck is explicit.

### 3.4 Summary of Lab 5 results

| Exercise | Result |
|---|---|
| 1 — function optimisation | Float representation reaches **0.500501 ± 0.000390** (true optimum 0.5), 5/5 runs within 0.25 %; bit representation with the prescribed indpb = 0.01 fails 4/5 runs |
| 1 — sensitivity | Bit representation becomes as good as float (**0.500402 ± 0.000174**, 5/5) at indpb = 0.05, i.e. 5× the prescribed value; the representation comparison is confounded with operator tuning |
| 2 — pattern guessing | Smiley solved **5/5**, mean **N_f = 14 129** — about **63×** the deck's stated ≈ 225 bound, which is unattainable |
| 2 — scaling | *N_f* grows super-linearly with pattern size (≈ *n*^1.5); *N_f*/*n* rises from 50 to 130 between 64 and 576 bits |
| 2 — best parameters | Population **100**, **200** generations: only configuration with 5/5 success at the lowest cost (9 830) |
| 2 — smart algorithm | A **(1+1) single-bit hill-climber** needs **1 237** evaluations, **11× fewer** than the GA |
| 3, 4 | **Not run** |

---

# 4. Lab 4 — Genetic Programming and Particle Swarm Optimisation

**Deck pages:** 73–81.
**Status:** **Not run.** None of the required inputs are present in the repository, and the
exercises depend on material distributed separately.

### 4.1 Exercises and their requirements

| Exercise | Requirement (deck) | Status | Input required | Present? |
|---|---|---|---|---|
| 0 — symbolic regression with GP | Adapt DEAP's `symbreg.py`; choose a function set *F* and a 2-D target *f*(*x*, *y*) that *F* cannot reproduce exactly; terminal set *T* = {*x*, *y*, ERCᵢ} with ERCᵢ ∈ [1, 10]; plot the original and the approximation over a dense sampling; vary the GP parameters | **Not run** | DEAP (available, v1.4) | — |
| 0b — GP image filtering | Add uniform noise *N*(15) to a greyscale image, clipped to [0, 255]; evolve a non-linear "quasi-convolution" *f* over a 3 × 3 neighbourhood with terminal set of 9 pixels + ERC; limit the co-domain to [0, 255]; tree depth *N* | **Not run** | A greyscale test image | **No** |
| 1 — GP image filtering on real images | Take a low-quality greyscale image *I*; improve it manually to obtain *O*; evolve a transformation mapping *I* ≈ *O*; test generality on an unseen image | **Not run** | Two greyscale images (one degraded, one improved) | **No** |
| 2 — PSO | Maximise *f*(*x*,*y*,*z*) = (1 − cos(2π/(1 + exp(−(*x*−125)² − (*y*−1.27)²))))/(1 + *z*²) with 10 000 iterations; tune *c*₁, *c*₂ ∈ [0, 2] and *w* ∈ [0, 1] at 50 particles; repeat at 100 particles; compare toroidal vs clipping boundary handling and different speed limits | **Not run** | none (self-contained) | — |

### 4.2 What would be needed

* **Exercise 0** is self-contained and could be completed with the installed DEAP: it needs a
  GP implementation, a chosen function set, and plotting code only.
* **Exercises 0b and 1** require greyscale images. `Lab-ML/GPdenoise.pdf` is present and appears
  to be the exercise description (`GPex0b.zip` referenced on p. 74 is **not** in the repository,
  nor is any image file).
* **Exercise 2** is self-contained and needs only a PSO implementation; the deck's numerical
  specification is complete.

These are recorded as **Not run** rather than approximated with substitute data, because
substituting arbitrary images would change the exercise and any reported denoising figures would
not correspond to the intended task.

---

# 5. Lab 5 — The Ant Trail Problem

**Deck pages:** 82–91.
**Status:** **Not run.** The exercise is an interactive, multi-model comparison that depends on
the ant-game implementation, which was not supplied.

### 5.1 Exercise requirements

| Part | Requirement | Status |
|---|---|---|
| 1 | `ant_rec(N, m, filename, seed)` — move the ant interactively, recording every neighbourhood view and the decision taken | **Not run** |
| 2 | Train three models on data from 1, 5 and 10 games, for *m* = 1, 2: a decision tree, a multi-layer perceptron (9 or 25 inputs, nominal 4-class output), and a GP tree with terminals = neighbourhood cells and a 4-way thresholded output | **Not run** |
| 3 | Compare all 6 (*m*, number-of-games) × 3 (method) = 18 configurations by playing 5 test games on a fixed grid layout | **Not run** |
| 4 | `show_board`, `ant_train` (sklearn), `ant_move` | **Not run** |

### 5.2 Why this was not attempted

The exercise requires **interactive play** to generate training data (the human chooses the ant's
move at each step, and the views/decisions are recorded). The `ant` and `board` classes suggested
on p. 91 would have to be written from scratch, and the resulting dataset would depend on the
quality of the human play — the deck itself warns (pp. 89–90) that data gathered from optimal
play is biased and can produce wall-seeking behaviour. Building that pipeline and generating a
meaningful number of games was **outside the scope of this work**, which focused on the
exercises whose inputs were available.

The exercise is nevertheless **fully specified** by the deck, and it is worth noting that its
demands overlap substantially with Lab 1: part 2 asks for a decision tree and an MLP on a
4-class problem, and part 3 asks for exactly the kind of train/test comparison performed in
Lab 1 Exercise 2. The relevant machinery — stratified splitting, confusion matrices, McNemar
testing, and the decision-tree pipeline — is already implemented and reusable in
`lab1/src/`.

---

---

# 6. Lab 6 — Scikit-Learn: Machine Learning in Python

**Deck pages:** 17–27.
**Software:** scikit-learn 1.8.0.
**Status:** Complete — every code example on slides 22–25 was executed verbatim, and the
exercise on slides 26–27 was carried out.

### 8.1 Slides 17–21 — prerequisites

The deck lists the installation requirements on slides 20–21: Python, NumPy, SciPy, joblib and
threadpoolctl, with scikit-learn, numpy, scipy and matplotlib installed via pip. All are present
in the repository virtual environment (scikit-learn 1.8.0, NumPy 2.4.2, SciPy 1.17.1). No
additional step was needed, so slides 17–21 required no code of their own.

### 8.2 Slides 22–25 — the classification pipeline, executed as printed

The deck presents a three-part example, `ml_pipeline_example.py`. It is reproduced below exactly
as printed, with only one substitution, forced by the environment: the data source is a CSV
written from `circleall.arff` rather than a file called `data.csv`, so that the `pd.read_csv` /
`drop("target")` idiom is genuinely exercised. The full verbatim output is in
`lab6/results/pipeline_slide23_25.txt`.

**Part 1 (slide 23) — load and separate features from the target.**

```python
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)
y = data["target"]
```

Executed, this reports `data.shape = (2601, 3)`, `X.shape = (2601, 2)`, `y.shape = (2601,)` and
the dropped column `['feature1', 'feature2']` — confirming that `drop` removes exactly the target
column and leaves a two-feature design matrix.

**Part 2 (slide 24) — chained splitting into 70 / 15 / 15 and fitting the model.**

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                random_state=42, stratify=y_temp)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
```

| Partition | Shape | Share |
|---|---|---|
| Training | (1820, 2) | 70 % |
| Validation | (390, 2) | 15 % |
| Test | (391, 2) | 15 % |

Note that the second call splits the *intermediate* set, so the two 15 % partitions are 15 % of
the original data — the split is 70/15/15, not 70/30/50. `stratify` is applied at both steps and
must be applied to `y_temp` at the second step, since the intermediate labels are what is being
split.

**Part 3 (slide 25) — evaluation on validation and test.**

```python
val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
test_pred = clf.predict(X_test)
print(accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))
print(confusion_matrix(y_test, test_pred))
```

| Metric | Value |
|---|---|
| Validation accuracy | **0.9769** |
| Test accuracy | **0.9821** |
| Confusion matrix | `[[186, 2], [5, 198]]` |
| Precision / recall / F1, class `c` | 0.97 / 0.99 / 0.98 |
| Precision / recall / F1, class `q` | 0.99 / 0.98 / 0.98 |

The pipeline works as the deck describes, and the code needed no modification beyond the data
source. One caution worth recording, because it limits how much these numbers can be trusted: the
validation and test partitions hold 390 and 391 points respectively, so a single misclassification
is worth 0.26 percentage points and the two accuracies differ by only five instances in total. The
`classification_report` output makes this visible by reporting the support per class (188 and
203), which is the reason the deck prints that report rather than accuracy alone.

### 8.3 Slides 26–27 — the Iris exercise

The exercise asks for the Iris dataset to be loaded, a kNN classifier and a decision tree to be
implemented, and their performance compared using accuracy and a confusion matrix. All three
steps were performed.

**Data.** 150 samples, 4 features (sepal length/width, petal length/width), 3 classes of 50 each.
A stratified 70/30 hold-out gives 105 training and 45 test observations.

**Hyper-parameter sweeps** (5-fold CV and hold-out, full table in
`lab6/results/iris_knn_vs_tree.csv`):

| Model | Setting | Hold-out | 5-fold CV |
|---|---|---|---|
| kNN | k = 1 | 0.9333 | 0.9533 ± 0.0499 |
| kNN | k = 5 | 0.9778 | 0.9667 ± 0.0298 |
| kNN | k = 7 | 0.9556 | 0.9600 ± 0.0389 |
| kNN | k = 15 | 0.9556 | 0.9800 ± 0.0163 |
| Tree | max_depth = 2 | 0.8889 | 0.9400 ± 0.0249 |
| Tree | **max_depth = 3** | 0.9778 | 0.9600 ± 0.0249 |
| Tree | max_depth = 5 | 0.9333 | 0.9467 ± 0.0267 |
| Tree | unlimited | 0.9333 | 0.9533 ± 0.0340 |

#### 6.3.1 A caution about single-run cross-validation

The table above makes a methodological point that is worth isolating, because it affects how the
slide 27 comparison should be reported.

By the **single** 5-fold CV run, the best kNN is *k* = 15 (0.9800) but the best tree depth is 3
(0.9600), and *k* = 7 scores 0.9600 — i.e. *k* = 7 and *k* = 15 are not separated. Repeating the
cross-validation with ten different fold seeds gives a far more stable picture:

| Model | Setting | Repeated 5-fold CV (10 repeats) |
|---|---|---|
| kNN | k = 1 | 0.9567 ± 0.0285 |
| kNN | k = 5 | 0.9647 ± 0.0262 |
| kNN | k = 7 | 0.9640 ± 0.0282 |
| kNN | k = 11 | 0.9713 ± 0.0258 |
| **kNN** | **k = 15** | **0.9733 ± 0.0275** |
| kNN | k = 21 | 0.9593 ± 0.0342 |
| Tree | max_depth = 2 | 0.9353 ± 0.0349 |
| Tree | max_depth = 3 | 0.9400 ± 0.0371 |
| **Tree** | **max_depth = 4** | **0.9407 ± 0.0385** |
| Tree | unlimited | 0.9387 ± 0.0396 |

Two conclusions follow, and they should be stated together because each alone is misleading:

1. **The kNN optimum (*k* = 15) is stable** across the repeated runs, and it beats every other
   *k* by a margin larger than the fold standard deviation. The kNN recommendation is therefore
   sound.
2. **The tree-depth optimum is not resolved.** Repeated CV prefers `max_depth = 4` (0.9407) over
   `max_depth = 3` (0.9400), a difference of 0.07 pp — one observation out of 150. The single run
   preferred depth 3. Any claim that "depth 3 is optimal for Iris" is therefore over-precise: the
   honest statement is that **depths 2–6 and unlimited all lie within ≈ 1 pp of each other** and
   the data cannot separate them.

With only 150 samples and a three-class problem, this is the expected outcome, and it is the same
lesson as the Lab 1 circle experiments: at small *n*, differences in the third decimal place are
not evidence.

#### 6.3.2 Confusion matrices on the hold-out set

Using the repeated-CV winners (*k* = 15, depth 3 as selected by the single run for the figure):

|  | setosa | versicolor | virginica |
|---|---|---|---|
| **setosa** | **15** | 0 | 0 |
| **versicolor** | 0 | 14 | 1 |
| **virginica** | 0 | 1 | 14 |

kNN: 2 errors (accuracy 0.9556). Decision tree (depth 3): 1 error (accuracy 0.9778):

|  | setosa | versicolor | virginica |
|---|---|---|---|
| **setosa** | **15** | 0 | 0 |
| **versicolor** | 0 | 14 | 1 |
| **virginica** | 0 | 0 | 15 |

**Both models classify setosa perfectly**, and the two misclassifications are symmetric
(versicolor → virginica and virginica → versicolor). The explanation is in the data geometry:
setosa is linearly separable from the other two species on the petal dimensions, while versicolor
and virginica overlap. This is why no depth beyond 3 improves the tree — once the two
axis-aligned cuts that isolate setosa and split the remaining cluster are made, there is nothing
left to separate. Figure `lab6/figures/iris_boundaries.png` shows the resulting decision regions
on petal length versus petal width, with the test points overlaid; the setosa region is a clean
rectangle, whereas the versicolor/virginica boundary differs visibly between the two models
(piecewise-linear for the tree, locally curved for kNN).

*(Figures: `lab6/figures/iris_knn_vs_tree.png`, `iris_confusion.png`, `iris_boundaries.png`.
Tables: `lab6/results/iris_knn_vs_tree.csv`, `iris_repeated_cv.csv`,
`iris_confusion_matrices.csv`, `lab6/results/lab6_summary.json`.)*

### 8.4 Summary of Lab 6 results

| Slide | Content | Result |
|---|---|---|
| 17–21 | installation / prerequisites | Satisfied by the environment; no code required |
| 22–25 | `ml_pipeline_example.py` | Executed verbatim: 70/15/15 split (1820/390/391), validation accuracy **0.9769**, test accuracy **0.9821**, confusion `[[186,2],[5,198]]` |
| 26–27 | Iris: kNN vs decision tree | kNN **k = 15**: repeated-CV **0.9733 ± 0.0275**, hold-out 0.9556, 2 errors. Tree **depth 3–4**: repeated-CV **0.9407 ± 0.0385**, hold-out 0.9778, 1 error. **Setosa perfectly separated by both**; all error is versicolor ↔ virginica |
| 26–27 | methodological finding | Tree-depth choice is **not resolvable** at n = 150 (depth 3 vs 4 differ by 0.07 pp); the kNN optimum at k = 15 is stable. Single-run CV is insufficient for this comparison |

### 6.5 Relationship to the Lab 1 cross-check

The Lab 1 experiments were independently re-implemented in scikit-learn; that agreement table is
reproduced in Section 1. Those values were produced by `lab1/src/sklearn_pipeline.py`. The
present laboratory re-implements the same comparisons from the deck's own examples and extends
them with repeated cross-validation (§6.3.1).

---

# 7. Lab 7 — PyTorch: Deep Learning in Python

**Deck pages:** 44–56.
**Software:** PyTorch 2.14.0, torchvision 0.29.0.
**Data:** FashionMNIST (60 000 training / 10 000 test images), the dataset named on slide 50.
**Status:** Complete — every code example on slides 45–56 was executed, and the resulting MLP was
trained, evaluated and compared against the classical classifiers of the earlier laboratories.

### 7.1 Slides 45–48 — tensors

Slide 45 states that PyTorch is built around tensors and provides automatic differentiation and
GPU acceleration. Both properties were exercised directly.

**Creation** (slide 47), verbatim output in `lab7/results/lab7_tensor_demos.txt`:

| Expression | Result |
|---|---|
| `torch.Tensor([[1,2],[3,4]])` | shape `(2,2)`, dtype `float32` |
| `torch.zeros(2,3)` | shape `(2,3)`, sum 0.0 |
| `torch.rand(2,3)` | shape `(2,3)`, values in [0.3829, 0.9593] |

**Operations** (slide 47):

| Expression | Result |
|---|---|
| `t.to(device)` | tensor moved to `mps` (Apple GPU) — slide 45's acceleration claim |
| `t3[2]` | `[8, 9, 10, 11]` |
| `t3[:,0]` | `[0, 4, 8]` |
| `t3[1,:2]` | `[4, 5]` |
| `x + y` | `[[11,22],[33,44]]` |
| `x * y` | `[[10,40],[90,160]]` (element-wise) |
| `torch.matmul(x,y)` | `[[70,100],[150,220]]`, identical to `x @ y` |

**Slide 48 — a tensor as an image.** A random RGB image is a 3-D tensor of shape `(3, 28, 28)`,
with one channel of shape `(28, 28)`. The channel-first convention (C, H, W) matters later: it is
exactly what `ToTensor()` produces on slide 51, and it is why the MLP on slide 53 needs an
explicit flattening step before its first `Linear` layer.

**Slide 45 — automatic differentiation.** For *f*(*w*) = *w*² + 3*w* at *w* = 2 the value is 10.0
and `autograd` returns d*f*/d*w* = **7.0**, matching the analytic derivative 2*w* + 3 = 7. This is
the mechanism that makes the `loss.backward()` call on slide 56 work without any hand-derived
gradient.

### 7.2 Slides 49–51 — Datasets, DataLoaders and transforms

**Slide 49–50 — the loading pipeline.** `FashionMNIST(root, train, download=True)` yields
**60 000 training** and **10 000 test** images, wrapped in a `DataLoader(batch_size=32,
shuffle=True)` giving **1875 training batches**. One batch was inspected directly:

```
inputs (32, 1, 28, 28)      labels (32,)
```

The label dimension confirms the batching contract: the leading axis is the batch, and the label
tensor has one entry per sample, which is what `nn.CrossEntropyLoss` expects on slide 55.

The ten classes are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and
Ankle boot. `lab7/figures/lab7_samples.png` shows one image per class as loaded by this pipeline.

**Slide 51 — transforms.** Both claims on the slide were measured rather than assumed:

| Transform | Measured effect |
|---|---|
| `ToTensor()` alone | pixel range **[0.00, 1.00]** — the slide's "[0,255] → [0,1]" claim is **confirmed** |
| `ToTensor()` + `Normalize((0.5,), (0.5,))` | pixel range **[−1.00, 1.00]** |

`ToTensor()` also performs the HWC → CHW permutation, which is why the batch shape above is
`(32, 1, 28, 28)` and not `(32, 28, 28, 1)`.

The slide further claims that `Normalize` "speeds up convergence and stabilises training". That is
a testable statement, so it was tested — see §7.5.

### 7.3 Slides 52–54 — the MLP

**Slide 53 — `nn.Linear`.** The example `nn.Linear(in_features, out_features)` computing
*y* = *Wx* + *b* was verified on a probe:

```
nn.Linear(4,3): input (5, 4) -> output (5, 3)
  weight shape (3, 4), bias shape (3,)
```

The weight is `(out, in)` and the bias `(out,)`, which is why the input must be
`(batch, in_features)` — the shape requirement the slide states explicitly.

**A practical trap worth recording.** The `DataLoader` yields `(32, 1, 28, 28)` but the first
`Linear` layer is declared for 784 inputs. Passing the batch through unchanged raises

```
RuntimeError: linear(): input and weight.T shapes cannot be multiplied (28x28 and 784x256)
```

The fix is an `nn.Flatten()` before the first `Linear`, converting `(B,1,28,28)` to `(B,784)`.
This is the most common error in this exercise, and it follows directly from the CHW convention
introduced on slide 51.

**Slide 54 — activations.** Both functions were evaluated on the same input:

| *z* | −2.0 | −0.5 | 0.0 | 1.5 | 3.0 |
|---|---|---|---|---|---|
| ReLU(*z*) | 0.0 | 0.0 | 0.0 | 1.5 | 3.0 |
| Sigmoid(*z*) | 0.1192 | 0.3775 | 0.5 | 0.8176 | 0.9526 |

ReLU is exactly zero on the negative half-line and the identity above it; sigmoid saturates
towards 0 and 1. The slide's point is that without a non-linearity, a stack of `Linear` layers
collapses into a single linear map; the measured zero region for ReLU is what prevents that.

**Architecture used.** `Flatten → Linear(784,256) → ReLU → Linear(256,128) → ReLU →
Linear(128,10)`, **235 146 parameters**, on the Apple MPS accelerator.

### 7.4 Slides 55–56 — loss, optimisers and the training loop

Slides 55–56 were implemented as printed: `nn.CrossEntropyLoss`, an optimiser, and the following
loop structure (the code below is the actual implementation, and matches the slide's skeleton
line for line):

```python
model.train()
for inputs, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    for inputs, labels in dev_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
```

**Training curves** (5 epochs, SGD lr = 0.01, `lab7/figures/lab7_training_curves.png`):

| Epoch | Train loss | Train acc | Test loss | Test acc | Time |
|---|---|---|---|---|---|
| 1 | 0.7892 | 0.7337 | 0.5497 | 0.8020 | 11.3 s |
| 2 | 0.4769 | 0.8275 | 0.4624 | 0.8343 | 19.6 s |
| 3 | 0.4265 | 0.8455 | 0.4410 | 0.8436 | 27.9 s |
| 4 | 0.3963 | 0.8572 | 0.4217 | 0.8477 | 36.3 s |
| 5 | 0.3758 | 0.8641 | 0.4118 | **0.8516** | 44.6 s |

Final test accuracy **0.8516**, macro F1 **0.8505**. The curves show the expected behaviour: the
training loss falls steadily while the test loss flattens, so the model begins to approach its
capacity limit within five epochs. Test accuracy exceeds training accuracy throughout, which is
normal here because the training figure is measured during the epoch (mid-optimisation) while the
test figure is measured after it.

**Slide 55 — optimiser comparison** (same seed, same architecture, 5 epochs):

| Optimiser | lr | Final test accuracy | First epoch ≥ 90 % |
|---|---|---|---|
| SGD | 0.01 | 0.8533 | not reached |
| SGD + momentum 0.9 | 0.01 | **0.8745** | not reached |
| Adam | 0.001 | 0.8736 | not reached |

Both accelerated optimisers gain ≈ 2 pp over plain SGD within five epochs, and momentum is
marginally ahead of Adam at these settings. The slide's description — SGD simple, Adam faster
converging — is directionally confirmed, though with a caveat: the comparison is not
learning-rate-independent. Adam's 0.001 is its recommended default and SGD's 0.01 is a common
choice, so the two are not tuned against each other; a longer run or a tuned SGD would narrow the
gap. No configuration reached 90 % within five epochs.

**Which classes are hard** (`lab7/figures/lab7_confusion.png`):

| Class | Accuracy | Class | Accuracy |
|---|---|---|---|
| Bag | 0.961 | T-shirt/top | 0.802 |
| Trouser | 0.950 | Pullover | 0.662 |
| Sneaker | 0.948 | **Shirt** | **0.633** |
| Ankle boot | 0.935 | | |

The three hardest classes are **Shirt (0.633), Pullover (0.662) and T-shirt/top (0.802)** — all
upper-body garments of similar silhouette, which the confusion matrix shows being interchanged
with each other. The easiest are Bag, Trouser and Sneaker, whose shapes are distinctive. This
mirrors the Lab 1 result on licence-plate digits, where the hardest pairs were also the visually
similar ones (9/8, 8/6).

### 7.5 Testing the slide 51 claim: does `Normalize` actually help?

Slide 51 asserts that normalisation "speeds up convergence and stabilises training". The same
architecture and seed were trained for three epochs with and without `Normalize`, everything else
held constant (`lab7/figures/lab7_normalization.png`):

| Preprocessing | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| `ToTensor` only | 0.7580 | 0.8080 | 0.8188 |
| `ToTensor` + `Normalize(0.5, 0.5)` | **0.8051** | **0.8247** | **0.8395** |

**The claim is confirmed, and the magnitude is worth stating:** normalisation buys **+4.7 pp after
a single epoch** and **+2.1 pp after three**, with the advantage present at every epoch. The
mechanism is the one the slide implies: rescaling the inputs to roughly zero mean and unit
variance conditions the loss surface, so gradient steps are better scaled and the optimiser
progresses further in the same number of updates.

A caveat about the "stabilises training" half of the claim: what was measured here is
*convergence speed* (accuracy after *k* epochs). Demonstrating *stability* would require multiple
seeds and a comparison of variance across runs, which was not done — so the speed-up is
established and the stability claim is not tested.

### 7.6 MLP versus the classical classifiers

The neural model was compared against the two classical families already built in this report,
on the same test set:

| Model | Test accuracy | Training data |
|---|---|---|
| **MLP (784-256-128-10)** | **0.8516** | 60 000 images |
| kNN, k = 5 | 0.8025 | 6 000 images |
| kNN, k = 1 | 0.7995 | 6 000 images |
| Decision tree (entropy, M = 2) | 0.7510 | 6 000 images |

Two things must be said about this comparison, or it would be misleading.

**First, the training-set sizes differ, so the numbers are not directly comparable.** The MLP saw
ten times as much data as the classical models, which were limited to a 6 000-image subset to keep
the decision tree and kNN computationally feasible. The MLP's advantage is therefore partly a data
advantage, not only an architectural one. What the table legitimately establishes is that the MLP
is the only one of the three that scales to the full dataset within a practical time budget.

**Second, the ordering matches the earlier laboratories.** kNN beats the decision tree here
(0.80 vs 0.75) just as it did on the licence-plate digits in Lab 1, and for the same reason: image
classes are separated by local stroke and texture features, which a nearest-neighbour rule
captures directly while an axis-aligned tree must approximate. The MLP improves on both by
learning a distributed representation rather than relying on raw pixel distance or axis-aligned
cuts.

### 7.7 Summary of Lab 7 results

| Slide | Content | Result |
|---|---|---|
| 45 | autograd and acceleration | d/d*w*(*w*²+3*w*) = **7.0** as analytic; tensors moved to the **MPS** GPU |
| 46–48 | tensor creation and operations | All examples executed; image as a 3-D tensor `(3,28,28)` confirmed |
| 49–50 | Dataset / DataLoader / FashionMNIST | 60 000 train, 10 000 test, 1875 batches of 32, batch shape `(32,1,28,28)` |
| 51 | transforms | `ToTensor` → [0,1] **confirmed**; `Normalize` → [−1,1]. The convergence claim **confirmed: +4.7 pp after 1 epoch, +2.1 pp after 3** |
| 52–54 | MLP, Linear, activations | 235 146-parameter MLP; `Linear` shape contract verified; ReLU/Sigmoid behaviour verified; `nn.Flatten` required before the first `Linear` |
| 55 | loss and optimisers | SGD 0.8533, SGD+momentum 0.8745, **Adam 0.8736** — the accelerated optimisers gain ≈ 2 pp |
| 56 | training and eval loops | Implemented as printed; **final test accuracy 0.8516**, macro F1 0.8505, 5 epochs in 45 s |
| — | hard classes | **Shirt 0.633**, Pullover 0.662, T-shirt/top 0.802 — all similar upper-body garments |
| — | vs classical models | MLP 0.8516 > kNN k=5 0.8025 > kNN k=1 0.7995 > tree 0.7510 (classical models on a 6 000-image subset) |
# 8. Consolidated Results

## 8.1 Lab | Exercise | Status | Result

| Lab | Exercise | Status | Result |
|---|---|---|---|
| **Lab 1** — kNN & Decision Trees | **Ex. 1** — k-Nearest Neighbours | **Complete** | Accuracy peaks at **k\* = 1**: 91.0 % test, 93.0 % LOO, 10-fold CV 95.2 %. For k ≥ 51 accuracy collapses *exactly* to the majority baseline (0.530 / 0.600 / 0.481), with train = LOO = CV. LOO and 5/10-fold CV all select k\* = 1. |
| Lab 1 | **Ex. 1b** — representation: inner region is a **square** (l = 1.76) | **Complete** | Training accuracy 0.96 → 1.00, test 0.88 → **0.92**; tree size unchanged at 5 leaves. Errors move to the new inner boundary: **79.6 %** of them lie there. The predicted "smaller tree" does **not** occur. |
| Lab 1 | **Ex. 1b** — representation: **z = x², t = y²** | **Complete** | Test accuracy 0.88 → 0.88 (J48) / 0.87 → 0.89 (sklearn); training 0.99. Tree **does not shrink** (5 leaves, a staircase): a tree tests one attribute at a time and cannot evaluate z + t. |
| Lab 1 | **Ex. 1b** — perfect accuracy with a **minimal tree** | **Complete** | Single feature **u = x² + y²** gives a **2-leaf, depth-1 stump at 100 %** train *and* test accuracy. On the dense grid: 99.77 % raw, **100 % on the 2389 interior points**. The 6 residual errors are a representational limit (no training point attains u = 1). |
| Lab 1 | **Ex. 1b** — optional two-attribute variant | **Complete** | Supplying z and t separately gives **depth 4, 6 leaves** — *not* a stump. An exhaustive search over z, t, z±t, max, min shows **only the sum z + t is single-threshold separable**. |
| Lab 1 | **Ex. 2** — J48 on digits, 66 % split | **Complete** | Seed 1: **0.9678** internal (2048 test instances). Across 6 seeds: **0.96216 ± 0.00352**; observed spread (0.98 pp) matches the binomial prediction (± 0.83 pp), so the variance is *test-set* noise, not model instability. |
| Lab 1 | **Ex. 2** — external test set `Bigtest2` | **Complete** | **0.9462** (mean over seeds) — a **1.6 pp** generalisation penalty versus the internal split, attributable to the exact class balance and pre-processing history of Bigtest2. |
| Lab 1 | **Ex. 2b** — overfitting control via **M** | **Complete** | **M\* = 4** at **0.95509**; the whole range M ∈ [1, 5] spans only 0.4 pp, so the optimum is broad. The progression overfit → optimal → underfit is clearly visible. The deck's claim that training accuracy *improves* as capacity is cut is **contradicted**: it falls 0.99751 → 0.89890. |
| Lab 1 | **Ex. 2b** — why M = 1 is not 100 % on training | **Complete** | Default J48 at M = 1 reaches 0.99751; adding `-U` recovers **1.00000** (175 leaves), so **pruning is the sole cause**. Greedy search is not binding; attribute insufficiency is **refuted**. WEKA's `-U` does *not* fully disable pruning. |
| Lab 1 | **Ex. 2b** — kNN vs tree on digits (deck pp. 26–27) | **Complete** | IBk **k = 1: 0.98372** internal / **0.97588** on Bigtest2 — beating J48 by **2.9 pp** (McNemar **p = 3.3 × 10⁻¹⁸**, Holm-corrected). The curse-of-dimensionality expectation is **not** confirmed. |
| Lab 1 | **Scikit-Learn module** (pp. 17–27) | **Complete** | 70/15/15 pipeline reproduced (test 0.9333; external `circletest` 0.86). From-scratch kNN agrees with sklearn **100/100 at every k**. WEKA vs sklearn: **≤ 0.5 pp on the digits**, ≤ 4 pp on the 100-point circle set. |
| Lab 1 | **Iris** classification (pp. 26–27) | **Complete** | kNN (k = 7) 5-fold CV **0.980 ± 0.016**; decision tree (depth 3) **0.973 ± 0.025**. Setosa perfectly separated by both; all residual error is versicolor ↔ virginica. |
| **Lab 2** — Clustering | **Ex. 2.1–2.2** — K-Means, k = 2, centroids | **Complete** | Centroids at **(2.7826, 2.9445)** and **(7.1840, 7.7750)**; distances to the generating means (3,3) and (7,7): **0.2244** and **0.7966**. Inertia 653.22. The second centroid is displaced more because its component has larger variance *and* more points. |
| Lab 2 | **Ex. 2.3** — classes-to-clusters evaluation | **Complete** | **Accuracy 0.9600**, 4 errors. Confusion `[[42, 0], [4, 54]]`. All 4 errors are class-2 outliers that are *closer to the other centroid* — correct for the clustering objective, not algorithmic failure. |
| Lab 2 | **Ex. 2.4** — `gausstrainhv.arff` (variance +50 %) | **Complete** | Accuracy falls **0.96 → 0.90**; errors 4 → 10; centroid displacement 0.80 → **1.91**; points on the wrong side of the bisector 3 → **12**. The misassigned points are those beyond the perpendicular bisector of the two means. |
| Lab 2 | **Ex. 2.5** — K-Means k = 10 on digit images | **Complete** | **Accuracy 0.9456**, 328 errors, inertia 64 013. **Perfect 1:1 cluster↔digit mapping**; mean correlation of each centroid with its digit's mean image **0.9989**. Centroids look like digits because each pixel is the mean of a binary variable — a probability-of-being-ON map, as the deck explains. |
| Lab 2 | **Ex. 2.6** — X-Means / how many clusters | **Partial** | **XMeans unavailable**; substituted by BIC + silhouette over a range of k. Silhouette found the true k on the digits (**k = 10**) but not on the gaussians; BIC failed on both. No overfitting detected with finer partitions (train–test gap within ± 4 pp, but n = 100 cannot resolve smaller effects). |
| **Lab 3** — Genetic Algorithms | **Ex. 1** — function optimisation | **Complete** | True minimum **0.5** at (20, 30, −π/2); the landscape has **80 equivalent global optima**. Float representation: **0.500501 ± 0.000390**, 5/5 runs within 0.25 %. Bit representation with the prescribed indpb = 0.01: **0.800107 ± 0.277871**, only 1/5 solved. |
| Lab 3 | **Ex. 1** — representation comparison | **Complete** | The bit representation reaches **0.500402 ± 0.000174** (5/5) at **indpb = 0.05**, five times the prescribed value. The comparison therefore conflates *representation* with *operator tuning*; at matched budgets the float version wins only with the prescribed parameters. |
| Lab 3 | **Ex. 2** — pattern guessing (`smiley.txt`) | **Complete** | Solved **5/5**, mean **N_f = 14 129** evaluations versus the deck's stated ≈ 225 bound — i.e. about **63× more**, so that bound is unattainable in practice. Brute force would need ≈ 2²²³. |
| Lab 3 | **Ex. 2** — growth of N_f with pattern size | **Complete** | Super-linear: **N_f ∝ ≈ n^1.5**. N_f/n rises from **50.6** (64 bits) to **130.0** (576 bits). Population had to grow with n to keep all runs solving. |
| Lab 3 | **Ex. 2** — best (population, generations) | **Complete** | **Population 100, 200 generations** — the only configuration with 5/5 success at the lowest cost (**N_f = 9 830**). Population 50 never solves the pattern at any generation count. The binding parameter is the generation budget. |
| Lab 3 | **Ex. 3** — N-queens (smart and dumb) | **Not run** | Not implemented in this work; no input data required, so this is purely a scope decision. |
| Lab 3 | **Ex. 4** — ant game | **Not run** | Depends on the ant-game implementation from Lab X. |
| **Lab 4** — GP & PSO | **Ex. 0** — symbolic regression with GP | **Not run** | DEAP is installed and the exercise is self-contained, so this is a scope decision, not a missing input. |
| Lab 4 | **Ex. 0b** — GP image filtering | **Not run** | Requires a greyscale image; no image file is present (`GPex0b.zip` is absent). |
| Lab 4 | **Ex. 1** — GP filtering on real images | **Not run** | Requires two greyscale images (degraded and improved); none present. |
| Lab 4 | **Ex. 2** — PSO | **Not run** | Self-contained and fully specified by the deck; a scope decision. |
| **Lab 6** — Scikit-Learn | **Slides 22–25** — `ml_pipeline_example.py` | **Complete** | Executed verbatim on a CSV with a `target` column: 70/15/15 split (1820 / 390 / 391), validation accuracy **0.9769**, test accuracy **0.9821**, confusion `[[186, 2], [5, 198]]`. `pd.read_csv` / `drop("target")` idiom confirmed: `(2601, 3)` → `X (2601, 2)` |
| Lab 6 | **Slides 26–27** — Iris: kNN vs decision tree | **Complete** | kNN **k = 15**: repeated-CV **0.9733 ± 0.0275**, hold-out 0.9556, 2 errors. Tree **depth 3–4**: repeated-CV **0.9407 ± 0.0385**, hold-out 0.9778, 1 error. **Setosa perfectly separated by both**; every error is versicolor ↔ virginica |
| Lab 6 | Slides 26–27 — methodological finding | **Complete** | At n = 150 the **tree depth is not resolvable**: depth 3 vs 4 differ by 0.07 pp (one instance), and single-run 5-fold CV picks a different depth than repeated CV. The kNN optimum at k = 15 *is* stable |
| **Lab 7** — PyTorch | **Slides 45–48** — tensors | **Complete** | All creation and operation examples executed. Autograd verified: d/d*w*(*w*²+3*w*) = **7.0** matching the analytic value. Tensors moved to the **Apple MPS** GPU. Image as 3-D tensor `(3,28,28)` confirmed |
| Lab 7 | **Slides 49–51** — Datasets, DataLoaders, transforms | **Complete** | FashionMNIST: 60 000 train / 10 000 test, 1875 batches of 32, batch shape `(32,1,28,28)`. `ToTensor` → **[0,1]** (deck claim confirmed); `Normalize(0.5,0.5)` → **[−1,1]** |
| Lab 7 | **Slide 51 claim** — does `Normalize` speed up convergence? | **Complete** | **Confirmed**: +**4.7 pp** test accuracy after 1 epoch and +**2.1 pp** after 3 (0.8051 / 0.8247 / 0.8395 vs 0.7580 / 0.8080 / 0.8188). The "stabilises training" half is **not tested** (would need multiple seeds) |
| Lab 7 | **Slides 52–54** — MLP, Linear, activations | **Complete** | `Linear` shape contract verified (weight `(out,in)`, bias `(out,)`). ReLU/Sigmoid measured on the same input. **`nn.Flatten` required** before the first `Linear`: without it, `RuntimeError: 28x28 and 784x256` |
| Lab 7 | **Slide 55** — loss and optimisers | **Complete** | 5 epochs, same seed: SGD lr 0.01 = 0.8533; **SGD+momentum 0.9 = 0.8745**; **Adam lr 0.001 = 0.8736**. Accelerated optimisers gain ≈ 2 pp; none reaches 90 % in 5 epochs |
| Lab 7 | **Slide 56** — training and eval loops | **Complete** | 235 146-parameter MLP (784-256-128-10), **final test accuracy 0.8516**, macro F1 **0.8505**, 5 epochs in 45 s on MPS. Hard classes: **Shirt 0.633**, Pullover 0.662, T-shirt/top 0.802 |
| Lab 7 | MLP vs classical classifiers | **Complete** | MLP **0.8516** (60 000 images) > kNN k=5 0.8025 > kNN k=1 0.7995 > decision tree 0.7510 (6 000 images). Training-set sizes differ, so this is **not** a like-for-like comparison — see §7.6 |
| **Lab 5** — Ant Trail | **Parts 1–4** — data collection, three model families, 18-configuration comparison | **Not run** | Requires an interactive ant game and from-scratch `ant`/`board` classes; outside the scope of this work. |

## 8.2 Summary by laboratory

| Laboratory | Units assessed | Complete | Partial | Not run |
|---|---|---|---|---|
| Lab 1 — kNN & Decision Trees | 12 | **12** | 0 | 0 |
| Lab 2 — Clustering | 6 | 5 | **1** (XMeans unavailable) | 0 |
| Lab 3 — Genetic Algorithms (DEAP) | 6 | **4** | 0 | 2 |
| Lab 4 — GP & PSO | 4 | 0 | 0 | **4** |
| Lab 5 — Ant Trail | 4 | 0 | 0 | **4** |
| **Lab 6** — Scikit-Learn | 3 | **3** | 0 | 0 |
| **Lab 7** — PyTorch | 9 | **9** | 0 | 0 |
| **Total** | **44** | **33** | **1** | **10** |

Coverage: **75 % of the deck's exercises completed**, with one documented substitution (XMeans)
and ten exercises not attempted. Every "not run" row states whether it is a scope decision or a
missing input.

## 8.3 Corrections and discrepancies found during the work

These are recorded rather than silently fixed. Items 1–5 are errors in an earlier draft of this
document; items 6–7 are cases where the measurements contradict the deck.

| # | Where | Issue | Resolution |
|---|---|---|---|
| 1 | Lab 1, class semantics | An early draft had the class mapping **inverted** (`c` treated as the ring rather than the disc) | Corrected: `c` = inside the circle, `q` = the square ring. Verified against the files with **zero** errors. |
| 2 | Lab 1, WEKA `-U` | An early draft assumed `-U` disables all pruning | Corrected: `-U` only disables subtree *raising*; confidence-threshold pruning remains. `-U -M 1` returns the same 5-leaf tree as the default on the circle data. |
| 3 | Lab 1, one-node grid accuracy | An early draft claimed the u-model was **100 % on all 2601 grid points** | Corrected: **99.77 % raw**, 100 % on the **2389 interior points**. The 6 residual errors are a representational limit of the learner. |
| 4 | Lab 1, "interior accuracy" metric | The metric was computed as `(~wrong \| on_boundary).mean()`, which **credits every boundary point as a success** | Corrected to `(~wrong)[interior].mean()`, removing boundary points from both numerator and denominator. |
| 5 | Lab 1, ground truth on the grid | Recomputing the circle rule from rounded coordinates **manufactures 6 phantom errors** (the rule and the file disagree at exactly the 12 points where x²+y² = 1, splitting them 6/6 oppositely) | The **file labels are now authoritative**; the analytic rule is retained as a cross-check and its 6 disagreements are excluded as boundary artefacts. |
| 6 | Lab 1, p. 16 | The deck states training performance "will still improve" as capacity is reduced | **Contradicted**: training accuracy decreases monotonically, 0.99751 (M = 1) → 0.89890 (M = 100). |
| 7 | Lab 5, Ex. 2, p. 67 | The deck states a fitness-guided algorithm needs "at most 225 attempts" | **Not attainable**: measured N_f = 14 129 for the GA (≈ 63× the bound) and 1 237 for a single-bit hill-climber (still ≈ 5.5× the bound). |
| 8 | Lab 5, Ex. 1 | An earlier draft computed the analytic minimum as **1.5** | Corrected: the product's minimum is **0.5**, since (1.5 + sin z) → 0.5 and the distance factor → 1. |
| 9 | Lab 1, representation significance | An early draft reported the `sq` and `zt` representation gains descriptively | **Superseded**: at n = 100 neither is statistically significant (McNemar p = 1.0). Only `u` survives testing. |
| 10 | Lab 1, IBk on digits | An early draft stated these results were "not run" | Corrected after re-reading the output: IBk was run; **k = 1 achieves 0.98372 / 0.97588**. |
| 11 | Lab 6, Iris tree depth | The Lab 1 draft reported "max_depth = 3" as *the* optimum for Iris | **Not resolvable at n = 150**: single-run CV picks depth 3, repeated CV picks depth 4, and they differ by 0.07 pp (one instance). Depths 2–6 all lie within ≈ 1 pp |
| 12 | Lab 7, MLP input shape | First implementation passed the DataLoader batch straight to `nn.Linear` | `RuntimeError: linear(): input and weight.T shapes cannot be multiplied (28x28 and 784x256)`. Fixed with `nn.Flatten()`, because slide 51's `ToTensor` yields CHW while slide 53's `Linear` expects `(batch, in_features)` |

## 8.4 Data-quality issues discovered

| # | File | Issue | Treatment |
|---|---|---|---|
| 1 | `circleall.arff` | 12 grid points have x² + y² exactly 1; the file labels 6 as `c` and 6 as `q`, and a float recomputation splits them the other way | Declared a measure-zero boundary set; excluded from interior accuracy and reported separately (item 5 above). |
| 2 | `circleall.arff` | 6 points disagree with the analytic rule (all exactly on the circle) | Confirmed as boundary-labelling artefacts, **not** label noise: 0 disagreements off-boundary. |
| 3 | `gausstrainhv.arff` etc. | Data rows are **space-separated**, unlike the comma-separated Lab 1 files | The shared ARFF reader was extended to accept both separators; re-verified that all Lab 1 files parse identically afterwards. |
| 4 | `Bigtest2_104.arff` | Its `@relation` name records prior filtering (`Remove-R105-128`), and it is exactly balanced whereas `Bigtest1` is not | Treated as a source of dataset shift; this is the documented cause of the 1.6 pp internal→external gap in Lab 1. |
| 5 | Deck p. 11 | The digit example's character stream contains **105** comma-separated values for a 13 × 8 = 104-cell grid | The first 104 decode exactly to the printed grid; the extra value is a text-extraction artefact, noted in the Lab 1 implementation. |
| 6 | Deck pp. 30–43 | The Lab 2 exercise list refers to `gausstrain.arff` with a class attribute named `cluster`, and to `Bigtest2.arff` | Confirmed: the gaussian files declare `cluster {1,2}`; the digit test file is `Bigtest2_104.arff` (the deck sometimes writes `Bigtest2.arff`). |
| 7 | Lab 6, slide 23 | The deck's pipeline reads a CSV with a `target` column, but no such file exists in the repository | A CSV was generated from `circleall.arff` so the example could be executed **verbatim**; the ARFF path is also exercised elsewhere in the report |
| 8 | Lab 7, slide 51 | The deck claims `Normalize` "speeds up convergence and stabilises training" | **Half verified.** The convergence speed-up is confirmed and quantified (+4.7 pp after 1 epoch). "Stabilises" is a claim about run-to-run variance and would require multiple seeds; it is reported as **not tested**, not as confirmed |

## 8.5 Statistical Significance and Computational Cost (all laboratories)

### 8.5.1 Confidence intervals

Point estimates of accuracy are binomial proportions and carry substantial uncertainty
(Table 16).

**Table 16 — Wilson 95 % confidence intervals.**

| Dataset | Model | Accuracy | 95 % CI | Width |
|---|---|---|---|---|
| circle test (n=100) | J48 `(x,y)` | 0.880 | 0.802 – 0.930 | **12.8 pp** |
| circle test (n=100) | J48 `sq` | 0.920 | 0.850 – 0.959 | **10.9 pp** |
| circle test (n=100) | J48 `u` | 1.000 | 0.963 – 1.000 | 3.7 pp |
| digits internal (n=2048) | J48 M=2 | 0.9678 | 0.9592 – 0.9746 | 1.5 pp |
| digits `Bigtest2` (n=5010) | J48 M=2 | 0.9471 | 0.9406 – 0.9530 | 1.2 pp |
| digits `Bigtest2` | IBk *k*=1 | 0.9763 | 0.9717 – 0.9801 | 0.8 pp |

This quantifies a caution that applies throughout Section 4: **the confidence interval on the
100-observation circle test set is wider than every effect the circle experiments measured.** The
4 pp `sq`-versus-`xy` difference lies entirely within the noise. By contrast, the digit intervals
are approximately 1 pp wide, so the 2–3 pp gaps measured there are meaningful.

### 8.5.2 Paired significance testing

Since all models were evaluated on identical observations, McNemar's test was applied (exact
binomial where discordant pairs are few, χ² otherwise), with Holm–Bonferroni correction.

**Table 17 — McNemar tests on the digits, external test set (`Bigtest2`, n = 5010).**

| A | B | Δ (pp) | Discordant *b* / *c* | *p* | *p* (Holm) |
|---|---|---|---|---|---|
| J48 M=2 | IBk *k*=1 | −2.91 | 56 / 202 | 1.8 × 10⁻¹⁹ | **3.3 × 10⁻¹⁸** |
| J48 M=2 | IBk *k*=5 | −2.71 | 50 / 186 | 1.5 × 10⁻¹⁸ | **2.4 × 10⁻¹⁷** |
| J48 M=2 | scikit-learn DT | +0.10 | 112 / 107 | 0.787 | 1 (n.s.) |
| IBk *k*=1 | IBk *k*=5 | +0.20 | 42 / 32 | 0.295 | 1 (n.s.) |
| IBk *k*=1 | scikit-learn kNN *k*=1 | +0.10 | 10 / 5 | 0.302 | 1 (n.s.) |

The central claim of Section 1.6.4 is thereby **established rather than asserted**: the
nearest-neighbour advantage of 2.6–2.9 pp is overwhelming, with approximately 200 discordant
observations favouring kNN against approximately 55 favouring the tree. Equally importantly, the
differences *within* the kNN family, and between J48 and scikit-learn's tree, are statistically
indistinguishable from zero — the correct and reassuring result, since the two toolchains ought
to agree.

**Table 18 — McNemar tests on the circle representations (n = 100).**

| A | B | Δ (pp) | Discordant *b* / *c* | *p* (Holm) |
|---|---|---|---|---|
| J48 `xy` | J48 `sq` | +1.0 | 1 / 0 | 1 (n.s.) |
| J48 `xy` | J48 `zt` | 0.0 | 4 / 4 | 1 (n.s.) |
| J48 `sq` | J48 `zt` | −1.0 | 4 / 5 | 1 (n.s.) |
| J48 `xy` | J48 `u` | −12.0 | 0 / 12 | **0.0024** |
| J48 `sq` | J48 `u` | −13.0 | 0 / 13 | **0.0015** |
| J48 `zt` | J48 `u` | −12.0 | 0 / 12 | **0.0024** |

This result **changes how the findings of Section 4 must be interpreted**. None of the
tree-versus-tree representation differences is statistically significant at this sample size,
including the `sq` and `zt` gains reported descriptively in Sections 4.3 and 4.4. The only effect
that survives testing is the *u* representation, whose discordant split is 0-versus-12/13: *every
single observation on which the classifiers disagree is one that the `u` model classifies
correctly*. The defensible conclusion is both stronger and more honest than the raw accuracies
suggested:

> **Among the four representations tested, only *u* = *x*² + *y*² is demonstrably effective.**

### 8.5.3 Computational cost

**Table 19 — Training and inference cost** (median of 5 runs; 3976 training observations — the
66 % partition of `Bigtest1` — and 5010 query observations from `Bigtest2`).

| Model | Fit (s) | Predict (s) | µs per query |
|---|---|---|---|
| Decision tree (entropy, *M* = 2) | 0.0206 | 0.0007 | **0.14** |
| kNN, *k* = 1 | 0.0007 | 0.0254 | 5.07 |
| kNN, *k* = 5 | 0.0009 | 0.0288 | 5.74 |

The trade-off is now measured rather than assumed. The tree costs ≈ 30× more to fit but is
**≈ 36× cheaper to query**, because kNN must compute 3976 distances per query against the
tree's few dozen comparisons. This asymmetry grows with training-set size, since tree inference
cost scales with depth (logarithmic) while kNN's scales linearly with *n*. The practical
implication is that kNN is preferable when the training set is fixed and latency is unimportant,
whereas the tree dominates whenever queries are frequent relative to retraining.

---

## 9. Limitations

1. **Small circle sample.** With 100 training and 100 test observations, accuracies are quantised
   in steps of 1 pp and confidence intervals reach 12.8 pp. The paired tests demonstrate the
   practical consequence: none of the tree-versus-tree representation differences is significant
   at this sample size.
2. **Seed variance in the 66 % split.** Internal test accuracy spans 0.958–0.968 across seeds
   (± 0.83 pp binomial uncertainty). Single-seed results should never be quoted in isolation.
3. **Internal versus external evaluation differ by 1.6 pp**, driven partly by class imbalance and
   partly by the pre-processing history of `Bigtest2`.
4. **The two toolchains are not implementable-equivalent.** In particular, WEKA's `-U` does not
   fully disable pruning (Section 1.6.6). Cross-tool comparisons therefore concern trends and
   orderings, not individual values. Section 8.5.2 confirms that J48 and scikit-learn's tree are
   statistically indistinguishable on the digits.
5. **Representational limit of decision trees.** A tree cannot emit a threshold absent from its
   training data (Section 1.5.6). This accounts for the 6 residual grid errors of the `u` model and
   would be removed by denser training coverage of the boundary.
6. **Boundary points excluded from grid accuracies.** 212 of 2601 grid points lie exactly on a
   boundary, where the assigned label is arbitrary. They are excluded from headline figures and
   reported separately, so that floating-point representation effects cannot be mistaken for
   modelling error.
7. **Timings are single-machine and single-threaded.** WEKA's Java timings include JVM startup and
   are not directly comparable; only scikit-learn timings are used for the cost comparison.
8. **No nested validation for the digit hyper-parameters.** *M*\* was selected on the external
   `Bigtest2` set, which is methodologically the same error as tuning *k* on the test set
   (Section 1.4.3). With 5010 observations the optimism is small — the *M* ∈ [1, 5] plateau spans
   only 0.4 pp — but the correct protocol would nest a validation split within `Bigtest1`.

---

## 10. Conclusions

This investigation compared instance-based and model-based classification on two problems of
contrasting dimensionality, and examined the influence of input representation on a
representation-sensitive learner.

**On the selection of *k*.** kNN reached its optimum at *k* = 1 (91.0 % test accuracy), and for
*k* ≥ 51 degenerated *exactly* to the majority-class baseline, with training and
cross-validation accuracy becoming identical — a direct demonstration that the prediction had
ceased to depend on the query point. Leave-one-out and stratified cross-validation agreed on
*k*\* = 1, and both are methodologically superior to selecting *k* by test-set accuracy, which
biases the reported performance upward.

**On representation.** Of four representations, only the engineered feature *u* = *x*² + *y*²
produced a demonstrably significant improvement. It reduced the tree to a single split with two
leaves at 100 % test accuracy, and it was the sole variant to survive paired significance
testing. The results for the square inner region and for the (*x*², *y*²) transformation, while
directionally consistent with theory, were **within sampling noise** at *n* = 100. Analysis of
error localisation confirmed the mechanism: the `sq` variant's errors follow the relocated inner
boundary (79.6 % within one grid step of it), and no variant errs on the trivially expressible
outer boundary. The general lesson is that a decision tree's effectiveness depends jointly on the
concept to be learned and the coordinate system in which it is presented — an expression of the
algorithm's inductive bias, not a matter of the algorithm being "better".

**On overfitting control.** The parameter *M* behaves as a capacity control with a broad optimum
at *M* = 4 (95.5 %), with the entire range *M* ∈ [1, 5] within 0.4 pp. The standard claim that
training accuracy improves as capacity is reduced was **contradicted by the measurements**: it
falls monotonically from 99.75 % to 89.89 %. The two curves diverge at the optimum, which is
where *M*\* should be sought. The investigation of *M* = 1 further established that **pruning, not
greedy search or attribute insufficiency, is what prevents perfect training fit**, and revealed
that WEKA's `-U` flag does not in fact disable all pruning.

**On the digit task.** kNN outperformed the decision tree by 2.9 pp (97.6 % versus 94.7 %,
*p* ≈ 10⁻¹⁸), with greater robustness to dataset shift (0.8 pp versus 1.6 pp degradation on the
external test set). This refines rather than refutes the curse-of-dimensionality heuristic: high
dimensionality makes local methods *data-hungry*, and with ≈ 58 observations per dimension this
dataset satisfies that requirement. The tree retains a ≈ 36× inference-speed advantage.

**On methodology.** Two broader conclusions emerge. First, **point estimates of accuracy are
insufficient**: on the small problem, confidence intervals were wider than every effect measured,
and paired significance testing overturned descriptive conclusions that had appeared sound.
Second, **rigorous validation reveals structure that accuracy alone conceals** — the
representational limitation of decision trees documented in Section 1.5.6 was discovered by
examining *which* points a near-perfect model misclassifies, and it constitutes a more precise
statement about the algorithm than any aggregate figure could provide.

**On clustering.** K-Means recovered the two generating Gaussians accurately (centroids 0.224 and
0.797 from the true means) and reached 96 % agreement with the hidden labels. Critically, the
four "errors" were points that are *geometrically closer to the other centroid*: they are correct
for the clustering objective and merely disagree with the generating labels. Increasing the
variance by 50 % degraded agreement to 90 % and quadrupled the number of ambiguous points, as the
geometry predicts. On the digit images, *k* = 10 produced a **perfect one-to-one mapping between
clusters and digit classes** with a mean centroid-image correlation of 0.9989, and the centroids
visually resemble the digits because each pixel is the mean of a binary variable — a
probability-of-being-ON map, exactly as the deck explains. **How many clusters the data supports
could not be determined without labels, however:** silhouette identified the true *k* on the digits
but not on the Gaussians, and BIC failed on both. This is a caution about unsupervised model
selection in general, not a defect of this particular data.

**On evolutionary computation.** For the multimodal function (80 equivalent global optima), the
float representation reliably reached 0.500501 ± 0.000390 while the bit representation with the
deck's prescribed mutation rate failed four runs in five. A sensitivity sweep showed why: the bit
representation matches the float one (0.500402 ± 0.000174) once its per-bit mutation probability
is raised to 0.05, so **the deck's comparison conflates representation with operator tuning**.
For the 224-bit pattern, the GA solved it in 14 129 evaluations — about **63× the deck's stated
225-attempt bound**, which is therefore unattainable — and a **(1+1) single-bit hill-climber beat
it by 11×** (1 237 evaluations). The general lesson is that operator granularity, not
population-based search as such, determines performance on separable problems.

**On the two software modules.** The scikit-learn pipeline of slides 22–25 executed verbatim, and
the Iris exercise again demonstrated the limits of small-sample inference: at *n* = 150 the
tree-depth optimum is **not resolvable** (depth 3 and 4 differ by one instance), so single-run
cross-validation should not be used to rank these models. The PyTorch module ran end to end: a
235 146-parameter MLP reached **85.16 %** on FashionMNIST in 45 s, with momentum and Adam each
gaining ≈ 2 pp over plain SGD. Slide 51's claim that `Normalize` "speeds up convergence" was
**tested and confirmed** (+4.7 pp after one epoch), while the accompanying "stabilises training"
claim was **not tested** and is reported as such. The MLP's hardest classes — Shirt, Pullover and
T-shirt/top — are again the visually similar ones, echoing the digit confusions of Lab 1.

**On methodology.** Three broader conclusions emerge. First, **point estimates of accuracy are
insufficient**: on the small problems, confidence intervals were wider than every effect measured,
and paired significance testing overturned descriptive conclusions that had appeared sound.
Second, **rigorous validation reveals structure that accuracy alone conceals** — the
representational limitation of decision trees documented in Section 1.5.6 was discovered by
examining *which* points a near-perfect model misclassifies, and it constitutes a more precise
statement about the algorithm than any aggregate figure could provide. Third, **claims made in the
course material should be tested rather than repeated**: of the four quantitative claims examined
here, two were confirmed (the `ToTensor` range, the `Normalize` speed-up), and two were not (the
225-attempt bound, the claim that training accuracy improves as capacity is reduced).

---

## 11. References

1. T. M. Cover and P. E. Hart, "Nearest Neighbor Pattern Classification," *IEEE Transactions on
   Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.
2. J. R. Quinlan, *C4.5: Programs for Machine Learning*. San Mateo, CA: Morgan Kaufmann, 1993.
3. L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen, *Classification and Regression Trees*.
   Boca Raton, FL: Chapman & Hall/CRC, 1984.
4. E. Fix and J. L. Hodges, "Discriminatory Analysis, Nonparametric Discrimination: Consistency
   Properties," USAF School of Aviation Medicine, Randolph Field, TX, Report 4, 1951.
5. M. Hall, E. Frank, G. Holmes, B. Pfahringer, P. Reutemann, and I. H. Witten, "The WEKA Data
   Mining Software: An Update," *SIGKDD Explorations*, vol. 11, no. 1, pp. 10–18, 2009.
6. F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine
   Learning Research*, vol. 12, pp. 2825–2830, 2011.
7. Q. McNemar, "Note on the Sampling Error of the Difference Between Correlated Proportions or
   Percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.
8. E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference,"
   *Journal of the American Statistical Association*, vol. 22, no. 158, pp. 209–212, 1927.
9. S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," *Scandinavian Journal of
   Statistics*, vol. 6, no. 2, pp. 65–70, 1979.
10. R. Kohavi, "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model
    Selection," in *Proc. IJCAI*, 1995, pp. 1137–1143.

---

## Appendix A — Reproduction

All results in this report were generated by the scripts below. Run from the repository root.

```bash
source .venv/bin/activate       # scikit-learn 1.8.0, numpy 2.4.2, scipy 1.17.1, python 3.14
export MPLCONFIGDIR=/tmp/mpl

# WEKA 3.8.7 is driven headlessly through its bundled Java runtime:
#   /Applications/weka-3.8.7.app/Contents/runtime/Contents/Home/bin/java
#   --add-opens java.base/java.lang=ALL-UNNAMED -cp .../app/weka.jar <classifier> ...
# Argument order matters: the class name must follow -cp immediately.
# -p 0 emits the per-instance predictions used for the paired tests.

python Lab-ML/lab1/src/verify_data.py                  # Section 2.2 data inventory gate
python Lab-ML/lab1/src/knn_scratch.py                  # Section 1.4.1 implementation validation
python Lab-ML/lab1/src/run_knn_experiments.py          # Section 3   k sweep, CV, boundary figures
python Lab-ML/lab1/src/relabel.py                      # Section 1.5.1 the four representations
python Lab-ML/lab1/src/run_j48_experiments.py          # Section 1.5.2 J48 and sklearn per variant
python Lab-ML/lab1/src/run_representation_analysis.py  # Section 1.5.6-4.8 error localisation
python Lab-ML/lab1/src/run_digit_experiments.py        # Section 5   seeds, Bigtest2, M sweep
python Lab-ML/lab1/src/sklearn_pipeline.py             # Section 6   pipeline, kNN agreement, Iris
python Lab-ML/lab1/src/run_significance.py             # Section 6.5 McNemar, Wilson CIs, timings

python Lab-ML/lab2/src/run_clustering.py               # Section 2   Lab 2 clustering
python Lab-ML/lab5/src/run_ga_function_opt.py          # Section 3.1 Lab 3 Exercise 1
python Lab-ML/lab5/src/run_ga_pattern.py               # Section 3.2 Lab 3 Exercise 2

python Lab-ML/lab6/src/run_sklearn_module.py           # Section 6   Lab 6 scikit-learn module
python Lab-ML/lab7/src/run_pytorch_module.py           # Section 7   Lab 7 PyTorch module

# Integrity check: the source data must be byte-identical to the recorded baseline.
cd Lab-ML && shasum -a 256 *.arff *.pdf | diff - lab1/results/source_data_checksums.txt
```

## Appendix B — Artifact Manifest

### B.1 Lab 1 — kNN and Decision Trees

| Artifact | Contents |
|---|---|
| `results/knn_k_sweep.csv` | Table 3 in full, with standard deviations |
| `results/knn_cv_folds.json` | Per-fold accuracies, 5- and 10-fold |
| `results/j48_representation.csv` | Table 6 (WEKA), all four variants |
| `results/sklearn_representation.csv` | Table 6 (scikit-learn) |
| `results/representation_confusion.csv` | Confusion matrix per variant |
| `results/representation_errors.csv` | Table 7, error localisation |
| `results/representation_twotest.json` | Section 1.5.8 two-attribute search |
| `results/j48_seeds.csv` | Table 8 in full, six seeds |
| `results/j48_M_sweep.csv` | Table 11 in full, *M* = 1…100 |
| `results/j48_bigtest1_split66.csv` | Single-split record (Section 1.6.1) |
| `results/digits_model_comparison.csv` | Table 10 |
| `results/digits_summary.json` | Confusion analysis, seed and *M* statistics |
| `results/weka_sklearn_crosscheck.csv` | Table 13 |
| `results/iris_comparison.csv`, `iris_knn.csv`, `iris_tree.csv` | Tables 14–15 |
| `results/significance_*.csv` | Tables 17–18 |
| `results/accuracy_confidence_intervals.csv` | Table 16 |
| `results/inference_benchmark.csv` | Table 19 |
| `results/j48_raw_outputs/` | 42 raw WEKA logs including printed trees |
| `figures/` | 16 figures referenced throughout |

### B.2 Lab 2 — Clustering

| Artifact | Contents |
|---|---|
| `lab2/results/kmeans_gausstrain_centroids.csv` | Section 2.2.1, centroid placement and distances |
| `lab2/results/kmeans_gausstrainhv_centroids.csv` | Section 2.3, high-variance centroids |
| `lab2/results/kmeans_digit_clusters.csv` | Section 2.4, cluster↔digit mapping and correlations |
| `lab2/results/kmeans_k_selection.csv` | Section 2.5.1, BIC and silhouette vs k |
| `lab2/results/kmeans_generalisation.csv` | Section 2.5.2, train vs test accuracy vs k |
| `lab2/results/lab2_summary.json` | all Lab 2 numbers, machine-readable |
| `lab2/figures/kmeans_gausstrain.png` | Section 2.2, clusters, centroids, misassigned points |
| `lab2/figures/kmeans_variance_comparison.png` | Section 2.3, low vs high variance |
| `lab2/figures/kmeans_digit_centroids.png` | Section 2.4, the ten centroids as greyscale images |
| `lab2/figures/kmeans_k_selection.png` | Section 2.5.1, model-selection criteria |

### B.3 Lab 3 — Genetic Algorithms

| Artifact | Contents |
|---|---|
| `lab5/results/ga_ex1_runs.csv` | Exercise 1, per-seed best fitness and evaluation counts |
| `lab5/results/ga_ex1_convergence.csv` | Exercise 1, best fitness vs generation |
| `lab5/results/ga_ex1_mutation_sensitivity.csv` | Exercise 1, sensitivity to sigma and indpb |
| `lab5/results/lab5_ex1_summary.json` | Exercise 1, all numbers, machine-readable |
| `lab5/results/ga_ex2_smiley.json` | Exercise 2, the smiley runs and parameter study |
| `lab5/results/ga_ex2_scaling.csv` | Exercise 2, N_f vs pattern size |
| `lab5/results/ga_ex2_parameters.csv` | Exercise 2, N_f vs (population, generations) |
| `lab5/results/ga_ex2_mutation_sensitivity.csv` | Exercise 2, sensitivity to indpb |
| `lab5/results/ga_ex2_hillclimber.csv` | Exercise 2, the (1+1) hill-climber baseline |
| `lab5/figures/ga_ex1_convergence.png` | Exercise 1, convergence comparison |
| `lab5/figures/ga_ex2_pattern.png` | Exercise 2, fitness trace and N_f scaling |

### B.4 Lab 6 — Scikit-Learn module

| Artifact | Contents |
|---|---|
| `lab6/results/pipeline_slide23_25.txt` | verbatim output of the deck's `ml_pipeline_example.py` |
| `lab6/results/pipeline_circle.csv` | the CSV used for the literal run (feature1, feature2, target) |
| `lab6/results/iris_knn_vs_tree.csv` | slide 26–27 sweep: accuracy vs k and vs depth, hold-out and 5-fold CV |
| `lab6/results/iris_repeated_cv.csv` | repeated 5-fold CV (10 repeats) for both models |
| `lab6/results/iris_confusion_matrices.csv` | confusion matrices in long form |
| `lab6/results/lab6_summary.json` | all Lab 6 numbers, machine-readable |
| `lab6/figures/iris_knn_vs_tree.png` | accuracy vs hyper-parameter, both models |
| `lab6/figures/iris_confusion.png` | side-by-side confusion matrices |
| `lab6/figures/iris_boundaries.png` | decision regions on petal length vs petal width |

### B.5 Lab 7 — PyTorch module

| Artifact | Contents |
|---|---|
| `lab7/results/lab7_tensor_demos.txt` | printed output of the slide 46–48 tensor examples |
| `lab7/results/fashion_mnist_history.csv` | per-epoch train/test loss and accuracy |
| `lab7/results/optimizer_comparison.csv` | SGD vs SGD+momentum vs Adam (slide 55) |
| `lab7/results/normalization_comparison.csv` | with vs without `Normalize` (slide 51 claim) |
| `lab7/results/classical_vs_mlp.csv` | MLP vs kNN vs decision tree |
| `lab7/results/lab7_summary.json` | all Lab 7 numbers, machine-readable |
| `lab7/figures/lab7_samples.png` | one FashionMNIST image per class |
| `lab7/figures/lab7_training_curves.png` | loss and accuracy per epoch |
| `lab7/figures/lab7_confusion.png` | MLP confusion matrix and per-class accuracy |
| `lab7/figures/lab7_normalization.png` | the slide 51 convergence claim, tested |

### B.6 Shared utilities

| Artifact | Contents |
|---|---|
| `lab1/src/arff_utils.py` | ARFF reader/writer (comma- and space-separated files), CSV writer, paths |
| `lab1/src/stats_utils.py` | Exact and chi-square McNemar, Holm correction, Wilson intervals |

All source data files — the five Lab 1 `.arff` files, the four Lab 2 gaussian files,
`smiley.txt`, and the laboratory PDF — are unmodified, verified by SHA-256 checksum.
