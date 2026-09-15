# k-Nearest Neighbours and Decision Trees
## A Comparative Study of Local and Global Classifiers, and the Role of Input Representation

**Course:** Machine Learning
**Assignment:** Laboratory 1 — k-Nearest Neighbours and Decision Trees
**Source material:** *ML2026_lab1_KNN_DecTrees-merged*, pp. 1–27

---

### Abstract

This report presents an experimental investigation of two classical supervised learning
algorithms — the *k*-nearest neighbour classifier (kNN) and the *univariate decision tree*
(C4.5/J48) — applied to two problems of contrasting difficulty: a synthetic two-dimensional
geometric discrimination task, and a 104-dimensional optical digit recognition task.

Three questions are addressed. First, how does *k* control the bias–variance trade-off, and how
should it be selected in principle? Second, how does the **representation** of the input space
change the accuracy and complexity of a decision tree, and can a learned representation yield a
trivially small tree at perfect accuracy? Third, how does the regularisation parameter *M*
(minimum samples per leaf) control overfitting, and does its observed behaviour match the
textbook account?

The principal findings are as follows. On the synthetic task, kNN attains its optimum at
*k* = 1 (91.0 % on the held-out test set), and for *k* ≥ 51 the classifier degenerates *exactly*
to the majority-class baseline — an unusually clean empirical demonstration of the underfitting
regime. Of four input representations tested, only the engineered feature
*u* = *x*² + *y*² produces a demonstrably significant improvement, collapsing the decision tree
to a single split with two leaves at 100 % test accuracy; a paired McNemar test confirms that the
remaining representational variants are **not** distinguishable from chance at this sample size.
On the digit task, kNN outperforms the decision tree by 2.9 percentage points
(97.6 % vs 94.7 %, *p* ≈ 10⁻¹⁸), overturning the theoretical expectation that the curse of
dimensionality would penalise the local method. Finally, the regularisation study confirms the
existence of a broad optimum at *M* = 4, but **contradicts** the common claim that training
accuracy improves as capacity is reduced: it decreases monotonically from 99.75 % to 89.89 %.

A further result emerged from rigorous validation rather than being sought: a decision tree
**cannot represent a decision threshold that does not occur in its training data**. This
representational limit, not any deficiency of the feature, accounts for the 6 residual
misclassifications of the otherwise optimal *u* model on the dense reference grid.

---

## 1. Introduction

### 1.1 Background

Supervised classification seeks a function *f* : ℝ^d → {1, …, C} inferred from a finite sample of
labelled observations. Two families of algorithms occupy opposite ends of a fundamental
trade-off.

**Instance-based (lazy) methods** such as kNN store the training sample and defer all computation
to prediction time. The decision boundary they induce is *local*: it is determined by the
Voronoi tessellation of the training points, and its resolution is therefore limited by local
sample density.

**Model-based (eager) methods** such as decision trees construct an explicit, compact hypothesis
at training time. The boundaries they induce are *global* and, for univariate splits, restricted
to be axis-aligned. A curved or oblique boundary must consequently be approximated by a
staircase of axis-aligned segments.

These structural differences predict different behaviour, and the central purpose of this
laboratory is to test those predictions experimentally rather than to accept them on authority.

### 1.2 The role of representation

A decision tree partitions the feature space by testing one attribute at a time. It follows that
the *encodability* of a concept depends on how the input is presented. A conjunction of
axis-aligned constraints is trivially expressible; an oblique or curved constraint is not. This
motivates the second theme of this report: holding the data points fixed and varying only the
representation, to isolate the effect of representation on tree size and accuracy.

### 1.3 Objectives

1. Implement kNN from first principles and validate it against a reference implementation.
2. Characterise the dependence of accuracy on *k* and establish a methodologically sound
   procedure for selecting it.
3. Quantify how four different input representations affect decision-tree accuracy, size, and
   *error localisation*.
4. Determine whether a decision tree can achieve perfect accuracy with a minimal (ideally
   single-node) hypothesis, and whether that result generalises.
5. Study overfitting control via the leaf-size parameter *M* on a ten-class problem.
6. Establish the statistical significance of all comparisons rather than relying on point
   estimates.
7. Reproduce the laboratory pipeline in a second ecosystem (scikit-learn) as an independent
   cross-check on the WEKA results.

### 1.4 Report structure

Section 2 defines the problems and describes datasets and methods. Section 3 reports the kNN
experiments. Section 4 examines representation effects. Section 5 treats the digit
classification task and overfitting control. Section 6 reports the cross-implementation
validation. Section 7 presents significance testing and computational cost. Section 8 states the
limitations, and Section 9 concludes.

---

## 2. Materials and Methods

### 2.1 Problem definitions

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

### 2.2 Datasets

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

### 2.3 Algorithms

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
so standardisation is likewise unnecessary there (though see Section 5.4 for why this does *not*
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
| `-U` | `ccp_alpha=0.0` | Request unpruned tree (see the caveat in Section 5.5) |
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

### 2.4 Software and reproducibility

WEKA 3.8.7 was driven headlessly through its bundled Java runtime. scikit-learn 1.8.0,
NumPy 2.4.2, and SciPy 1.17.1 were used on Python 3.14. All random seeds are recorded, no
source data file was modified, and byte-level integrity was verified by SHA-256 checksum before
and after the experiments. The complete pipeline is reproducible from the scripts listed in
Appendix A.

---

## 3. Exercise 1 — k-Nearest Neighbours

### 3.1 Validation of the implementation

Before any scientific claim, the from-scratch implementation was validated against two reference
implementations:

* the pairwise distance function agrees with `scipy.spatial.distance.cdist` to within
  1.39 × 10⁻¹⁴ (floating-point round-off);
* the classifier's predictions agree **point-for-point** with scikit-learn's
  `KNeighborsClassifier` on all 100 test observations at every value of *k* tested
  (*k* = 1, 3, 5, 7, 15, 31).

Any subsequent difference between the two therefore reflects the experiment, not a defect in the
implementation.

### 3.2 Accuracy as a function of *k*

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

### 3.3 Selection of *k*

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

### 3.4 Behaviour on the dense reference grid

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

## 4. Exercise 1b — The Effect of Input Representation

### 4.1 Experimental design

Four representations were constructed from the *same* points, so that any difference in outcome
is attributable to representation alone:

| Variant | Attributes | Labels |
|---|---|---|
| `xy` | *x*, *y* | original (circle) |
| `sq` | *x*, *y* | inner region replaced by the side-1.76 square |
| `zt` | *z* = *x*², *t* = *y*² | original (circle) |
| `u` | *u* = *x*² + *y*² | original (circle) |

### 4.2 Results

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

### 4.3 Is the square inner region easier to learn? (p. 7)

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

### 4.4 Does the (*x*², *y*²) representation help? (p. 8)

**Prediction.** Yes. In (*z*, *t*) coordinates the circle becomes the exact linear constraint
*z* + *t* ≤ 1. This is confirmed numerically:

```
max(z + t | c) = 0.999033       min(z + t | q) = 1.000280
```

— perfect separation, and identical to the *x*² + *y*² figures of Section 2.1.1 by construction.

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

The reason is the same structural limitation as in Section 4.3: **a decision tree tests one
attribute at a time, and *z* + *t* is not an attribute.** The exact straight-line boundary must
still be approximated by an axis-aligned staircase. The transformation helps only insofar as the
staircase in (*z*, *t*) aligns better with the true boundary than the one in (*x*, *y*). The full
collapse requires supplying the *sum* as a feature — see Section 4.5.

### 4.5 Can perfect accuracy be achieved with a minimal tree? (p. 9)

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

### 4.6 Does the perfect result generalise?

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

### 4.7 Where do the errors actually occur?

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

### 4.8 The optional two-attribute variant

Finally, the *z* and *t* attributes were supplied as two separate features (rather than
pre-summed) to test whether a one-node tree is achievable that way. It is not:

* the resulting tree has **depth 4 and 6 leaves** (7 leaves fully grown), with test accuracy
  0.89;
* an exhaustive search over *z*, *t*, *z* + *t*, *z* − *t*, max(*z*, *t*), and min(*z*, *t*)
  shows that **only the sum *z* + *t* is separable by a single threshold**.

This settles the question definitively: a tree cannot evaluate a sum of attributes, so separating
the components of *u* destroys the very property that made *u* effective.

---

## 5. Exercise 2 — Decision Trees on the Digit Data

### 5.1 Seed sensitivity of the 66 % split (pp. 12–13)

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

### 5.2 External validation on `Bigtest2`

The mean accuracy on the independent test set is 0.9462, against 0.9622 on the internal split — a
gap of **1.6 pp**. This is the honest generalisation penalty, and its causes are identifiable:

* `Bigtest2` is **exactly balanced** (501 per digit) whereas `Bigtest1` is imbalanced (525–739).
  Training on the imbalanced subset and testing on a balanced set alters the effective class
  priors.
* `Bigtest2` is a **pre-processed derivative**, as its own relation name records
  (`digit-weka.filters.unsupervised.attribute.Remove-R105-128`).

The methodological conclusion is that an internal random split of the training file
systematically **overstates** performance relative to a genuinely independent sample.

### 5.3 Error analysis

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

### 5.4 Comparison with the nearest-neighbour classifier

**Table 10 — Model comparison on the digit task**, means across six seeds.

| Model | Internal test | `Bigtest2` (external) |
|---|---|---|
| J48, *M* = 2 | 0.96216 ± 0.00352 | 0.94611 ± 0.00200 |
| scikit-learn tree, entropy, *M* = 2 | 0.95939 ± 0.00428 | 0.94674 ± 0.00244 |
| **IBk (kNN), *k* = 1** | **0.98372 ± 0.00171** | **0.97588 ± 0.00125** |
| **IBk (kNN), *k* = 5** | **0.98389 ± 0.00127** | 0.97542 ± 0.00089 |

**The nearest-neighbour classifier outperforms the decision tree by 2.2 pp on the internal split
and 3.0 pp on the external test set**, with a smaller seed-to-seed spread. This is the *reverse*
of the circle problem (Section 3.2) and it contradicts the expectation that the curse of
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
  method is also **more robust to the dataset shift** described in Section 5.2 — consistent with
  its reliance on local similarity rather than on globally tuned thresholds.

The correct conclusion is therefore a refinement of the standard heuristic rather than a
refutation of it: high dimensionality makes kNN *data-hungry*, and this dataset is large enough to
satisfy that appetite. The tree retains a decisive advantage in computational cost
(Section 7.3).

### 5.5 Overfitting control via *M* (pp. 14–16)

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

### 5.6 Why *M* = 1 does not yield 100 % training accuracy (p. 15)

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
   5-leaf tree as the pruned default (Section 4.2). scikit-learn's `ccp_alpha = 0` is genuinely
   unpruned. The two "unpruned" modes are therefore **not** equivalent and must not be treated
   as such.
2. **Greedy search is not globally optimal — plausible in general, but not the binding
   constraint here.** Since a fully grown tree reaches exactly 100 %, greedy induction was
   evidently able to isolate every training observation.
3. **The attribute set may be insufficient — refuted.** With 104 binary attributes, a fully grown
   tree memorises all 6024 observations.

### 5.7 The *k* / *M* correspondence

*k* in kNN and *M* in J48 are the same concept viewed from opposite directions. Small *k* gives
low bias and high variance with zero training error; large *k* degenerates to the majority class.
Small *M* gives high capacity and near-perfect training fit; large *M* produces leaves too coarse
to separate the classes. Both are selected on validation data, and both exhibit a broad rather
than sharp optimum (*k*\* = 1, *M*\* = 4). The mechanisms differ — kNN regularises by *smoothing a
local neighbourhood*, a tree by *refusing to split* — but the bias–variance trade-off they trace
is the same.

---

## 6. Cross-Implementation Validation with scikit-learn

### 6.1 The reference pipeline (pp. 23–25)

The pipeline prescribed in the laboratory (load → two sequential `train_test_split` calls
producing a stratified 70/15/15 partition → fit → evaluate with `accuracy_score`,
`classification_report`, `confusion_matrix`) was implemented and generalised to read the ARFF
data directly. Both the ARFF path and a literal CSV round-trip via `pandas.read_csv` were
exercised:

```
Training set: (70, 2)      Validation set: (15, 2)      Test set: (15, 2)
Validation accuracy: 0.9333
Test accuracy:       0.9333        confusion [[7 1] [0 7]]
External test (circletest.arff): 0.8600   confusion [[50 10] [4 36]]
CSV round-trip test accuracy: 0.9333
```

The 15-observation validation and test sets make these figures extremely coarse — a single
observation is 6.7 pp — which is precisely why the principal kNN analysis (Section 3) uses LOO
and 5/10-fold CV over all 100 observations instead.

### 6.2 Independent verification of the kNN implementation

scikit-learn's `KNeighborsClassifier` and the from-scratch implementation agree **point-for-point
at every value of *k*** (100/100 test observations for *k* = 1, 3, 5, 7, 9, 11, 13, 15, 21, 31,
51). This is the strongest available evidence that the implementation of Section 2.3.1 is
correct, including its tie-breaking behaviour.

### 6.3 Agreement on the digit task

**Table 13 — WEKA versus scikit-learn, matched configurations** (`results/weka_sklearn_crosscheck.csv`).

| Dataset | Hyper-parameter | WEKA | scikit-learn | Δ (pp) |
|---|---|---|---|---|
| digits (`Bigtest1`→`Bigtest2`) | M = 1 | 0.9515 | 0.9559 | +0.44 |
| digits | M = 2 | 0.9531 | 0.9525 | −0.06 |
| digits | M = 3 | 0.9535 | 0.9553 | +0.18 |
| digits | M = 5 | 0.9523 | 0.9513 | −0.10 |
| digits | M = 10 | 0.9441 | 0.9457 | +0.16 |
| circle `(x,y)` | pruned M=2 | 0.880 | 0.870 | −1.00 |
| circle square-inner | pruned M=2 | 0.920 | 0.960 | +4.00 |
| circle `(x²,y²)` | pruned M=2 | 0.880 | 0.890 | +1.00 |
| **circle `u` = `x²+y²`** | **pruned M=2** | **1.000** | **1.000** | **0.00** |

**On the large digit datasets the agreement is excellent** — every difference is below 0.5 pp —
which constitutes strong mutual validation of the two implementations. On the 100-point circle
problem the differences reach 3–4 pp, but this must be interpreted in proportion: with 100 test
observations, 4 pp is *four observations*. The cause is the documented divergence in split
tie-breaking and pruning, which produces structurally different trees (5 versus 9 leaves). The
defensible claims are about the *shape* of trends and the *ordering* of representations, both of
which the two toolchains reproduce; exact per-observation agreement was never expected.

### 6.4 Iris classification (pp. 26–27)

**Table 14 — Iris: kNN versus decision tree.**

| Model | Selected hyper-parameter | 5-fold CV | Hold-out (30 %) | Leaves |
|---|---|---|---|---|
| kNN | *k* = 7 | 0.980 ± 0.016 | 0.956 | — |
| Decision tree | `max_depth` = 3 | 0.973 ± 0.025 | 0.978 | 5 |

Both models perform strongly, and the confusion matrices (`figures/iris_confusion.png`) explain
why: **the Setosa class is perfectly separated by both models**, being linearly separable from
the other two species on petal dimensions. Every residual error is *versicolor* versus
*virginica*, which overlap in feature space and are not separable by any axis-aligned rule a
5-leaf tree can express. This is why the tree's cross-validated accuracy saturates at depth 3 and
does not improve with greater depth (Table 15).

**Table 15 — Decision tree accuracy versus depth.**

| `max_depth` | 5-fold CV | Leaves |
|---|---|---|
| 1 | 0.667 | 2 |
| 2 | 0.933 | 3 |
| **3** | **0.973** | **5** |
| 5 | 0.953 | 8 |
| unlimited | 0.953 | 8 |

The single visible discrepancy between the two models — the hold-out comparison (kNN 0.956 vs
tree 0.978) — amounts to one observation out of 45 and is not significant. The cross-validated
estimates, which use all 150 observations, place kNN marginally ahead.

---

## 7. Statistical Significance and Computational Cost

### 7.1 Confidence intervals

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

### 7.2 Paired significance testing

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

The central claim of Section 5.4 is thereby **established rather than asserted**: the
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

### 7.3 Computational cost

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

## 8. Limitations

1. **Small circle sample.** With 100 training and 100 test observations, accuracies are quantised
   in steps of 1 pp and confidence intervals reach 12.8 pp. The paired tests demonstrate the
   practical consequence: none of the tree-versus-tree representation differences is significant
   at this sample size.
2. **Seed variance in the 66 % split.** Internal test accuracy spans 0.958–0.968 across seeds
   (± 0.83 pp binomial uncertainty). Single-seed results should never be quoted in isolation.
3. **Internal versus external evaluation differ by 1.6 pp**, driven partly by class imbalance and
   partly by the pre-processing history of `Bigtest2`.
4. **The two toolchains are not implementable-equivalent.** In particular, WEKA's `-U` does not
   fully disable pruning (Section 5.6). Cross-tool comparisons therefore concern trends and
   orderings, not individual values. Section 7.2 confirms that J48 and scikit-learn's tree are
   statistically indistinguishable on the digits.
5. **Representational limit of decision trees.** A tree cannot emit a threshold absent from its
   training data (Section 4.6). This accounts for the 6 residual grid errors of the `u` model and
   would be removed by denser training coverage of the boundary.
6. **Boundary points excluded from grid accuracies.** 212 of 2601 grid points lie exactly on a
   boundary, where the assigned label is arbitrary. They are excluded from headline figures and
   reported separately, so that floating-point representation effects cannot be mistaken for
   modelling error.
7. **Timings are single-machine and single-threaded.** WEKA's Java timings include JVM startup and
   are not directly comparable; only scikit-learn timings are used for the cost comparison.
8. **No nested validation for the digit hyper-parameters.** *M*\* was selected on the external
   `Bigtest2` set, which is methodologically the same error as tuning *k* on the test set
   (Section 3.3). With 5010 observations the optimism is small — the *M* ∈ [1, 5] plateau spans
   only 0.4 pp — but the correct protocol would nest a validation split within `Bigtest1`.

---

## 9. Conclusions

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
boundary (80.3 % within one grid step of it), and no variant errs on the trivially expressible
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
representational limitation of decision trees documented in Section 4.6 was discovered by
examining *which* points a near-perfect model misclassifies, and it constitutes a more precise
statement about the algorithm than any aggregate figure could provide.

---

## References

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
python Lab-ML/lab1/src/knn_scratch.py                  # Section 3.1 implementation validation
python Lab-ML/lab1/src/run_knn_experiments.py          # Section 3   k sweep, CV, boundary figures
python Lab-ML/lab1/src/relabel.py                      # Section 4.1 the four representations
python Lab-ML/lab1/src/run_j48_experiments.py          # Section 4.2 J48 and sklearn per variant
python Lab-ML/lab1/src/run_representation_analysis.py  # Section 4.6-4.8 error localisation
python Lab-ML/lab1/src/run_digit_experiments.py        # Section 5   seeds, Bigtest2, M sweep
python Lab-ML/lab1/src/sklearn_pipeline.py             # Section 6   pipeline, kNN agreement, Iris
python Lab-ML/lab1/src/run_significance.py             # Section 7   McNemar, Wilson CIs, timings

# Integrity check: the source data must be byte-identical to the recorded baseline.
cd Lab-ML && shasum -a 256 *.arff *.pdf | diff - lab1/results/source_data_checksums.txt
```

## Appendix B — Artifact Manifest

| Artifact | Contents |
|---|---|
| `results/knn_k_sweep.csv` | Table 3 in full, with standard deviations |
| `results/knn_cv_folds.json` | Per-fold accuracies, 5- and 10-fold |
| `results/j48_representation.csv` | Table 6 (WEKA), all four variants |
| `results/sklearn_representation.csv` | Table 6 (scikit-learn) |
| `results/representation_confusion.csv` | Confusion matrix per variant |
| `results/representation_errors.csv` | Table 7, error localisation |
| `results/representation_twotest.json` | Section 4.8 two-attribute search |
| `results/j48_seeds.csv` | Table 8 in full, six seeds |
| `results/j48_M_sweep.csv` | Table 11 in full, *M* = 1…100 |
| `results/j48_bigtest1_split66.csv` | Single-split record (Section 5.1) |
| `results/digits_model_comparison.csv` | Table 10 |
| `results/digits_summary.json` | Confusion analysis, seed and *M* statistics |
| `results/weka_sklearn_crosscheck.csv` | Table 13 |
| `results/iris_comparison.csv`, `iris_knn.csv`, `iris_tree.csv` | Tables 14–15 |
| `results/significance_*.csv` | Tables 17–18 |
| `results/accuracy_confidence_intervals.csv` | Table 16 |
| `results/inference_benchmark.csv` | Table 19 |
| `results/j48_raw_outputs/` | 42 raw WEKA logs including printed trees |
| `figures/` | 16 figures referenced throughout |

The five source `.arff` files and the laboratory PDF are unmodified, verified by SHA-256
checksum.
