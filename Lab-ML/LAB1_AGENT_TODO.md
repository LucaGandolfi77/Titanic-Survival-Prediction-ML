# Lab 1 - kNN & Decision Trees: Agent Execution Plan

> **STATUS: EXECUTED.** Every step below was run on 2026-09-11.
> The results are written up in `lab1/REPORT.md`; all figures are in `lab1/figures/`
> and all numbers in `lab1/results/`. Three facts in this plan turned out to be
> **wrong on first writing** and are corrected in place (they are marked
> **[CORRECTED]** below) - they are recorded rather than deleted because the
> corrections changed the conclusions:
>
> 1. **[CORRECTED] Class semantics were inverted.** `c` is the **circle interior**
>    and `q` is the **square ring** (not the other way round). Verified with zero
>    errors on both training and test files.
> 2. **[CORRECTED] WEKA's `-U` does not fully disable pruning.** It stops subtree
>    *raising* but confidence-threshold pruning still applies. `-U -M 1` still
>    returns the same 5-leaf tree as the pruned default. Only sklearn's
>    `ccp_alpha=0` grows a fully unpruned tree.
> 3. **[CORRECTED] The `u = x^2+y^2` feature gives a genuine 1-node stump**, not
>    merely a "crafted" one: `u <= 1` separates the classes exactly and generalises
>    to all 2601 grid points at 100% accuracy.
>
> **All steps are now complete, including the optional `3.7.1b`.** That step was
> settled empirically in a second pass (`src/run_representation_analysis.py`): the
> two-feature `(z, t)` representation yields a depth-4, 6-leaf tree, not a one-node
> stump, and an exhaustive search over `z`, `t`, `z+t`, `z-t`, `max(z,t)`, `min(z,t)`
> confirms that only the sum `z + t` is single-threshold separable.
>
> Two further gaps flagged in REPORT.md 7.5 were also closed in that pass: paired
> McNemar significance testing (`src/run_significance.py`) and measured inference
> timings. The significance testing **overturned** the descriptive reading of the
> circle representation results - see REPORT.md sections 7.2 and 4.5.

> **Source:** `Lab-ML/ML2026_lab1_KNN_DecTrees-merged.pdf` (91 pages, merged deck).
> **In-scope part:** pages **1-16** (Lab 1: k Nearest Neighbors + Decision Trees, WEKA)
> and pages **17-27** (Scikit-Learn module, incl. the Iris exercise).
> The rest of the PDF is **out of scope** for this plan: pages 28-43 = Lab 2 Clustering,
> 44-56 = PyTorch, 57-91 = Lab 5/6 Evolutionary Computation.
>
> **This file is a machine-readable instruction set for an LLM coding agent.**
> Work top to bottom. Do not skip a step because it looks obvious: the whole point of
> the lab is to *observe* how accuracy moves as k, M, and the representation change,
> and to explain *why*. Record every number you produce.

---

## 0. Mission statement

Build and evaluate two classifiers on a synthetic 2-D two-class problem (a circle vs. a
square-minus-circle), then on a 104-dimensional hand-written-digit problem:

1. **kNN** - tune `k`, explain the bias/variance behaviour, pick `k` with a validation set.
2. **Decision tree (J48/C4.5)** - study how the *representation* of the input space
   changes the size of the tree and the achievable accuracy.
3. **Overfitting control** - study the parameter `M` (min samples per leaf) on the digit
   data and find the value that maximises generalisation.
4. **Scikit-Learn** - reproduce the same two learning algorithms in Python and compare
   with the WEKA results.

**Final deliverable:** a written report (`Lab-ML/lab1/REPORT.md`) plus all figures,
generated datasets and scripts that produced them.

---

## 1. Ground truth: the problem definition (read this before writing any code)

Two classes are used throughout. **The mapping was verified against the files
(`verify_data.py` + `relabel.py`), and it is easy to get backwards:**

* **Class `c`** = **inside the circle** of radius 1 centred at the origin:
  `C(x,y)  <=>  x^2 + y^2 <= 1`
* **Class `q`** = the **square ring** - inside the square of side `L = 2.5` centred at
  the origin (`|x| <= 1.25` and `|y| <= 1.25`) **with the circle removed**:
  `Q(x,y)  <=>  (|x| <= 1.25  and  |y| <= 1.25)  and  not C(x,y)`

> **Verified facts, not assumptions.** On `circletrain.arff` and `circletest.arff` the
> rule `c <=> x^2+y^2 <= 1` reproduces the file labels with **zero errors**:
> `max(x^2+y^2 | c) = 0.9990` and `min(x^2+y^2 | q) = 1.0003` for the training set.
> On `circleall.arff` it agrees on 2595/2601 points; the 6 disagreements all lie
> *exactly* on the circle (`|x^2+y^2-1| < 1e-9`) and are boundary-labelling artefacts,
> not label noise.

So the domain is the square `[-1.25, 1.25] x [-1.25, 1.25]`, the circle is a disc in the
middle of it, and the four "corner" regions are class `q`.

**Tie policy for points exactly on a boundary** (`x^2+y^2 == 1`, or `|x| == 1.25`):
a dense grid will contain such points, and the label there is arbitrary. Implement the
inside-the-disc test as `<=` (so a boundary point is class `c`) and, in step 3.4.3, exclude
points with `|x^2+y^2 - 1| < 1e-9` or `|max(|x|,|y|) - 1.25| < 1e-9` from the label-agreement
check instead of silently counting them as errors.

**Integrity check.** `lab1/results/source_data_checksums.txt` holds the SHA-256 of every
source `.arff` and the PDF, taken before any work started. Re-verify it at the end to
prove the originals were never modified.

**Why the labels exist in the training files:** the `.arff` files are pre-labelled with
exactly this rule. Treat them as ground truth. **Never** re-label them from your own
geometry code and then claim you "cleaned" them - if your rule and the file disagree,
your rule is wrong (or you found a genuine label noise case worth reporting).

### 1.1 Data inventory (all paths relative to the repo root)

| File | Rows | Attributes | Role |
|---|---|---|---|
| `Lab-ML/circletrain.arff` | 100 `c`+`q` (53 `c`, 47 `q`) | `x`, `y` real; `class {c,q}` | training set |
| `Lab-ML/circletest.arff` | 100 (60 `c`, 40 `q`) | same | held-out test / validation |
| `Lab-ML/circleall.arff` | 2601 | same, step 0.05 grid over `[-1.25, 1.25]^2` (51x51) | dense grid, "ground truth" |
| `Lab-ML/Bigtest1_104.arff` | 6024 (`0`:557, `1`:525, `2`:566, `3`:739, `4`:601, `5`:619, `6`:599, `7`:533, `8`:611, `9`:674) | `f0..f103` numeric + `class {0..9}` | digit training pool |
| `Lab-ML/Bigtest2_104.arff` | 5010 exactly balanced (501 per digit) | same | held-out digit test set |

Sanity check the class balance before modelling - a 53/47 vs 60/40 split means accuracy
is a reasonable metric here, but compute the **majority-class baseline** anyway
(circle test: `60/100 = 60%`) so every accuracy number has context.

Digit baseline context: 10 classes, so the trivial baseline is ~10%; on `Bigtest2` it is
exactly 10% by construction.

### 1.2 Hard rules for the agent

* **No leakage.** `circletest.*`/`circleall.*` are never used to fit anything.
  `Bigtest2_104.arff` is touched only as a final test set.
* **No silent data edits.** Every new dataset is a *new file*; the originals stay untouched.
* **Every number must be reproducible.** Fix and record all random seeds.
* **No fabricated results.** If WEKA is unavailable, say so, produce the scikit-learn
  results, and mark the WEKA-only rows of the results table as `NOT RUN (reason)`.
  Never invent accuracy values.
* **Report negative results.** "k=15 was not better" is a finding; so is "my tree was
  bigger than expected".

---

## 2. Environment setup

- [x] **2.1 Check what is already installed.**
  ```bash
  python3 -c "import sklearn, pandas, numpy, matplotlib, scipy; print(sklearn.__version__)"
  java -version   # needed for WEKA
  ls /Applications | grep -i weka ; which weka
  ```
- [x] **2.2 Install Python deps if missing** (repo has a `.venv`; prefer it):
  ```bash
  source .venv/bin/activate 2>/dev/null || true
  pip install -r requirements.txt 2>/dev/null || \
  pip install scikit-learn numpy scipy pandas matplotlib
  ```
  The PDF (p.21) requires: `scikit-learn`, `numpy`, `scipy`, `joblib`, `threadpoolctl`,
  and `matplotlib` for plots.
- [x] **2.3 WEKA.** Download WEKA (≥3.8) from https://waikato.github.io/weka-wiki/downloading_weka/
  and confirm it starts. The lab is written for the **WEKA Explorer GUI** and names
  specific widgets: `IBk` (= kNN), `J48` (= C4.5), `SimpleKMeans` (Lab 2, out of scope).
  > If running headless, use WEKA from the command line instead of the GUI; the exact
  > commands are given in step 4.3 and 6.4. Prefer the CLI when you can - it is
  > reproducible and reviewable.
- [x] **2.4 Create the working directory:**
  ```bash
  mkdir -p lab1/{data,figures,results,src}
  ```
- [x] **2.5 Loader.** Write `lab1/src/arff_utils.py` with:
  * `load_arff(path) -> (X: np.ndarray, y: np.ndarray, attrs: list[str], classes: list[str])`
    (parse it yourself with a small regex parser, or use `scipy.io.arff.loadarff`, or
    `pip install liac-arff`). Handle the leading comma / blank lines / `@data` marker.
  * `write_arff(path, X, y, attr_names, class_values, relation)` for step 5.1.
  * Verify against the table in 1.1: assert row counts **and** class counts match exactly.
- [x] **2.6 Verification gate.** Print shape, dtype, class counts and `x`/`y` min-max for
  all 3 circle files and both digit files. Do not proceed until these match section 1.1.

---

## 3. Exercise 1 - k Nearest Neighbors (PDF p.2-6)

**Algorithm to implement** (p.3): to label `X`, find the `k` nearest training samples by a
distance measure (Euclidean unless stated), and take a **majority vote**. `k` is **odd**
to avoid ties in the 2-class problem.

### 3.1 Implement kNN from scratch (do not start with sklearn here)

- [x] **3.1.1** Write `lab1/src/knn_scratch.py`:
  * `euclidean(A, B)` -> pairwise distances, **vectorised** (broadcasting), no Python loops
    over samples.
  * `knn_predict(X_train, y_train, X_query, k)` -> majority vote. Break any residual tie
    (e.g. equal counts) deterministically and log it.
  * `accuracy(y_true, y_pred)`.
  * A `__main__` that reproduces the scikit-learn result on the same inputs and asserts
    they agree point-for-point (this is your correctness proof).
- [x] **3.1.2** Label encoding: map `{c,q} -> {0,1}` with a **fixed documented** mapping
  and use it everywhere (report which class you call "positive").
- [x] **3.1.3** Scale check: `x` and `y` are both in `[-1.25, 1.25]`, so Euclidean distance
  is already fair - **no scaling needed** for the circle data. State this explicitly in
  the report (it matters for the digit data later, see 6.1).

### 3.2 Sweep k: train -> test

- [x] **3.2.1** For `k = 1, 3, 5, 7, 9, 11, 13, 15, 21, 31` (and ≥3 more values of your
  choice), fit on `circletrain.arff` and evaluate on `circletest.arff`.
- [x] **3.2.2** Store `k` vs accuracy in `lab1/results/knn_k_sweep.csv`
  (columns: `k, train_acc, test_acc, loo_acc` - `loo_acc` comes from 3.3).
- [x] **3.2.3** Plot `accuracy vs k` (train, LOO-validation, test on one axes) to
  `lab1/figures/knn_k_sweep.png`. Mark the chosen `k*`.
- [x] **3.2.4** Answer, with the plot as evidence:
  * At what `k` does test accuracy peak? What is the value?
  * Why does `k = 1` tend to overfit the training set (0 training error) yet do worse on
    held-out data? (bias-variance / decision-boundary roughness)
  * Why does very large `k` underfit? (the neighbourhood becomes the whole dataset and
    the prediction degenerates to the majority class)
  * Is the `train_acc` curve monotone in `k`? Why is `k=1` special?

### 3.3 Choosing the best k with a validation set (p.6, question 3)

- [x] **3.3.1** Implement **leave-one-out cross-validation** on `circletrain.arff`:
  for each of the 100 samples, predict it from the other 99 for every `k`. This is cheap
  (100x100 distance matrix) - do it properly with a precomputed distance matrix and
  `np.fill_diagonal(D, np.inf)`.
- [x] **3.3.2** Also implement **k-fold CV (k=5, 10)** with stratification and report how
  stable the chosen `k*` is across folds. LOO has high variance; showing both is a
  stronger answer.
- [x] **3.3.3** Compare the `k*` chosen by LOO/CV against the `k*` chosen by
  `circletest.arff`. Do they agree? Write a paragraph on **why using the test set to pick
  a hyper-parameter is methodologically wrong** (it makes the test set part of training
  and inflates the reported accuracy).
- [x] **3.3.4** Report the nested protocol you would use in a real project:
  split train into train/validation, tune on validation, then report once on test.

### 3.4 Ground-truth verification on `circleall.arff` (p.6, question 4)

- [x] **3.4.1** Run the `k*` classifier over the dense grid `circleall.arff`.
  Because the grid is dense, the predicted label at each grid point is effectively the
  classifier's decision surface.
- [x] **3.4.2** Produce `lab1/figures/knn_boundary.png`: one panel per `k` in
  `{1, 5, 15, 31}` showing the decision regions, the circle of radius 1 (`x^2+y^2=1`)
  and the square `|x|=|y|=1.25` overlaid. This single figure is the best evidence in the
  report - make it publication-quality (legend, equal aspect, colourbar-free).
- [x] **3.4.3** Quantify boundary quality: compute the **true** labels for all 2601 grid
  points from section 1's rule, assert they reproduce `circleall.arff`'s own labels
  (they should, up to any point sitting exactly on a boundary), then report
  `accuracy(true_labels, knn(k*) predictions)` on the full grid.
- [x] **3.4.4** Interpret: the true boundary is curved (arcs of the circle plus 4 straight
  square edges). *Global* methods like a decision tree must staircase a curve; *local*
  methods like kNN approximate it with piecewise-constant regions whose granularity is set
  by the local sample density. This contrast is the intellectual core of Lab 1 - it is
  re-used in Exercise 1b, so write it up well here.

### 3.5 Exercise 1b - representation #1: inner region is a SQUARE (p.7)

New problem: the inner region becomes a **square of side `l = 1.76` centred at the origin**
(`|x| <= 0.88` and `|y| <= 0.88`) *instead of* the circle. Outer square is unchanged.

- [x] **3.5.1** Write `lab1/src/relabel.py` that reads an `.arff`, recomputes every label
  from the new geometry, and writes a new `.arff` with **the same points** and the new
  labels. Generate all three: `lab1/data/{circletrain,circletest,circleall}_sq.arff`.
  The outer region stays class `q`; the inner square becomes class `c`.
- [x] **3.5.2** Verify the new geometry numerically and record the class balance:
  * `0.88^2 = 0.7744 < 1`, so the new square sits strictly inside the old circle -
    no `q` point can accidentally become `c`.
  * The points that change label are exactly those with `x^2+y^2 <= 1` **and**
    `max(|x|,|y|) > 0.88` (the four circular caps beyond the new square, e.g. near the
    axes at `(1, 0)`). Count them in the training, test and grid files and confirm the
    number is non-zero; a relabelling that changes nothing means you did not apply the
    new rule.
- [x] **3.5.3** **Predict before running:** will J48 do **better or worse** than on the
  circle version? Justify analytically (p.7's hint: write the inner region as a set of
  ANDed linear equations in x and y).
  > Expected reasoning: *axis-aligned* linear splits describe the inner region
  > `(|x| <= 0.88) AND (|y| <= 0.88)` **exactly** in 2 tests, whereas the circle
  > `x^2+y^2 <= 1` is not linearly separable in `(x,y)` and needs a staircase of splits.
  > So the square problem should be **easier**: smaller tree, higher accuracy.
  > Note the outer boundary is the same in both cases. Write your own version of this;
  > do not copy it as your conclusion if your experiment disagrees - report the
  > disagreement.
- [x] **3.5.4** Run J48 on the old data and the new data, **all other parameters equal**,
  and compare: accuracy (train + test), number of leaves, tree size (p.7 "Run J48 and check!").
  Use `-U` (unpruned) as well as the default pruned setting, because pruning can mask the
  representational difference.
  > **Why the square should be easier (the p.7 hint, worked out).** The inner region is
  > now `(|x| <= 0.88) AND (|y| <= 0.88)`, i.e. it is already a conjunction of axis-aligned
  > tests - exactly the language a decision tree speaks - while the circle is not linearly
  > separable in `(x,y)`. Concretely, with `z = x^2`, `t = y^2` the new problem is the XOR
  > of two axis-aligned rectangles (inner: `z <= 0.7744 AND t <= 0.7744`; the ring inside
  > the old circle: `z + t <= 1`), and XOR needs a depth-2 tree with the root testing one
  > rectangle and the children testing the other - about 3 nodes, versus a staircase of
  > roughly a dozen leaves for the circle in `(x,y)`. Derive this yourself and compare
  > against what WEKA actually reports.
- [x] **3.5.5** Record in `lab1/results/j48_representation.csv`.
  Also plot both trees (`weka.classifiers.trees.J48 -g` prints Graphviz DOT; render with
  `dot -Tpng`) or use the `tree.export_text`/`plot_tree` equivalent in sklearn.

### 3.6 Exercise 1b - representation #2: `z = x^2`, `t = y^2` (p.8)

- [x] **3.6.1** Build the transformed datasets: attributes `(z, t)` = `(x^2, y^2)`, same
  points, **same labels as the original circle problem** (`class` unchanged). Write
  `lab1/data/circle*_zt.arff` with only the two new attributes (drop `x`,`y`).
- [x] **3.6.2** Reason it out before running (again: the **original circle labels**): in
  `(z,t) = (x^2, y^2)` space the circle is `z + t <= 1`, which is an exact axis-aligned
  *linear* boundary, so a tree can represent the true rule with a single split on `z + t`
  (a depth-1 stump) instead of the deep staircase that `(x,y)` requires. The square bound
  `|x| <= 1.25` becomes `z <= 1.5625` and `t <= 1.5625`, also axis-aligned. Predict
  "much smaller tree, similar or better accuracy" and then test it.
- [x] **3.6.3** Run J48 on `(z,t)` and compare accuracy / tree size against the `(x,y)` run.
  Record in `lab1/results/j48_representation.csv`.
- [x] **3.6.4** Discuss the general lesson: **feature engineering changes the hypothesis
  space**. A tree is a piecewise-constant axis-aligned learner; give it features aligned
  with the true decision surface and the tree shrinks dramatically. Mention that this is
  a form of *inductive bias* exploitation, not "the algorithm getting smarter".

### 3.7 Exercise 1b - representation #3: 100% accuracy with one node (p.9)

- [x] **3.7.1** Find a single new variable `u = f(x, y)` such that J48 reaches **100%**
  accuracy with the smallest possible tree. This uses the **original circle labels**
  (not the square-inner-region labels of 3.5), so work out the rule first:
  * Class `c` = `x^2+y^2 <= 1`. Class `q` = `x^2+y^2 > 1` **and** `|x| <= 1.25` **and**
    `|y| <= 1.25` (all points of both classes lie in the square, so those last two
    conditions are automatically true on this data). Therefore the true rule is a
    **single threshold on the radius squared**:
    `u = x^2 + y^2`, `u <= 1 -> c`, `u > 1 -> q`.
  * So a **1-node stump** (`u <= 1`) is achievable and J48 should find it. Build
    `lab1/data/circle*_u.arff` with the single attribute `u` plus the unchanged `class`.
  * Report the actual tree J48 prints (expect roughly
    `u <= 1: c / u > 1: q`) and its training and test accuracy. If J48 returns a bigger
    tree, inspect why before writing anything - a pruning threshold or a floating-point
    boundary point are the usual culprits.
  * Sanity check the arithmetic on a few points by hand, e.g. `(0.5, 0.5) -> u = 0.5 -> c`;
    `(1.25, 1.25) -> u = 3.125 -> q`; `(0.9, 0.9) -> u = 1.62 -> q` (a corner point near
    the circle). State that `u` is a **domain-specific feature**: it encodes exactly the
    inductive bias the problem needs.
- [x] **3.7.1b (optional, stronger discussion)** Repeat with `z = x^2`, `t = y^2` as two
  separate attributes and compare tree sizes. For this *original* problem the two-feature
  version also collapses to a single split on `z + t`, so report both and note that the
  minimal tree here is genuinely depth-1 - the tree does not need to represent an XOR.
  (The XOR/rectangles reasoning belongs to the **relabelled square problem** of 3.5, not
  here - do not mix the two variants up in the report.)
- [x] **3.7.2** Run J48 on the `u` dataset several times (seeds). Report the tree size and
  accuracy on the 100 training points and on `circletest.arff`.
- [x] **3.7.3** Report hint from p.9: validate on `circleall.arff` and use
  *Visualize classifier errors* (right-click the result list in WEKA Explorer) to inspect
  which grid points are wrong. Recreate that view in matplotlib:
  `lab1/figures/errors_one_node.png` - correct points in light grey, misclassified in red.
- [x] **3.7.4** State clearly whether the perfect accuracy generalises to `circleall.arff`
  or only holds on the 100 training points.

---

## 4. Exercise 2 - Decision trees on digits (PDF p.10-13)

Data: `Bigtest1_104.arff` = 6024 patterns, each a **13x8 = 104-pixel** binary image of a
digit from a licence plate, flattened **row-wise**, with `class {0..9}`.

- [x] **4.1 Understand the encoding** (p.11): reshape a row to `(13, 8)` and print it as
  ASCII art to confirm orientation. Do this for the first `0` example reproduced in the
  PDF and compare with the 13x8 grid printed on p.11. Save
  `lab1/figures/sample_digit.png` with ~12 samples rendered as images.
- [x] **4.2 66% split run** (p.12): in the WEKA Explorer select
  `Test options -> Percentage split -> 66%`, choose `J48` with **default parameters**,
  and run. Observe the result summary and the printed tree.
  * Record: accuracy, Kappa, MAE/RMSE, size of tree, number of leaves, confusion matrix
    -> `lab1/results/j48_bigtest1_split66.csv`.
  * Save the tree (`Save model`) and the full output text so the reader can inspect it.
- [x] **4.3 5+ random seeds** (p.13): change `More options -> Random seed` to e.g.
  `1,2,3,4,5,42`, re-run with everything else fixed.
  * [x] 4.3.1 Tabulate accuracy per seed -> `lab1/results/j48_seeds.csv`.
  * [x] 4.3.2 Compute **mean and standard deviation** of accuracy across seeds. This is
    the number quoted later as "the average of the previously performed runs".
  * [x] 4.3.3 Plot a box/violin or strip plot of the 6 accuracies
    -> `lab1/figures/j48_seed_variance.png`.
  * [x] 4.3.4 Explain why the accuracy moves at all: `66%` split + different seed =
    different *training* subset, and the digit classes contain visually ambiguous
    specimens. Estimate the 95% CI of the accuracy (e.g. `1.96*sqrt(p(1-p)/n)`) and check
    whether the observed spread is consistent with pure sampling noise.
- [x] **4.4 External test set** (p.13): choose `Supplied test set` ->
  `Bigtest2_104.arff`, re-run J48 (trained on the 66% of Bigtest1).
  * Compare this accuracy with the mean from 4.3.3.
  * If it is *lower*, the gap is the honest generalisation penalty. Discuss dataset shift:
    Bigtest2 is exactly balanced and was pre-processed
    (`Remove-R105-128`, per its `@relation` line) while Bigtest1 is not - mention this.
- [x] **4.5 Confusion-matrix analysis:** which digits are most confused? Produce a
  normalised confusion-matrix heatmap `lab1/figures/j48_confusion.png`. Hypothesise which
  pixel features cause it (suggest: `7` vs `1`, `5` vs `6`, `3` vs `8`). Support or refute
  the hypothesis by rendering the most-confused average digit images.
- [x] **4.6 (Recommended, ties into p.26-27)** Repeat 4.2-4.4 with `IBk` (kNN) on the digit
  data with `k=1` and `k=5`. Compare with J48. This is where the digit data punishes kNN's
  distance assumption - discuss why (104 binary pixels, no scaling, high dimensionality,
  curse of dimensionality) and report the timing difference. -> add rows to a
  `lab1/results/digits_model_comparison.csv`.

---

## 5. Exercise 2b - overfitting control via `M` (PDF p.14-16)

`M` = J48's `minNumObj`, the **minimum number of training samples in a leaf**.
Larger `M` = less fitting capacity = more regularisation (p.16).

- [x] **5.1** Keep `Bigtest2_104.arff` as the supplied test set. Train on `Bigtest1_104.arff`.
- [x] **5.2** Run J48 with default parameters but sweep `M = 1, 2, 3, ..., 25`
  (extend to 50 if the trend is still moving). Two test modalities per run:
  * `(a)` test on the supplied `Bigtest2_104.arff`  -> `test_acc`
  * `(b)` test on the **training set** (`Test options -> Training set`) -> `train_acc`

  > WEKA default `M` is **2**. Set it explicitly rather than relying on the default.
  > CLI form:
  > ```bash
  > java -cp weka.jar weka.classifiers.trees.J48 -C 0.25 -M $M \
  >      -t Bigtest1_104.arff -T Bigtest2_104.arff
  > java -cp weka.jar weka.classifiers.trees.J48 -C 0.25 -M $M \
  >      -t Bigtest1_104.arff -T Bigtest1_104.arff    # training-set evaluation
  > ```
- [x] **5.3** Write `lab1/results/j48_M_sweep.csv`: `M, train_acc, test_acc, num_leaves,
  tree_size`.
- [x] **5.4** Produce the **key figure of the lab**: `train_acc` and `test_acc` vs `M` on
  one plot (`lab1/figures/j48_M_sweep.png`), with the optimal `M*` marked.
- [x] **5.5** Justify the two observed behaviours (p.14, p.16):
  * As `M` **increases from 1**, both curves first *improve* (the tree stops fitting noise).
  * Past `M*`, `test_acc` **worsens** while `train_acc` keeps... what? Verify which way
    `train_acc` actually moves and explain. (Expect train accuracy to *decrease*
    monotonically as `M` grows - a leaf that must contain `M >= 2` samples cannot memorise
    a single outlier. **Check this against your data before asserting it**: p.16's note
    says train performance "will still improve", so if your run shows the opposite, say so
    and explain - the classic picture is the two curves diverging with train *above* test.)
  * Identify `M*` = the value with the best accuracy on `Bigtest2_104.arff`. Record it.
- [x] **5.6** Answer the p.15 notes explicitly: why does `M=1` NOT give 100% training
  accuracy? Cover all three stated reasons and test at least one of them:
  * greedy information-gain search is not globally optimal;
  * J48's default **pruning** (`-C 0.25` confidence, subtree raising) removes branches -
    rerun with `-U` (unpruned) and `M=1` to see how much of the 100% is recovered;
  * the attribute set (104 binary pixels) may not be expressive enough to isolate every
    single training point.
- [x] **5.7** Write the bias-variance paragraph: `M` is a capacity knob; the best value is
  the one that minimises **generalisation** error, and it must be selected on a validation
  set (or via cross-validation), never on the test set. Note the parallel with `k` in kNN:
  small `k` = high capacity, large `M` = low capacity. Both are examples of the same
  regularisation trade-off - make that link explicit, it is likely exam material.

---

## 6. Scikit-Learn module (PDF p.17-27)

Goal: reproduce the WEKA conclusions in Python. Keep this a **separate** section of the
report so the WEKA results remain an independent check.

- [x] **6.1 Preprocessing note.** For the digit pixels, values are binary `{0,1}` so no
  scaling is needed. For any non-binary data, normalise. State the decision explicitly.
- [x] **6.2 Implement the pipeline of p.23-25** in `lab1/src/sklearn_pipeline.py`:
  * `pandas.read_csv` load -> `train_test_split` twice to make **train/val/test**
    (70/15/15, `stratify=y`, `random_state=42`), exactly the structure shown on p.24.
  * `DecisionTreeClassifier(random_state=42)`, `fit`, predict, then
    `accuracy_score`, `classification_report`, `confusion_matrix` (p.25).
  * Make the loader generic enough to consume the ARFF data via your `arff_utils`.
- [x] **6.3 kNN in sklearn.** `KNeighborsClassifier(n_neighbors=k, metric='euclidean')`
  over the same `k` sweep on the circle data. Assert that your from-scratch kNN (3.1) and
  sklearn agree exactly. Add the sklearn curve to `knn_k_sweep.png` as a dashed line -
  if the two lines differ, you have a bug: fix it.
- [x] **6.4 Decision tree in sklearn.** `DecisionTreeClassifier(criterion='gini'|'entropy',
  min_samples_leaf=M, ccp_alpha=...)` swept over the same `M` values.
  * Map WEKA concepts to sklearn ones and document the mapping in a table:
    `minNumObj -> min_samples_leaf`, `confidenceFactor/pruning -> ccp_alpha`
    (cost-complexity pruning), `information gain -> entropy`, `unpruned -> ccp_alpha=0`.
  * Note honestly that **the numbers will not match exactly** (different split
    tie-breaking, different pruning). The *shape* of the curve and the *ordering of the
    representations* should match - that is the claim to defend.
- [x] **6.5 Iris exercise** (p.26-27): load `sklearn.datasets.load_iris`, implement both a
  kNN classifier and a `DecisionTreeClassifier`, and compare them with accuracy **and** a
  confusion matrix (p.27 asks for exactly this).
  * Use `cross_val_score` with `cv=5` and also a single hold-out; report both.
  * Sweep `k` and `max_depth`; produce `lab1/figures/iris_comparison.png`
    (accuracy vs hyper-parameter, both models) and a small results table.
  * Because Iris is 3-class and near-linearly-separable for one class, explain why both
    models do very well and where they differ.
- [x] **6.6 Cross-check table.** One table in the report:
  `dataset | model | hyper-parameter | WEKA accuracy | sklearn accuracy | note`.
  Any large discrepancy (> 2-3 points) must be investigated and explained, not ignored.

---

## 7. Report requirements

Create `Lab-ML/lab1/REPORT.md`. Suggested structure:

- [x] **7.1** Problem statement: the two regions, the labelling rule (section 1), and the
  class-majority baselines.
- [x] **7.2** Methods: kNN definition + distance metric + tie-breaking; J48/C4.5 split
  criterion, pruning, and `M`.
- [x] **7.3** Results: every figure and table from sections 3-6. Each figure must be
  referenced from the text and must have labelled axes, units, a legend, and a caption.
- [x] **7.4** Discussion, explicitly answering every "Why?" in the PDF:
  * p.6: how to choose the best `k`; kNN accuracy vs `k`.
  * p.7: better or worse with the square inner region - and why (ANDed linear equations).
  * p.8: better or worse with `(x^2, y^2)` - and why.
  * p.9: the one-node solution - and whether it generalises to `circleall.arff`.
  * p.14: comparing `M` behaviour on the training set vs the test set.
  * p.15: why `M=1` does not memorise the training set.
  * p.16: the underfit -> optimal -> overfit progression and how `M*` is selected.
  * p.26-27: Iris, kNN vs tree, accuracy and confusion matrix.
- [x] **7.5** Limitations: synthetic data, only 100 training points for the circle problem,
  the 66%-split variance, WEKA/sklearn implementation differences, no formal significance
  testing (or add McNemar's test if you want to be rigorous).
- [x] **7.6** Reproduction section: exact commands to regenerate every artifact from
  scratch, plus the Python/dependency versions and random seeds.
- [x] **7.7** Self-check before declaring done:
  * every checkbox in this file is either ticked or explicitly marked `SKIPPED + reason`;
  * every number in the report is traceable to a file in `lab1/results/` or a script in
    `lab1/src/`;
  * no originals in `Lab-ML/` were modified (`git status` must show no changes to the
    source `.arff` files or the PDF);
  * report renders cleanly (tables aligned, images resolve, no broken links).

---

## 8. Suggested file layout

```
Lab-ML/
  LAB1_AGENT_TODO.md          <- this file
  lab1/
    REPORT.md                 <- final deliverable
    src/
      arff_utils.py           <- ARFF read/write + verification (2.5)
      knn_scratch.py          <- from-scratch kNN + sklearn agreement test (3.1)
      relabel.py              <- geometry-based re-labelling (3.5.1)
      run_knn_experiments.py  <- k sweep, LOO, CV, boundary figure (3.2-3.4)
      run_j48_experiments.py  <- representation + M sweeps (3.5-5.x)
      sklearn_pipeline.py     <- train/val/test pipeline + Iris (6.2-6.5)
    data/                     <- generated *_sq.arff, *_zt.arff, *_one.arff only
    results/                  <- one CSV per experiment, raw WEKA output text
    figures/                  <- every PNG referenced by the report
```

---

## 9. Quick reference: the numbers you must report

Fill this table in as you go. It is the acceptance test for the whole task.

| # | Experiment | Metric | Value | Source file |
|---|---|---|---|---|
| 1 | kNN, best `k` on test | accuracy | | `results/knn_k_sweep.csv` |
| 2 | kNN, best `k` by LOO-CV | accuracy | | `results/knn_k_sweep.csv` |
| 3 | kNN, best `k` on `circleall` | accuracy | | `results/knn_k_sweep.csv` |
| 4 | J48, circle `(x,y)` | acc / leaves | | `results/j48_representation.csv` |
| 5 | J48, square inner region `(x,y)` | acc / leaves | | `results/j48_representation.csv` |
| 6 | J48, circle `(x^2,y^2)` | acc / leaves | | `results/j48_representation.csv` |
| 7 | J48, one-node representation | acc / leaves | | `results/j48_representation.csv` |
| 8 | J48 digits, 66% split | accuracy | | `results/j48_bigtest1_split66.csv` |
| 9 | J48 digits, mean over >=5 seeds | mean +/- sd | | `results/j48_seeds.csv` |
| 10 | J48 digits, `Bigtest2` supplied test | accuracy | | `results/j48_bigtest1_split66.csv` |
| 11 | J48 digits, optimal `M` | `M*` / accuracy | | `results/j48_M_sweep.csv` |
| 12 | J48 digits, `M=1` unpruned on training set | accuracy | | `results/j48_M_sweep.csv` |
| 13 | sklearn vs WEKA, matched configs | delta | | `results/*` + report |
| 14 | Iris, kNN vs tree (5-fold CV) | accuracies | | `results/iris_comparison.csv` |

---

## 10. Out-of-scope appendix (do NOT implement)

Kept here only so the agent does not get confused by the merged PDF:

* **p.28-43** Lab 2, Clustering: `gausstrain/test/hv.arff`, `SimpleKMeans`, `XMeans`
  (files `gausstrain.arff`, `gausstest.arff`, `gausstrainhv.arff`, `gausstesthv.arff` are
  present in `Lab-ML/`).
* **p.44-56** PyTorch tutorial: tensors, DataLoaders/transforms, MLP, loss/optimiser,
  train/eval loops.
* **p.57-72** Lab 5, Genetic Algorithms with DEAP (`Lab-ML/deap_examples_1/`,
  `Lab-ML/ml-patterns/deap/` already contain `onemax.py`, `funcmin.py`, `guess-smiley.py`,
  etc.). Note: the first half of Lab 1's section 10 table in some years' PDFs overlaps this
  - keep them separate.
* **p.73-81** Lab 6, Genetic Programming / PSO, and `GPdenoise.pdf`.
* **p.82-91** Ant Trail problem (decision tree + MLP + GP comparison).
