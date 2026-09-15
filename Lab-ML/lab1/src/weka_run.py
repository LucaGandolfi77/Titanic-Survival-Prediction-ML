"""WEKA CLI helpers: run classifiers headlessly and parse the output.

The WEKA GUI is not scriptable, so every WEKA experiment in this project is
executed through ``java -cp weka.jar ...`` using the JRE bundles with the
Weka application.  ``--add-opens java.base/java.lang=ALL-UNNAMED`` is required
by the Java 25 runtime shipped with Weka 3.8.7.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

WEKA_JAVA = "/Applications/weka-3.8.7.app/Contents/runtime/Contents/Home/bin/java"
WEKA_JAR = "/Applications/weka-3.8.7.app/Contents/app/weka.jar"
JAVA_OPTS = ["--add-opens", "java.base/java.lang=ALL-UNNAMED"]

J48 = "weka.classifiers.trees.J48"
IBK = "weka.classifiers.lazy.IBk"


def weka_available() -> bool:
    return os.path.exists(WEKA_JAVA) and os.path.exists(WEKA_JAR)


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


@dataclass
class WekaResult:
    """Parsed summary of a WEKA evaluation run."""

    raw: str
    accuracy: Optional[float] = None
    correctly_classified: Optional[int] = None
    incorrectly_classified: Optional[int] = None
    total_instances: Optional[int] = None
    kappa: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    size_of_tree: Optional[float] = None
    num_leaves: Optional[int] = None
    tree_text: str = ""
    confusion: List[List[int]] = field(default_factory=list)
    class_labels: List[str] = field(default_factory=list)
    build_time_s: Optional[float] = None

    def confusion_dict(self) -> Dict[str, object]:
        return {
            "class_labels": self.class_labels,
            "matrix": self.confusion,
        }


def _search(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    return cast(m.group(1))


def split_evaluations(text: str) -> Dict[str, str]:
    """Split a WEKA run's output into its evaluation sections.

    When both ``-t`` and ``-T`` are given, WEKA prints **two** complete
    evaluations: first ``=== Error on training data ===`` and then
    ``=== Error on test data ===``.  Parsing the whole text naively would
    silently report the *training* accuracy, so the sections are separated
    here and the caller picks the one it means.

    Returns ``{section_name: text}``; an empty-key entry holds the preamble
    (the printed model / tree).
    """
    pattern = re.compile(r"^===\s*(Error on (?:training|test) data|Cross-validation)\s*===\s*$",
                         re.MULTILINE)
    marks = list(pattern.finditer(text))
    out: Dict[str, str] = {"": text[: marks[0].start()] if marks else text}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1)] = text[m.start():end]
    return out


def parse_weka_output(text: str, section: Optional[str] = None) -> WekaResult:
    """Parse a WEKA summary.

    ``section`` selects which evaluation to parse when WEKA printed more than
    one: ``"Error on test data"`` (the default when present, because a test
    evaluation is what the experiments care about), ``"Error on training data"``,
    or ``"Cross-validation"``.  Pass ``"all"`` to get the preamble instead
    (useful for reading a printed tree).
    """
    evaluations = split_evaluations(text)

    if section is None:
        if "Error on test data" in evaluations:
            section = "Error on test data"
        elif "Error on training data" in evaluations:
            section = "Error on training data"
        elif "Cross-validation" in evaluations:
            section = "Cross-validation"
        else:
            section = ""

    if section == "all":
        # The preamble holds the printed model / tree, which has no accuracy
        # metrics but does carry "Number of Leaves" and "Size of the tree".
        body = evaluations.get("", "")
    else:
        body = evaluations.get(section)
    if body is None:
        raise KeyError(f"section {section!r} not present; "
                       f"available: {[k for k in evaluations if k]}")
    # Force the confusion-matrix / metric regexes onto the selected section.
    text = body
    res = WekaResult(raw=body)
    res.full_output = evaluations.get("", "") + body  # type: ignore[attr-defined]

    m = re.search(rf"Correctly Classified Instances\s+(\d+)\s+({_NUM})\s*%", text)
    if m:
        res.correctly_classified = int(m.group(1))
        res.accuracy = float(m.group(2))
    m = re.search(r"Incorrectly Classified Instances\s+(\d+)\s+(" + _NUM + r")\s*%", text)
    if m:
        res.incorrectly_classified = int(m.group(1))
    m = re.search(r"Total Number of Instances\s+(\d+)", text)
    if m:
        res.total_instances = int(m.group(1))

    res.kappa = _search(r"Kappa statistic\s+(" + _NUM + ")", text)
    res.mae = _search(r"Mean absolute error\s+(" + _NUM + ")", text)
    res.rmse = _search(r"Root mean squared error\s+(" + _NUM + ")", text)
    res.build_time_s = _search(r"Time taken to build model:\s*(" + _NUM + ")", text)

    size = _search(r"Size of the tree\s*:\s*(\d+)", text)
    leaves = _search(r"Number of Leaves\s*:\s*(\d+)", text)
    res.size_of_tree = float(size) if size is not None else None
    res.num_leaves = int(leaves) if leaves is not None else None

    # Tree / model text block.
    for header in ("J48 pruned tree", "J48 unpruned tree"):
        if header in text:
            block = text.split(header, 1)[1]
            block = block.split("Number of Leaves", 1)[0]
            res.tree_text = block.strip()
            break

    # Confusion matrix.  WEKA prints:
    #     === Confusion Matrix ===
    #        a   b   <-- classified as
    #      354   0 |   a = 0
    # The letter row is only a placeholder, so the digit classes are read back
    # from the trailing "= <class>" on each body line when present.
    if "=== Confusion Matrix ===" in text:
        block = text.split("=== Confusion Matrix ===", 1)[1]
        # Stop at the next section header, if any.
        block = block.split("\n\n\n", 1)[0]
        body_lines = [ln for ln in block.splitlines() if "|" in ln]
        matrix, labels = [], []
        for ln in body_lines:
            left, _, right = ln.partition("|")
            nums = re.findall(r"\d+", left)
            if not nums:
                continue
            matrix.append([int(v) for v in nums])
            m = re.search(r"=\s*(\S+)\s*$", right)
            labels.append(m.group(1) if m else "")
        if matrix and len({len(r) for r in matrix}) == 1:
            res.confusion = matrix
            if labels and all(labels):
                res.class_labels = labels
            else:
                res.class_labels = [chr(ord("a") + i) for i in range(len(matrix))]

    return res


def run_weka(
    classifier_args: List[str],
    train: Optional[str] = None,
    test: Optional[str] = None,
    extra: Optional[List[str]] = None,
    general: Optional[List[str]] = None,
    section: Optional[str] = None,
    timeout: int = 1800,
) -> WekaResult:
    """Run a WEKA classifier and return the parsed result.

    WEKA's CLI option order is significant: ``java -cp weka.jar CLASSIFIER
    <general options> -t <train> <classifier options> -T <test>``.  The class
    name must come **immediately** after ``-cp`` - anything placed between them
    is interpreted as a JVM option and the run dies with "Unrecognized option".

    Note that J48 has **no** randomness: its option list contains only
    ``-U -C -M -R -N -B -S`` where ``-S`` is *not* a seed but "do not perform
    subtree raising".  Seeds belong to the evaluation options (``-s`` for
    cross-validation or percentage split), not to J48.

    When both ``-t`` and ``-T`` are supplied WEKA prints a training evaluation
    *and* a test evaluation; by default the **test** one is parsed.  Use
    ``section="Error on training data"`` to get the other, or
    ``section="all"`` to parse the preamble (the printed tree).
    """
    cmd = [WEKA_JAVA] + JAVA_OPTS + ["-cp", WEKA_JAR]
    cmd += list(classifier_args)
    cmd += list(general or [])
    if train:
        cmd += ["-t", train]
    if test:
        cmd += ["-T", test]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0 and "Correctly Classified" not in out:
        raise RuntimeError(
            "WEKA command failed (%s):\n%s" % (" ".join(cmd), out[-4000:])
        )
    res = parse_weka_output(out, section=section)
    res.full_output = out  # type: ignore[attr-defined]
    return res


def run_j48(
    train: str,
    test: Optional[str] = None,
    min_num_obj: int = 2,
    confidence: float = 0.25,
    unpruned: bool = False,
    general: Optional[List[str]] = None,
) -> WekaResult:
    """Run J48 (WEKA's C4.5).  Deterministic: no seed option exists."""
    args = [J48, "-C", str(confidence), "-M", str(min_num_obj)]
    if unpruned:
        args.append("-U")
    return run_weka(args, train=train, test=test, general=general)


def run_weka_predictions(
    classifier_args: List[str],
    train: str,
    test: str,
    extra: Optional[List[str]] = None,
    timeout: int = 1800,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run WEKA with ``-p 0`` and return (predicted_class_indices, actual_class_indices).

    WEKA's prediction table is::

        inst#     actual  predicted error distribution
            1        6:5        6:5       0,0,0,0,0,*1,0,0,0,0

    Parsing the actual class *from the file* rather than from this table is
    deliberate in the callers - here we return both so the caller can decide.
    Class indices are 1-based in WEKA and are converted to 0-based.
    """
    cmd = [WEKA_JAVA] + JAVA_OPTS + ["-cp", WEKA_JAR]
    cmd += list(classifier_args)
    if extra:
        cmd += list(extra)
    cmd += ["-t", train, "-T", test, "-p", "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if "Predictions on test data" not in out:
        raise RuntimeError(
            "WEKA prediction run failed (%s):\n%s" % (" ".join(cmd), out[-4000:])
        )

    predicted, actual = [], []
    for line in out.splitlines():
        # Data rows look like: "   1   6:5   6:5   0,0,0,*1,0"
        m = re.match(r"\s*(\d+)\s+([\d?]+):(\S*)\s+([\d?]+):(\S*)", line)
        if not m:
            continue
        actual.append(int(m.group(2)))
        predicted.append(int(m.group(4)))
    if not predicted:
        raise RuntimeError("no prediction rows parsed from WEKA output")
    return (np.array(predicted, dtype=int) - 1, np.array(actual, dtype=int) - 1)


def run_ibk(
    train: str,
    test: Optional[str] = None,
    k: int = 1,
    seed: int = 1,
) -> WekaResult:
    """Run IBk (WEKA's kNN).  ``-K`` neighbours, ``-W 0`` = no distance weighting."""
    # IBk has no seed either; -S on IBk means "do not search for k".
    args = [IBK, "-K", str(k), "-W", "0"]
    return run_weka(args, train=train, test=test)


# --------------------------------------------------------------------------
# Replication of WEKA's percentage split
# --------------------------------------------------------------------------


def weka_percentage_split_indices(
    y: np.ndarray, percentage: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce WEKA's ``-split-percentage`` train/test partition.

    WEKA (``Instances.trainCV``/``testCV`` with folds=1) shuffles the instance
    order with ``java.util.Random(seed)``, then takes the first
    ``round(percentage/100 * n)`` instances as the training set.  ``java.util.Random``
    is a 48-bit LCG, reimplemented here so that the split is reproducible without
    shelling out.
    """
    n = len(y)
    rnd = JavaRandom(seed)
    order = list(range(n))
    # Fisher-Yates as implemented in weka.core.Instances.randomize / Utils
    for i in range(n - 1, 0, -1):
        j = rnd.next_int(i + 1)
        order[i], order[j] = order[j], order[i]
    n_train = int(round(percentage / 100.0 * n))
    train_idx = np.array(order[:n_train], dtype=int)
    test_idx = np.array(order[n_train:], dtype=int)
    return train_idx, test_idx


class JavaRandom:
    """Minimal reimplementation of ``java.util.Random`` (48-bit LCG)."""

    _MULT = 0x5DEECE66D
    _ADD = 0xB
    _MASK = (1 << 48) - 1

    def __init__(self, seed: int):
        self.seed = (seed ^ self._MULT) & self._MASK

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * self._MULT + self._ADD) & self._MASK
        return self.seed >> (48 - bits)

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if (bound & -bound) == bound:  # power of two
            return (bound * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            val = bits % bound
            # Reject values that would make the distribution non-uniform.
            if bits - val + (bound - 1) < (1 << 31):
                return val


__all__ = [
    "WEKA_JAVA",
    "WEKA_JAR",
    "J48",
    "IBK",
    "WekaResult",
    "weka_available",
    "parse_weka_output",
    "split_evaluations",
    "run_weka",
    "run_weka_predictions",
    "run_j48",
    "run_ibk",
    "weka_percentage_split_indices",
    "JavaRandom",
]
