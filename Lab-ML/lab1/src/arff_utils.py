"""ARFF reading/writing utilities for Lab 1.

Self-contained parser (no external ARFF dependency) plus a writer used to
generate the relabelled / re-represented datasets.

Public API
----------
load_arff(path)  -> (X, y, attrs, classes, relation)
write_arff(path, X, y, attr_names, classes, relation)
load_circle(name)  -> convenience loader for the three circle files
"""

from __future__ import annotations

import os
import re
from typing import List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants shared by every experiment
# --------------------------------------------------------------------------

#: Repository root and the directories holding the (read-only) source data.
LAB1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(LAB1_DIR))
SOURCE_DATA_DIR = os.path.join(REPO_ROOT, "Lab-ML")

DATA_DIR = os.path.join(LAB1_DIR, "data")
RESULTS_DIR = os.path.join(LAB1_DIR, "results")
FIGURES_DIR = os.path.join(LAB1_DIR, "figures")

#: Documented label encoding used *everywhere* in the project.
#: c (inside the circle / inner region) -> 0, q (outer square ring) -> 1.
#: "Positive" class in the binary sense = q (label 1).
LABEL_TO_INT = {"c": 0, "q": 1}
INT_TO_LABEL = {0: "c", 1: "q"}


def encode_labels(y: Sequence[str]) -> np.ndarray:
    """Map the nominal labels {'c','q'} to integers {0,1}."""
    return np.array([LABEL_TO_INT[str(v).strip()] for v in y], dtype=int)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------

_ATTR_RE = re.compile(
    r"^\s*@attribute\s+(?P<name>'[^']*'|\"[^\"]*\"|\S+)\s+(?P<type>.+?)\s*$",
    re.IGNORECASE,
)
_RELATION_RE = re.compile(r"^\s*@relation\s+(?P<name>.+?)\s*$", re.IGNORECASE)


def _strip_quotes(name: str) -> str:
    name = name.strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in "'\"":
        return name[1:-1]
    return name


def _parse_nominal(type_spec: str) -> List[str]:
    """Extract the values of a nominal declaration ``{a,b,c}``."""
    m = re.search(r"\{(.*)\}", type_spec)
    if not m:
        raise ValueError(f"not a nominal type: {type_spec!r}")
    return [v.strip() for v in m.group(1).split(",") if v.strip() != ""]


def load_arff(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], str]:
    """Load an ARFF file with numeric features and a nominal class attribute.

    Returns
    -------
    X : (n_samples, n_features) float array
    y : (n_samples,) array of str (nominal class values, as written in the file)
    attrs : list of feature attribute names (class excluded)
    classes : list of nominal class values in declaration order
    relation : the @relation name
    """
    attrs: List[str] = []
    attr_types: List[str] = []
    classes: List[str] = []
    relation = ""
    rows: List[List[str]] = []
    in_data = False

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue

            if not in_data:
                m = _RELATION_RE.match(line)
                if m:
                    relation = _strip_quotes(m.group("name"))
                    continue
                if line.lower().startswith("@data"):
                    in_data = True
                    continue
                m = _ATTR_RE.match(line)
                if m:
                    attrs.append(_strip_quotes(m.group("name")))
                    attr_types.append(m.group("type").strip())
                    continue
                # Any other @-directive (@inputs etc.) is ignored.
                continue

            # --- data section ---
            # Values may be comma-separated (WEKA's own writer) or whitespace-
            # separated (MATLAB-generated files such as the Lab 2 gaussian sets).
            if line.startswith(","):
                line = line[1:]
            if "," in line:
                values = [v.strip() for v in line.split(",")]
            else:
                values = line.split()
            # A trailing empty field is an artefact of a trailing comma.
            if values and values[-1] == "":
                values.pop()
            rows.append(values)

    if not attrs:
        raise ValueError(f"no @attribute lines found in {path}")
    if attr_types and attr_types[-1].startswith("{"):
        classes = _parse_nominal(attr_types[-1])
        feature_attrs = attrs[:-1]
        feature_types = attr_types[:-1]
    else:  # every attribute is a feature (unsupervised ARFF)
        feature_attrs = attrs
        feature_types = attr_types

    if not rows:
        raise ValueError(f"no data rows found in {path}")

    n_features = len(feature_attrs)
    X = np.empty((len(rows), n_features), dtype=float)
    y = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        if len(row) < n_features:
            raise ValueError(
                f"{path}: row {i} has {len(row)} values, expected >= {n_features}"
            )
        for j in range(n_features):
            X[i, j] = float(row[j])
        y[i] = row[n_features] if len(row) > n_features else ""

    # Keep the declared type information around for the writer.
    load_arff.last_types = feature_types  # type: ignore[attr-defined]
    return X, np.asarray(y, dtype=str), feature_attrs, classes, relation


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


def write_csv(path: str, rows: Sequence[dict], columns: Sequence[str]) -> None:
    """Write a list of dicts to CSV with correct quoting.

    Naive ``",".join`` breaks as soon as a field contains a comma (e.g. the model
    name "IBk (kNN, k=1)"), silently shifting every later column.  Everything
    goes through :mod:`csv` so that cannot happen.
    """
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})


def write_arff(
    path: str,
    X: np.ndarray,
    y: Sequence,
    attr_names: Sequence[str],
    classes: Sequence[str],
    relation: str = "lab1",
    decimals: int = 7,
) -> None:
    """Write a numeric-feature ARFF file with one trailing nominal class.

    ``y`` may hold either the nominal strings themselves (``['c','q']``) or
    integer indices into ``classes``.  Both are normalised to the declared
    nominal strings, because WEKA rejects data whose class value is not a
    declared nominal token (an integer ``1`` is *not* the same as ``q``).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] != len(attr_names):
        raise ValueError("attr_names length does not match X columns")
    if len(y) != X.shape[0]:
        raise ValueError("y length does not match X rows")

    class_list = [str(c) for c in classes]
    class_set = set(class_list)
    labels: List[str] = []
    for value in y:
        if isinstance(value, (int, np.integer)) or (
            isinstance(value, (float, np.floating)) and float(value).is_integer()
        ):
            idx = int(value)
            if not 0 <= idx < len(class_list):
                raise ValueError(f"class index {idx} out of range for {class_list}")
            labels.append(class_list[idx])
        else:
            token = str(value).strip()
            if token not in class_set:
                raise ValueError(f"label {token!r} not declared in {class_list}")
            labels.append(token)

    fmt = f"%.{decimals}f"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"@relation {relation}\n\n")
        for name in attr_names:
            fh.write(f"@attribute {name} numeric\n")
        fh.write("@attribute class {" + ",".join(class_list) + "}\n")
        fh.write("\n@data\n")
        for row, label in zip(X, labels):
            values = ",".join(fmt % v for v in row)
            fh.write(f"{values},{label}\n")


# --------------------------------------------------------------------------
# Convenience loaders
# --------------------------------------------------------------------------

CIRCLE_FILES = {
    "train": "circletrain.arff",
    "test": "circletest.arff",
    "all": "circleall.arff",
}


def load_circle(name: str, variant: str = "") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a circle dataset and return (X, y_int, attrs).

    ``variant`` is a suffix appended to the file stem, e.g. '_sq' for the
    square-inner-region relabelling or '_zt' for the (x^2, y^2) representation.
    """
    stem, ext = os.path.splitext(CIRCLE_FILES[name])
    filename = f"{stem}{variant}{ext}"
    # Generated variants live in lab1/data, the originals in Lab-ML.
    for directory in (DATA_DIR, SOURCE_DATA_DIR):
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            X, y_str, attrs, _classes, _rel = load_arff(candidate)
            return X, encode_labels(y_str), attrs
    raise FileNotFoundError(filename)


def digit_path(which: int) -> str:
    return os.path.join(SOURCE_DATA_DIR, f"Bigtest{which}_104.arff")


__all__ = [
    "LAB1_DIR",
    "REPO_ROOT",
    "SOURCE_DATA_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "LABEL_TO_INT",
    "INT_TO_LABEL",
    "encode_labels",
    "load_arff",
    "write_arff",
    "load_circle",
    "digit_path",
    "CIRCLE_FILES",
]
