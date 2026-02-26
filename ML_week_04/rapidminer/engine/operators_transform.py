"""
operators_transform.py – Data‑transformation operators.

Implements: Select Attributes, Filter Examples, Rename, Set Role,
Replace Missing, Normalize, Discretize, Generate Attributes,
Remove Duplicates, Sort, Sample, Split Data, Append, Join,
Aggregate, Transpose, Pivot.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.operator_base import (
    Operator,
    OpCategory,
    ParamKind,
    ParamSpec,
    Port,
    PortType,
    register_operator,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Select Attributes
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class SelectAttributes(Operator):
    op_type = "Select Attributes"
    category = OpCategory.TRANSFORM
    description = "Keep or remove selected columns from the ExampleSet."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("mode", ParamKind.CHOICE, default="keep", choices=["keep", "remove"], description="Keep or remove the listed columns."),
            ParamSpec("columns", ParamKind.COLUMN_LIST, default=[], description="Comma‑separated column names."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        cols = self._parse_cols()
        if self.get_param("mode") == "keep":
            valid = [c for c in cols if c in df.columns]
            df = df[valid]
        else:
            df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
        return {"out": df}

    def _parse_cols(self) -> List[str]:
        raw = self.get_param("columns")
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            return [c.strip() for c in raw.split(",") if c.strip()]
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Filter Examples
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class FilterExamples(Operator):
    op_type = "Filter Examples"
    category = OpCategory.TRANSFORM
    description = "Filter rows using a condition (e.g. 'age > 30')."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("condition", ParamKind.TEXT, default="", description="Pandas query string, e.g. 'age > 30 and sex == \"male\"'."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        cond = self.get_param("condition")
        if cond:
            df = df.query(cond)
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Rename
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class RenameOp(Operator):
    op_type = "Rename"
    category = OpCategory.TRANSFORM
    description = "Rename columns (old→new, comma‑separated pairs)."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("mapping", ParamKind.TEXT, default="", description="Pairs: old1=new1, old2=new2"),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        raw = self.get_param("mapping") or ""
        rename_map: Dict[str, str] = {}
        for pair in raw.split(","):
            if "=" in pair:
                old, new = pair.split("=", 1)
                rename_map[old.strip()] = new.strip()
        if rename_map:
            df = df.rename(columns=rename_map)
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Set Role
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class SetRole(Operator):
    op_type = "Set Role"
    category = OpCategory.TRANSFORM
    description = "Designate a column as label, id, weight, or feature."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("column", ParamKind.COLUMN, default="", description="Column to set the role on."),
            ParamSpec("role", ParamKind.CHOICE, default="label", choices=["label", "id", "weight", "feature"], description="Target role."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        col = self.get_param("column")
        role = self.get_param("role")
        if col and col in df.columns:
            df.attrs[f"_role_{col}"] = role
            # Store role metadata on the dataframe
            roles: Dict[str, str] = df.attrs.get("_roles", {})
            roles[col] = role
            df.attrs["_roles"] = roles
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Replace Missing Values
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ReplaceMissing(Operator):
    op_type = "Replace Missing"
    category = OpCategory.TRANSFORM
    description = "Fill missing values with mean / median / mode / constant."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("strategy", ParamKind.CHOICE, default="mean", choices=["mean", "median", "mode", "constant"], description="Imputation strategy."),
            ParamSpec("constant", ParamKind.STRING, default="0", description="Value when strategy='constant'."),
            ParamSpec("columns", ParamKind.COLUMN_LIST, default=[], description="Columns to impute (empty = all numeric)."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        cols = self._parse_cols(df)
        strategy = self.get_param("strategy")

        for c in cols:
            if c not in df.columns:
                continue
            if strategy == "mean":
                df[c] = df[c].fillna(df[c].mean())
            elif strategy == "median":
                df[c] = df[c].fillna(df[c].median())
            elif strategy == "mode":
                mode_val = df[c].mode()
                if len(mode_val):
                    df[c] = df[c].fillna(mode_val.iloc[0])
            else:
                df[c] = df[c].fillna(self.get_param("constant"))
        return {"out": df}

    def _parse_cols(self, df: pd.DataFrame) -> List[str]:
        raw = self.get_param("columns")
        if isinstance(raw, str):
            cols = [c.strip() for c in raw.split(",") if c.strip()]
        elif isinstance(raw, list) and raw:
            cols = raw
        else:
            cols = df.select_dtypes(include="number").columns.tolist()
        return cols


# ═══════════════════════════════════════════════════════════════════════════
# Normalize
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class Normalize(Operator):
    op_type = "Normalize"
    category = OpCategory.TRANSFORM
    description = "Normalize numeric columns (z‑score, min‑max, or log)."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("method", ParamKind.CHOICE, default="z-score", choices=["z-score", "min-max", "log"], description="Normalization method."),
            ParamSpec("columns", ParamKind.COLUMN_LIST, default=[], description="Columns (empty = all numeric)."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        method = self.get_param("method")
        cols = self._parse_cols(df)
        num_cols = [c for c in cols if c in df.select_dtypes(include="number").columns]

        for c in num_cols:
            if method == "z-score":
                std = df[c].std()
                df[c] = (df[c] - df[c].mean()) / (std if std else 1)
            elif method == "min-max":
                mn, mx = df[c].min(), df[c].max()
                rng = mx - mn if mx != mn else 1
                df[c] = (df[c] - mn) / rng
            elif method == "log":
                df[c] = np.log1p(df[c])
        return {"out": df}

    def _parse_cols(self, df: pd.DataFrame) -> List[str]:
        raw = self.get_param("columns")
        if isinstance(raw, str):
            cols = [c.strip() for c in raw.split(",") if c.strip()]
        elif isinstance(raw, list) and raw:
            cols = raw
        else:
            cols = df.select_dtypes(include="number").columns.tolist()
        return cols


# ═══════════════════════════════════════════════════════════════════════════
# Discretize
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class Discretize(Operator):
    op_type = "Discretize"
    category = OpCategory.TRANSFORM
    description = "Bin a numeric column into discrete intervals."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("column", ParamKind.COLUMN, default="", description="Column to discretize."),
            ParamSpec("bins", ParamKind.INT, default=5, min_val=2, max_val=100, description="Number of bins."),
            ParamSpec("labels", ParamKind.STRING, default="", description="Comma‑sep labels (empty = auto)."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        col = self.get_param("column")
        n_bins = int(self.get_param("bins") or 5)
        labels_raw = self.get_param("labels") or ""
        labels: Any = [l.strip() for l in labels_raw.split(",") if l.strip()] or None
        if labels is not None and len(labels) != n_bins:
            labels = False
        if col and col in df.columns:
            kw = {"bins": n_bins}
            if labels is not None:
                kw["labels"] = labels
            df[col] = pd.cut(df[col], **kw)
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Generate Attributes
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class GenerateAttributes(Operator):
    op_type = "Generate Attributes"
    category = OpCategory.TRANSFORM
    description = "Create a new column using a formula (pandas eval)."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("name", ParamKind.STRING, default="new_col", description="New column name."),
            ParamSpec("formula", ParamKind.TEXT, default="", description="Pandas eval expression, e.g. 'col_a + col_b'."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        name = self.get_param("name")
        formula = self.get_param("formula")
        if formula:
            df[name] = df.eval(formula)
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Remove Duplicates
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class RemoveDuplicates(Operator):
    op_type = "Remove Duplicates"
    category = OpCategory.TRANSFORM
    description = "Drop duplicate rows (optionally based on key columns)."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("columns", ParamKind.COLUMN_LIST, default=[], description="Key columns (empty = all)."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        cols = self._parse_cols()
        subset = cols if cols else None
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        return {"out": df}

    def _parse_cols(self) -> List[str]:
        raw = self.get_param("columns")
        if isinstance(raw, str):
            return [c.strip() for c in raw.split(",") if c.strip()]
        if isinstance(raw, list):
            return raw
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Sort
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class SortOp(Operator):
    op_type = "Sort"
    category = OpCategory.TRANSFORM
    description = "Sort the ExampleSet by one or more columns."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("column", ParamKind.COLUMN, default="", description="Column to sort by."),
            ParamSpec("ascending", ParamKind.BOOL, default=True, description="Sort ascending."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        col = self.get_param("column")
        if col and col in df.columns:
            df = df.sort_values(col, ascending=self.get_param("ascending")).reset_index(drop=True)
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Sample
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class SampleOp(Operator):
    op_type = "Sample"
    category = OpCategory.TRANSFORM
    description = "Random sample of rows (fraction or fixed n)."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("mode", ParamKind.CHOICE, default="fraction", choices=["fraction", "absolute"], description="Sampling mode."),
            ParamSpec("fraction", ParamKind.FLOAT, default=0.5, min_val=0.0, max_val=1.0, description="Fraction (0–1)."),
            ParamSpec("n", ParamKind.INT, default=100, min_val=1, description="Absolute number of rows."),
            ParamSpec("seed", ParamKind.INT, default=42, description="Random seed."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"].copy()
        seed = int(self.get_param("seed") or 42)
        if self.get_param("mode") == "fraction":
            frac = float(self.get_param("fraction") or 0.5)
            df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
        else:
            n = min(int(self.get_param("n") or 100), len(df))
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Split Data
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class SplitData(Operator):
    op_type = "Split Data"
    category = OpCategory.TRANSFORM
    description = "Split into training and testing sets."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["train"] = Port("train", PortType.EXAMPLE_SET)
        self.outputs["test"] = Port("test", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("ratio", ParamKind.FLOAT, default=0.7, min_val=0.1, max_val=0.99, description="Training fraction."),
            ParamSpec("stratified", ParamKind.BOOL, default=False, description="Stratify on label column."),
            ParamSpec("label_column", ParamKind.COLUMN, default="", description="Label column for stratification."),
            ParamSpec("seed", ParamKind.INT, default=42, description="Random seed."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        ratio = float(self.get_param("ratio") or 0.7)
        seed = int(self.get_param("seed") or 42)
        stratified = self.get_param("stratified")
        label_col = self.get_param("label_column")

        if stratified and label_col and label_col in df.columns:
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                df, train_size=ratio, stratify=df[label_col], random_state=seed,
            )
        else:
            train_df = df.sample(frac=ratio, random_state=seed)
            test_df = df.drop(train_df.index)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        # Propagate roles metadata
        for attr_key in ("_roles",):
            if attr_key in df.attrs:
                train_df.attrs[attr_key] = df.attrs[attr_key]
                test_df.attrs[attr_key] = df.attrs[attr_key]
        logger.info("Split: %d train, %d test", len(train_df), len(test_df))
        return {"train": train_df, "test": test_df}


# ═══════════════════════════════════════════════════════════════════════════
# Append
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class AppendOp(Operator):
    op_type = "Append"
    category = OpCategory.TRANSFORM
    description = "Vertically concatenate two ExampleSets."

    def _build_ports(self) -> None:
        self.inputs["in1"] = Port("in1", PortType.EXAMPLE_SET)
        self.inputs["in2"] = Port("in2", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df1: pd.DataFrame = inputs["in1"]
        df2: pd.DataFrame = inputs["in2"]
        merged = pd.concat([df1, df2], ignore_index=True)
        return {"out": merged}


# ═══════════════════════════════════════════════════════════════════════════
# Join
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class JoinOp(Operator):
    op_type = "Join"
    category = OpCategory.TRANSFORM
    description = "Join two ExampleSets on key columns."

    def _build_ports(self) -> None:
        self.inputs["left"] = Port("left", PortType.EXAMPLE_SET)
        self.inputs["right"] = Port("right", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("key_columns", ParamKind.COLUMN_LIST, default=[], description="Key columns for the join."),
            ParamSpec("join_type", ParamKind.CHOICE, default="inner", choices=["inner", "left", "right", "outer"], description="Type of join."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        left: pd.DataFrame = inputs["left"]
        right: pd.DataFrame = inputs["right"]
        keys = self.get_param("key_columns")
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        how = self.get_param("join_type") or "inner"
        if keys:
            merged = pd.merge(left, right, on=keys, how=how)
        else:
            merged = pd.merge(left, right, left_index=True, right_index=True, how=how)
        return {"out": merged}


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class AggregateOp(Operator):
    op_type = "Aggregate"
    category = OpCategory.TRANSFORM
    description = "Group‑by aggregation (sum, mean, count, min, max)."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("group_by", ParamKind.COLUMN_LIST, default=[], description="Columns to group by."),
            ParamSpec("aggregations", ParamKind.TEXT, default="mean", description="Aggregation function(s): mean, sum, count, min, max. Comma‑sep for multiple."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        group_cols = self.get_param("group_by")
        if isinstance(group_cols, str):
            group_cols = [c.strip() for c in group_cols.split(",") if c.strip()]
        agg_raw = self.get_param("aggregations") or "mean"
        agg_funcs = [a.strip() for a in agg_raw.split(",") if a.strip()]
        if group_cols:
            result = df.groupby(group_cols).agg(agg_funcs).reset_index()
            result.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in result.columns]
        else:
            result = df.agg(agg_funcs).to_frame().T
        return {"out": result}


# ═══════════════════════════════════════════════════════════════════════════
# Transpose
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class TransposeOp(Operator):
    op_type = "Transpose"
    category = OpCategory.TRANSFORM
    description = "Transpose rows and columns."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        return {"out": df.T.reset_index()}


# ═══════════════════════════════════════════════════════════════════════════
# Pivot
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class PivotOp(Operator):
    op_type = "Pivot"
    category = OpCategory.TRANSFORM
    description = "Pivot table: index, columns, values, aggfunc."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("index", ParamKind.COLUMN, default="", description="Row index column."),
            ParamSpec("columns", ParamKind.COLUMN, default="", description="Column‑level column."),
            ParamSpec("values", ParamKind.COLUMN, default="", description="Values column."),
            ParamSpec("aggfunc", ParamKind.CHOICE, default="mean", choices=["mean", "sum", "count", "min", "max"], description="Aggregation function."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        idx = self.get_param("index")
        cols = self.get_param("columns")
        vals = self.get_param("values")
        agg = self.get_param("aggfunc") or "mean"
        if idx and cols and vals:
            result = pd.pivot_table(df, index=idx, columns=cols, values=vals, aggfunc=agg).reset_index()
        else:
            result = df.copy()
        return {"out": result}
