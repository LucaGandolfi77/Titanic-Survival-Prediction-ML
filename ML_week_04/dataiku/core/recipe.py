"""
RecipeEngine – implements all data transformation recipes:
Prepare, Join, Group By, and Custom Python.

Each recipe is a JSON-serialisable config dict that can be executed
against one or more DataFrames to produce an output DataFrame.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step executors (used by the Prepare recipe)
# ---------------------------------------------------------------------------

def _step_filter_rows(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Filter rows: keep rows where column <op> value."""
    col = cfg["column"]
    op = cfg["operator"]
    val = cfg["value"]
    s = df[col]
    # Try numeric comparison
    try:
        val_num = float(val)
        if op == "==":
            return df[s == val_num]
        elif op == "!=":
            return df[s != val_num]
        elif op == ">":
            return df[s > val_num]
        elif op == ">=":
            return df[s >= val_num]
        elif op == "<":
            return df[s < val_num]
        elif op == "<=":
            return df[s <= val_num]
    except (ValueError, TypeError):
        pass
    # String comparison
    if op == "==":
        return df[s.astype(str) == str(val)]
    elif op == "!=":
        return df[s.astype(str) != str(val)]
    elif op == "contains":
        return df[s.astype(str).str.contains(str(val), na=False)]
    elif op == "not_contains":
        return df[~s.astype(str).str.contains(str(val), na=False)]
    return df


def _step_drop_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Drop specified columns."""
    cols = cfg.get("columns", [])
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")


def _step_rename_column(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Rename a column."""
    return df.rename(columns={cfg["old_name"]: cfg["new_name"]})


def _step_fill_na(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Fill missing values in a column."""
    col = cfg["column"]
    method = cfg.get("method", "constant")
    if method == "constant":
        df[col] = df[col].fillna(cfg.get("value", ""))
    elif method == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif method == "median":
        df[col] = df[col].fillna(df[col].median())
    elif method == "mode":
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val.iloc[0])
    elif method == "ffill":
        df[col] = df[col].ffill()
    elif method == "bfill":
        df[col] = df[col].bfill()
    return df


def _step_label_encode(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Label-encode a categorical column."""
    col = cfg["column"]
    mapping = {v: i for i, v in enumerate(df[col].dropna().unique())}
    df[col] = df[col].map(mapping)
    return df


def _step_onehot_encode(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """One-hot encode a column."""
    col = cfg["column"]
    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def _step_normalize(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Min-max normalise a numeric column to [0, 1]."""
    col = cfg["column"]
    mn, mx = df[col].min(), df[col].max()
    if mx - mn != 0:
        df[col] = (df[col] - mn) / (mx - mn)
    return df


def _step_standardize(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Z-score standardise a numeric column."""
    col = cfg["column"]
    mean, std = df[col].mean(), df[col].std()
    if std != 0:
        df[col] = (df[col] - mean) / std
    return df


def _step_extract_datetime(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Extract year / month / day / weekday from a datetime column."""
    col = cfg["column"]
    dt = pd.to_datetime(df[col], errors="coerce")
    features = cfg.get("features", ["year", "month", "day", "weekday"])
    if "year" in features:
        df[f"{col}_year"] = dt.dt.year
    if "month" in features:
        df[f"{col}_month"] = dt.dt.month
    if "day" in features:
        df[f"{col}_day"] = dt.dt.day
    if "weekday" in features:
        df[f"{col}_weekday"] = dt.dt.weekday
    if "hour" in features:
        df[f"{col}_hour"] = dt.dt.hour
    return df


def _step_custom_formula(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create a new column using a pandas eval expression."""
    new_col = cfg.get("new_column", "new_col")
    formula = cfg["formula"]
    df[new_col] = df.eval(formula)
    return df


# Registry of step handlers
STEP_HANDLERS = {
    "filter_rows": _step_filter_rows,
    "drop_columns": _step_drop_columns,
    "rename_column": _step_rename_column,
    "fill_na": _step_fill_na,
    "label_encode": _step_label_encode,
    "onehot_encode": _step_onehot_encode,
    "normalize": _step_normalize,
    "standardize": _step_standardize,
    "extract_datetime": _step_extract_datetime,
    "custom_formula": _step_custom_formula,
}


# ---------------------------------------------------------------------------
# RecipeEngine
# ---------------------------------------------------------------------------

class RecipeEngine:
    """Executes data transformation recipes.

    Supported recipe types:
        - prepare: sequential step-based transformations
        - join: merge two DataFrames
        - groupby: aggregation
        - python: custom Python code executed via exec()
    """

    # -- Prepare recipe ------------------------------------------------------

    @staticmethod
    def execute_prepare(
        df: pd.DataFrame,
        steps: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Run a Prepare recipe (list of steps) on *df*.

        Args:
            df: Input DataFrame (will NOT be mutated — a copy is used).
            steps: Ordered list of step configs.

        Returns:
            Transformed DataFrame.
        """
        result = df.copy()
        for step in steps:
            step_type = step.get("type")
            handler = STEP_HANDLERS.get(step_type)
            if handler is None:
                raise ValueError(f"Unknown step type: {step_type}")
            result = handler(result, step.get("config", {}))
        return result

    @staticmethod
    def preview_prepare(
        df: pd.DataFrame,
        steps: List[Dict[str, Any]],
        n_rows: int = 5,
    ) -> pd.DataFrame:
        """Preview the first *n_rows* after applying steps."""
        result = RecipeEngine.execute_prepare(df.head(max(n_rows * 5, 200)), steps)
        return result.head(n_rows)

    # -- Join recipe ---------------------------------------------------------

    @staticmethod
    def execute_join(
        left: pd.DataFrame,
        right: pd.DataFrame,
        how: str = "inner",
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        suffixes: Tuple[str, str] = ("_left", "_right"),
    ) -> pd.DataFrame:
        """Merge two DataFrames.

        Args:
            left: Left DataFrame.
            right: Right DataFrame.
            how: Join type ('inner', 'left', 'right', 'outer').
            left_on: Column in *left* to join on.
            right_on: Column in *right* to join on.
            suffixes: Suffixes for overlapping column names.

        Returns:
            Merged DataFrame.
        """
        return pd.merge(
            left,
            right,
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
        )

    # -- Group By recipe -----------------------------------------------------

    @staticmethod
    def execute_groupby(
        df: pd.DataFrame,
        group_cols: List[str],
        agg_dict: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """Group by columns and aggregate.

        Args:
            df: Input DataFrame.
            group_cols: Columns to group by.
            agg_dict: ``{column: [func1, func2, ...]}``.

        Returns:
            Aggregated DataFrame with flattened column names.
        """
        result = df.groupby(group_cols).agg(agg_dict)
        # Flatten multi-level columns
        result.columns = ["_".join(col).strip("_") for col in result.columns.values]
        result = result.reset_index()
        return result

    # -- Custom Python recipe ------------------------------------------------

    @staticmethod
    def execute_python(
        df: pd.DataFrame,
        code: str,
        extra_ns: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Run user Python code.  The code should assign to ``output_df``.

        Available in the namespace: ``df`` (input), ``pd``, ``np``.

        Args:
            df: Input DataFrame.
            code: Python source code string.
            extra_ns: Extra variables to inject.

        Returns:
            The ``output_df`` produced by the code, or the (possibly mutated) *df*.
        """
        ns: Dict[str, Any] = {
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "output_df": None,
        }
        if extra_ns:
            ns.update(extra_ns)
        exec(code, ns)
        out = ns.get("output_df")
        if out is not None and isinstance(out, pd.DataFrame):
            return out
        return ns["df"]

    # -- Generic dispatcher --------------------------------------------------

    @staticmethod
    def execute_recipe(
        recipe_type: str,
        config: Dict[str, Any],
        datasets: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Execute a recipe by type, pulling inputs from *datasets*.

        Args:
            recipe_type: One of 'prepare', 'join', 'groupby', 'python'.
            config: Recipe-specific configuration dict.
            datasets: Name → DataFrame mapping for inputs.

        Returns:
            Output DataFrame.
        """
        if recipe_type == "prepare":
            input_name = config["input"]
            steps = config.get("steps", [])
            return RecipeEngine.execute_prepare(datasets[input_name], steps)

        elif recipe_type == "join":
            left = datasets[config["left"]]
            right = datasets[config["right"]]
            return RecipeEngine.execute_join(
                left,
                right,
                how=config.get("how", "inner"),
                left_on=config.get("left_on"),
                right_on=config.get("right_on"),
            )

        elif recipe_type == "groupby":
            input_name = config["input"]
            return RecipeEngine.execute_groupby(
                datasets[input_name],
                group_cols=config["group_cols"],
                agg_dict=config["agg_dict"],
            )

        elif recipe_type == "python":
            input_name = config.get("input")
            df = datasets[input_name] if input_name else pd.DataFrame()
            return RecipeEngine.execute_python(df, config.get("code", ""))

        else:
            raise ValueError(f"Unknown recipe type: {recipe_type}")


# ---------------------------------------------------------------------------
# Recipe config builder helpers (for the GUI)
# ---------------------------------------------------------------------------

def new_prepare_step(step_type: str, **kwargs: Any) -> Dict[str, Any]:
    """Return a step config dict."""
    return {"type": step_type, "config": kwargs, "id": uuid.uuid4().hex[:8]}


def new_recipe_config(
    recipe_type: str,
    input_name: str = "",
    output_name: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return a full recipe config dict ready for persistence."""
    return {
        "id": uuid.uuid4().hex[:10],
        "type": recipe_type,
        "input": input_name,
        "output": output_name,
        **kwargs,
    }
