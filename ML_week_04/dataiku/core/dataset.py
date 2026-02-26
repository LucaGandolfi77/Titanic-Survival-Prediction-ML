"""
DatasetManager – in-memory dataset registry with import, preview,
schema editing, and column statistics.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.helpers import (
    column_stats,
    dataframe_memory,
    detect_column_type,
    read_dataframe,
    write_dataframe,
)


# ---------------------------------------------------------------------------
# DatasetInfo – metadata wrapper
# ---------------------------------------------------------------------------

class DatasetInfo:
    """Metadata for a loaded dataset."""

    def __init__(
        self,
        name: str,
        uid: str,
        source_path: Optional[str],
        row_count: int,
        col_count: int,
        memory: str,
        columns: List[Dict[str, Any]],
    ) -> None:
        self.name = name
        self.uid = uid
        self.source_path = source_path
        self.row_count = row_count
        self.col_count = col_count
        self.memory = memory
        self.columns = columns  # [{name, dtype, semantic_type, role}]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "uid": self.uid,
            "source_path": self.source_path,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "memory": self.memory,
            "columns": self.columns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        return cls(**data)


# ---------------------------------------------------------------------------
# DatasetManager
# ---------------------------------------------------------------------------

class DatasetManager:
    """Manages dataset import, storage, preview, and schema editing.

    Datasets are kept in-memory as pandas DataFrames and also referenced
    by name for the flow canvas and ML lab.
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._info: Dict[str, DatasetInfo] = {}

    # -- import --------------------------------------------------------------

    def import_file(self, path: Path, name: Optional[str] = None) -> DatasetInfo:
        """Import a data file and register it.

        Args:
            path: File system path to a supported data file.
            name: Optional display name; defaults to the file stem.

        Returns:
            DatasetInfo describing the imported dataset.
        """
        df = read_dataframe(path)
        name = name or path.stem
        # Ensure unique name
        if name in self._datasets:
            name = f"{name}_{uuid.uuid4().hex[:6]}"
        return self.register(df, name, source_path=str(path))

    def register(
        self,
        df: pd.DataFrame,
        name: str,
        source_path: Optional[str] = None,
    ) -> DatasetInfo:
        """Register a DataFrame under *name*.

        Args:
            df: pandas DataFrame.
            name: Display name.
            source_path: Where the data originated (can be None for derived).

        Returns:
            DatasetInfo for the newly registered dataset.
        """
        uid = uuid.uuid4().hex[:12]
        cols: List[Dict[str, Any]] = []
        for c in df.columns:
            cols.append({
                "name": c,
                "dtype": str(df[c].dtype),
                "semantic_type": detect_column_type(df[c]),
                "role": "feature",  # default
            })
        info = DatasetInfo(
            name=name,
            uid=uid,
            source_path=source_path,
            row_count=len(df),
            col_count=len(df.columns),
            memory=dataframe_memory(df),
            columns=cols,
        )
        self._datasets[name] = df
        self._info[name] = info
        return info

    # -- access --------------------------------------------------------------

    def get_df(self, name: str) -> pd.DataFrame:
        """Return the DataFrame for *name*."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found.")
        return self._datasets[name]

    def get_info(self, name: str) -> DatasetInfo:
        """Return DatasetInfo for *name*."""
        return self._info[name]

    def list_names(self) -> List[str]:
        """Return a sorted list of all registered dataset names."""
        return sorted(self._datasets.keys())

    def has(self, name: str) -> bool:
        return name in self._datasets

    # -- preview / pagination ------------------------------------------------

    def preview(
        self,
        name: str,
        page: int = 0,
        page_size: int = 100,
    ) -> Tuple[pd.DataFrame, int]:
        """Return a page of data and total number of pages.

        Args:
            name: Dataset name.
            page: Zero-based page index.
            page_size: Rows per page.

        Returns:
            (page_df, total_pages)
        """
        df = self.get_df(name)
        total_pages = max(1, -(-len(df) // page_size))  # ceil division
        start = page * page_size
        end = start + page_size
        return df.iloc[start:end], total_pages

    # -- column stats --------------------------------------------------------

    def column_statistics(self, name: str, col: str) -> Dict[str, Any]:
        """Return statistics for column *col* in dataset *name*."""
        return column_stats(self.get_df(name), col)

    # -- schema editing ------------------------------------------------------

    def rename_column(self, name: str, old_col: str, new_col: str) -> None:
        """Rename a column in the dataset."""
        df = self.get_df(name)
        if old_col not in df.columns:
            raise KeyError(f"Column '{old_col}' not in dataset '{name}'.")
        df.rename(columns={old_col: new_col}, inplace=True)
        # Update info
        for c in self._info[name].columns:
            if c["name"] == old_col:
                c["name"] = new_col
                break
        self._refresh_info(name)

    def change_dtype(self, name: str, col: str, new_dtype: str) -> None:
        """Change the pandas dtype of a column."""
        df = self.get_df(name)
        if new_dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif new_dtype == "category":
            df[col] = df[col].astype("category")
        else:
            df[col] = df[col].astype(new_dtype)
        self._refresh_info(name)

    def set_column_role(self, name: str, col: str, role: str) -> None:
        """Set column role to 'feature', 'target', or 'ignore'."""
        info = self._info[name]
        for c in info.columns:
            if c["name"] == col:
                c["role"] = role
                break

    # -- update / replace ----------------------------------------------------

    def update_df(self, name: str, df: pd.DataFrame) -> DatasetInfo:
        """Replace the DataFrame for an existing dataset."""
        self._datasets[name] = df
        self._refresh_info(name)
        return self._info[name]

    # -- export --------------------------------------------------------------

    def export(self, name: str, path: Path) -> None:
        """Export dataset to a file."""
        write_dataframe(self.get_df(name), path)

    # -- remove --------------------------------------------------------------

    def remove(self, name: str) -> None:
        """Remove a dataset from the registry."""
        self._datasets.pop(name, None)
        self._info.pop(name, None)

    # -- internal ------------------------------------------------------------

    def _refresh_info(self, name: str) -> None:
        """Recompute info metadata from the current DataFrame."""
        df = self._datasets[name]
        info = self._info[name]
        info.row_count = len(df)
        info.col_count = len(df.columns)
        info.memory = dataframe_memory(df)
        # Rebuild column list keeping roles
        old_roles = {c["name"]: c.get("role", "feature") for c in info.columns}
        cols: List[Dict[str, Any]] = []
        for c in df.columns:
            cols.append({
                "name": c,
                "dtype": str(df[c].dtype),
                "semantic_type": detect_column_type(df[c]),
                "role": old_roles.get(c, "feature"),
            })
        info.columns = cols

    # -- serialisation helpers (for project save/load) -----------------------

    def datasets_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialise all dataset *infos* (NOT the data) to a dict."""
        return {n: i.to_dict() for n, i in self._info.items()}

    def save_data_to_dir(self, directory: Path) -> None:
        """Persist all DataFrames as parquet files in *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        for name, df in self._datasets.items():
            df.to_parquet(directory / f"{name}.parquet", index=False)

    def load_data_from_dir(self, directory: Path, info_dict: Dict[str, Dict[str, Any]]) -> None:
        """Reload datasets from parquet files stored by *save_data_to_dir*."""
        for name, meta in info_dict.items():
            pq = directory / f"{name}.parquet"
            if pq.exists():
                df = pd.read_parquet(pq)
                self._datasets[name] = df
                self._info[name] = DatasetInfo.from_dict(meta)
