"""Tests for core/dataset.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from core.dataset import DatasetManager, DatasetInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [88.5, 92.0, 76.3, 95.1, 81.0],
    })


@pytest.fixture
def manager(sample_df: pd.DataFrame) -> DatasetManager:
    dm = DatasetManager()
    dm.register(sample_df, "test")
    return dm


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_creates_info(self, manager: DatasetManager) -> None:
        info = manager.get_info("test")
        assert info.name == "test"
        assert info.row_count == 5
        assert info.col_count == 3

    def test_register_unique_names(self, manager: DatasetManager, sample_df: pd.DataFrame) -> None:
        info2 = manager.register(sample_df, "test2")
        assert info2.name == "test2"
        assert "test" in manager.list_names()
        assert "test2" in manager.list_names()

    def test_list_names(self, manager: DatasetManager) -> None:
        names = manager.list_names()
        assert "test" in names

    def test_has(self, manager: DatasetManager) -> None:
        assert manager.has("test")
        assert not manager.has("nonexistent")


# ---------------------------------------------------------------------------
# Preview / pagination
# ---------------------------------------------------------------------------

class TestPreview:
    def test_preview_first_page(self, manager: DatasetManager) -> None:
        page_df, total = manager.preview("test", page=0, page_size=3)
        assert len(page_df) == 3
        assert total == 2  # ceil(5/3) = 2

    def test_preview_last_page(self, manager: DatasetManager) -> None:
        page_df, total = manager.preview("test", page=1, page_size=3)
        assert len(page_df) == 2
        assert total == 2

    def test_full_page(self, manager: DatasetManager) -> None:
        page_df, total = manager.preview("test", page=0, page_size=100)
        assert len(page_df) == 5
        assert total == 1


# ---------------------------------------------------------------------------
# Column stats
# ---------------------------------------------------------------------------

class TestColumnStatistics:
    def test_numeric_column(self, manager: DatasetManager) -> None:
        stats = manager.column_statistics("test", "age")
        assert stats["null_count"] == 0
        assert stats["unique_count"] == 5
        assert "mean" in stats

    def test_string_column(self, manager: DatasetManager) -> None:
        stats = manager.column_statistics("test", "name")
        assert stats["null_count"] == 0
        assert "mean" not in stats


# ---------------------------------------------------------------------------
# Schema editing
# ---------------------------------------------------------------------------

class TestSchemaEditing:
    def test_rename_column(self, manager: DatasetManager) -> None:
        manager.rename_column("test", "age", "years")
        df = manager.get_df("test")
        assert "years" in df.columns
        assert "age" not in df.columns

    def test_rename_nonexistent_raises(self, manager: DatasetManager) -> None:
        with pytest.raises(KeyError):
            manager.rename_column("test", "nonexistent", "new")

    def test_set_column_role(self, manager: DatasetManager) -> None:
        manager.set_column_role("test", "score", "target")
        info = manager.get_info("test")
        score_col = [c for c in info.columns if c["name"] == "score"][0]
        assert score_col["role"] == "target"

    def test_change_dtype(self, manager: DatasetManager) -> None:
        manager.change_dtype("test", "age", "float64")
        df = manager.get_df("test")
        assert df["age"].dtype == "float64"


# ---------------------------------------------------------------------------
# Update & remove
# ---------------------------------------------------------------------------

class TestUpdateRemove:
    def test_update_df(self, manager: DatasetManager) -> None:
        new_df = pd.DataFrame({"a": [1, 2]})
        manager.update_df("test", new_df)
        assert manager.get_info("test").row_count == 2

    def test_remove(self, manager: DatasetManager) -> None:
        manager.remove("test")
        assert not manager.has("test")
        assert "test" not in manager.list_names()


# ---------------------------------------------------------------------------
# Import from file
# ---------------------------------------------------------------------------

class TestImportFile:
    def test_import_csv(self, sample_df: pd.DataFrame) -> None:
        dm = DatasetManager()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            sample_df.to_csv(path, index=False)
            info = dm.import_file(path)
            assert info.row_count == 5
            assert dm.has(info.name)

    def test_import_nonexistent_raises(self) -> None:
        dm = DatasetManager()
        with pytest.raises(FileNotFoundError):
            dm.import_file(Path("/tmp/no_such_file.csv"))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_csv(self, manager: DatasetManager) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            manager.export("test", path)
            assert path.exists()
            df2 = pd.read_csv(path)
            assert len(df2) == 5


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_datasets_to_dict(self, manager: DatasetManager) -> None:
        d = manager.datasets_to_dict()
        assert "test" in d
        assert d["test"]["row_count"] == 5

    def test_save_load_data_dir(self, manager: DatasetManager) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            manager.save_data_to_dir(data_dir)
            info_dict = manager.datasets_to_dict()

            dm2 = DatasetManager()
            dm2.load_data_from_dir(data_dir, info_dict)
            assert dm2.has("test")
            assert dm2.get_info("test").row_count == 5


# ---------------------------------------------------------------------------
# DatasetInfo serialisation
# ---------------------------------------------------------------------------

class TestDatasetInfo:
    def test_to_from_dict(self) -> None:
        info = DatasetInfo(
            name="ds", uid="abc", source_path="/tmp/x.csv",
            row_count=10, col_count=3, memory="1.0 KB",
            columns=[{"name": "a", "dtype": "int64", "semantic_type": "numeric", "role": "feature"}],
        )
        d = info.to_dict()
        info2 = DatasetInfo.from_dict(d)
        assert info2.name == "ds"
        assert info2.row_count == 10
        assert len(info2.columns) == 1
