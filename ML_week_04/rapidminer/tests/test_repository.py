"""
test_repository.py – Tests for persistence layer (repository.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.process_runner import Process
from engine.operator_base import Connection, get_operator_class
from engine import repository as repo


class TestSaveLoadProcess:
    def test_roundtrip(self, tmp_dir):
        proc = Process("Test Process")
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", "/tmp/data.csv")
        proc.add_operator(op)

        path = repo.save_process(proc, path=tmp_dir / "test.rmp")
        assert path.exists()

        loaded = repo.load_process(path)
        assert loaded.name == "Test Process"
        assert len(loaded.operators) == 1

    def test_load_missing_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            repo.load_process(tmp_dir / "nope.rmp")

    def test_file_is_valid_json(self, tmp_dir):
        proc = Process("JSON Check")
        path = repo.save_process(proc, path=tmp_dir / "check.rmp")
        data = json.loads(path.read_text())
        assert data["name"] == "JSON Check"


class TestStoreLoadResult:
    def test_roundtrip_dataframe(self, tmp_dir, iris_df):
        original_dir = repo.RESULTS_DIR
        repo.RESULTS_DIR = tmp_dir
        try:
            repo.store_result("test_iris", iris_df)
            loaded = repo.load_result("test_iris")
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) == len(iris_df)
        finally:
            repo.RESULTS_DIR = original_dir

    def test_load_missing_raises(self, tmp_dir):
        original_dir = repo.RESULTS_DIR
        repo.RESULTS_DIR = tmp_dir
        try:
            with pytest.raises(FileNotFoundError):
                repo.load_result("does_not_exist")
        finally:
            repo.RESULTS_DIR = original_dir


class TestListSampleProcesses:
    def test_returns_list(self):
        samples = repo.list_sample_processes()
        assert isinstance(samples, list)
        # We know sample_processes/ exists with .rmp files
        if samples:
            assert all(p.suffix == ".rmp" for p in samples)


class TestDeleteProcess:
    def test_delete(self, tmp_dir):
        proc = Process("To Delete")
        path = repo.save_process(proc, path=tmp_dir / "delete_me.rmp")
        assert path.exists()
        repo.delete_process(path)
        assert not path.exists()
