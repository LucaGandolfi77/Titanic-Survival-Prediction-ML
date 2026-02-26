"""Tests for core/project.py"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.project import Project, ProjectManager, ProjectMeta


# ---------------------------------------------------------------------------
# ProjectMeta
# ---------------------------------------------------------------------------

class TestProjectMeta:
    def test_creation(self) -> None:
        meta = ProjectMeta(name="demo", uid="abc123", path=Path("/tmp/demo"),
                          created="2024-01-01", modified="2024-01-01")
        assert meta.name == "demo"
        assert meta.uid == "abc123"

    def test_to_from_dict(self) -> None:
        meta = ProjectMeta(name="test_proj", uid="xyz", path=Path("/tmp/tp"),
                          created="2024-01-01", modified="2024-01-02")
        d = meta.to_dict()
        assert d["name"] == "test_proj"
        assert d["uid"] == "xyz"


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

def _make_meta(name="p1"):
    return ProjectMeta(name=name, uid="uid1", path=Path("/tmp/p1"),
                       created="2024-01-01", modified="2024-01-01")


class TestProject:
    def test_defaults(self) -> None:
        p = Project(meta=_make_meta("p1"))
        assert p.meta.name == "p1"
        assert p.datasets == {}
        assert p.recipes == []
        assert p.models == []
        assert p.flow == {"nodes": [], "edges": []}
        assert p.notebook_cells == []
        assert p.charts == []

    def test_to_from_dict_roundtrip(self) -> None:
        p = Project(meta=_make_meta("round"))
        p.datasets = {"ds1": {"path": "/tmp/a.csv"}}
        p.recipes = [{"type": "prepare", "steps": []}]
        p.models = [{"name": "m1"}]
        p.flow = {"nodes": [{"id": "n1"}], "edges": []}
        p.notebook_cells = [{"source": "x=1"}]
        p.charts = [{"type": "scatter"}]

        d = p.to_dict()
        p2 = Project.from_dict(d)
        assert p2.meta.name == "round"
        assert p2.datasets == p.datasets
        assert p2.recipes == p.recipes
        assert p2.models == p.models
        assert p2.flow == p.flow
        assert p2.notebook_cells == p.notebook_cells
        assert p2.charts == p.charts


# ---------------------------------------------------------------------------
# ProjectManager
# ---------------------------------------------------------------------------

class TestProjectManager:
    """Each test gets a fresh temp directory as the project root."""

    @pytest.fixture(autouse=True)
    def tmp_root(self, tmp_path) -> None:
        self.root = Path(tmp_path)
        self.pm = ProjectManager(base_dir=self.root)

    # -- create --------------------------------------------------------
    def test_create_project(self) -> None:
        proj = self.pm.create_project("alpha")
        assert proj.meta.name == "alpha"
        assert Path(proj.meta.path).is_dir()

    def test_create_writes_json(self) -> None:
        proj = self.pm.create_project("beta")
        json_path = Path(proj.meta.path) / "project.json"
        assert json_path.is_file()
        data = json.loads(json_path.read_text())
        assert data["meta"]["name"] == "beta"

    # -- list ----------------------------------------------------------
    def test_list_projects(self) -> None:
        self.pm.create_project("a")
        self.pm.create_project("b")
        names = [m.name for m in self.pm.list_projects()]
        assert "a" in names and "b" in names

    # -- open ----------------------------------------------------------
    def test_open_project(self) -> None:
        proj = self.pm.create_project("gamma")
        loaded = self.pm.open_project(Path(proj.meta.path))
        assert loaded.meta.name == "gamma"

    def test_open_nonexistent(self) -> None:
        with pytest.raises(FileNotFoundError):
            self.pm.open_project(Path("/tmp/nope-does-not-exist"))

    # -- save ----------------------------------------------------------
    def test_save_project(self) -> None:
        proj = self.pm.create_project("delta")
        proj.datasets = {"ds": {"cols": ["a", "b"]}}
        self.pm.save_project(proj)
        reloaded = self.pm.open_project(Path(proj.meta.path))
        assert reloaded.datasets == {"ds": {"cols": ["a", "b"]}}

    # -- delete --------------------------------------------------------
    def test_delete_project(self) -> None:
        proj = self.pm.create_project("epsilon")
        proj_path = Path(proj.meta.path)
        self.pm.delete_project(proj_path)
        assert not proj_path.exists()

    def test_delete_nonexistent(self) -> None:
        # should not raise
        self.pm.delete_project(Path("/tmp/no-such-id"))

    # -- current project -----------------------------------------------
    def test_current_project(self) -> None:
        proj = self.pm.create_project("zeta")
        assert self.pm.current_project.meta.name == "zeta"
