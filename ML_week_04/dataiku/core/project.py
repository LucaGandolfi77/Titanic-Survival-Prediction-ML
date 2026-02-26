"""
ProjectManager – handles creating, opening, saving, and listing
DataikuLite projects stored as JSON bundles on disk.
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.helpers import NumpyEncoder, load_json, save_json, timestamp_str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROJECTS_DIR = Path.home() / ".dataikulite" / "projects"
PROJECT_FILE = "project.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ProjectMeta:
    """Lightweight metadata for a project shown in the explorer."""

    def __init__(self, name: str, uid: str, path: Path, created: str, modified: str) -> None:
        self.name = name
        self.uid = uid
        self.path = path
        self.created = created
        self.modified = modified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "uid": self.uid,
            "path": str(self.path),
            "created": self.created,
            "modified": self.modified,
        }


class Project:
    """In-memory representation of an open project.

    Attributes:
        meta: ProjectMeta with name / timestamps.
        datasets: mapping *name → serialised dataset info*.
        recipes: list of recipe config dicts.
        models: list of model config dicts.
        flow: list of node/edge dicts for the canvas.
        notebook_cells: list of code-cell dicts.
    """

    def __init__(self, meta: ProjectMeta) -> None:
        self.meta = meta
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.recipes: List[Dict[str, Any]] = []
        self.models: List[Dict[str, Any]] = []
        self.flow: Dict[str, Any] = {"nodes": [], "edges": []}
        self.notebook_cells: List[Dict[str, Any]] = []
        self.charts: List[Dict[str, Any]] = []

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the whole project to a JSON-safe dict."""
        return {
            "meta": self.meta.to_dict(),
            "datasets": self.datasets,
            "recipes": self.recipes,
            "models": self.models,
            "flow": self.flow,
            "notebook_cells": self.notebook_cells,
            "charts": self.charts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Reconstruct a Project from a dict (loaded from JSON)."""
        m = data["meta"]
        meta = ProjectMeta(
            name=m["name"],
            uid=m["uid"],
            path=Path(m["path"]),
            created=m["created"],
            modified=m["modified"],
        )
        proj = cls(meta)
        proj.datasets = data.get("datasets", {})
        proj.recipes = data.get("recipes", [])
        proj.models = data.get("models", [])
        proj.flow = data.get("flow", {"nodes": [], "edges": []})
        proj.notebook_cells = data.get("notebook_cells", [])
        proj.charts = data.get("charts", [])
        return proj


# ---------------------------------------------------------------------------
# ProjectManager
# ---------------------------------------------------------------------------

class ProjectManager:
    """Handles CRUD operations for DataikuLite projects."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir: Path = base_dir or DEFAULT_PROJECTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_project: Optional[Project] = None

    # -- listing -------------------------------------------------------------

    def list_projects(self) -> List[ProjectMeta]:
        """Return metadata for every project in *base_dir*."""
        projects: List[ProjectMeta] = []
        if not self.base_dir.exists():
            return projects
        for child in sorted(self.base_dir.iterdir()):
            pf = child / PROJECT_FILE
            if pf.exists():
                try:
                    data = load_json(pf)
                    m = data["meta"]
                    projects.append(
                        ProjectMeta(
                            name=m["name"],
                            uid=m["uid"],
                            path=Path(m["path"]),
                            created=m["created"],
                            modified=m["modified"],
                        )
                    )
                except Exception:
                    pass  # skip corrupt projects
        return projects

    # -- create --------------------------------------------------------------

    def create_project(self, name: str) -> Project:
        """Create a new empty project and persist it.

        Args:
            name: Human-readable project name.

        Returns:
            The newly created Project.
        """
        uid = uuid.uuid4().hex[:12]
        proj_dir = self.base_dir / f"{name}_{uid}"
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "data").mkdir(exist_ok=True)
        (proj_dir / "models").mkdir(exist_ok=True)

        now = timestamp_str()
        meta = ProjectMeta(name=name, uid=uid, path=proj_dir, created=now, modified=now)
        project = Project(meta)
        self._save(project)
        self.current_project = project
        return project

    # -- open / save ---------------------------------------------------------

    def open_project(self, path: Path) -> Project:
        """Open a project from *path* (directory containing project.json).

        Args:
            path: Directory of the project.

        Returns:
            The loaded Project.
        """
        pf = path / PROJECT_FILE
        if not pf.exists():
            raise FileNotFoundError(f"No project.json found in {path}")
        data = load_json(pf)
        project = Project.from_dict(data)
        project.meta.path = path  # normalise
        self.current_project = project
        return project

    def save_project(self, project: Optional[Project] = None) -> None:
        """Persist the given (or current) project to disk."""
        project = project or self.current_project
        if project is None:
            raise RuntimeError("No project is currently open.")
        project.meta.modified = timestamp_str()
        self._save(project)

    def _save(self, project: Project) -> None:
        proj_dir = Path(project.meta.path)
        proj_dir.mkdir(parents=True, exist_ok=True)
        save_json(project.to_dict(), proj_dir / PROJECT_FILE)

    # -- delete --------------------------------------------------------------

    def delete_project(self, path: Path) -> None:
        """Remove a project directory entirely."""
        if path.exists():
            shutil.rmtree(path)
