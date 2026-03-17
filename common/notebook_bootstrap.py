"""Bootstrap helpers for notebooks and ad hoc scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .paths import repo_root


def ensure_repo_on_sys_path() -> Path:
    """Add the repository root to sys.path if needed."""
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def chdir_repo_root() -> Path:
    """Change the current working directory to the repository root."""
    root = repo_root()
    os.chdir(root)
    return root


def in_notebook() -> bool:
    """Return True when running inside a Jupyter notebook kernel."""
    return "ipykernel" in sys.modules or "JPY_PARENT_PID" in os.environ


def bootstrap(change_cwd: bool = False) -> Path:
    """Prepare imports for notebooks and scripts and optionally chdir to the repo root."""
    root = ensure_repo_on_sys_path()
    if change_cwd:
        os.chdir(root)
    return root
