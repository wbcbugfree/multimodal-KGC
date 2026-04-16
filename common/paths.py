"""Path helpers for the repository's dataset-first layout."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root for this checkout."""
    return Path(__file__).resolve().parent.parent


def vistext_dir() -> Path:
    return repo_root() / "vistext"


def vistext_data_dir() -> Path:
    return vistext_dir() / "data"


def vistext_split_dir(split: str = "test") -> Path:
    return vistext_data_dir() / split


def vistext_images_dir(split: str = "test") -> Path:
    return vistext_split_dir(split) / "images"


def vistext_labels_dir(split: str = "test") -> Path:
    return vistext_split_dir(split) / "labels"


def vistext_turtle_dir(split: str = "test") -> Path:
    return vistext_split_dir(split) / "turtle"


def soil_health_dir() -> Path:
    return repo_root() / "soil_health"
