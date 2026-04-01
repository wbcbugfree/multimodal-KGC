"""Path helpers for the repository's dataset-first layout."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root for this checkout."""
    return Path(__file__).resolve().parent.parent


def docs_dir() -> Path:
    return repo_root() / "docs"


def vistext_dir() -> Path:
    return repo_root() / "vistext"


def vistext_data_dir() -> Path:
    return vistext_dir() / "data"


def vistext_json2ttl_dir() -> Path:
    return vistext_data_dir() / "json2ttl"


def vistext_images_dir() -> Path:
    return vistext_data_dir() / "images"


def vistext_labels_dir() -> Path:
    return vistext_data_dir() / "labels"


def soil_dataset_dir() -> Path:
    return repo_root() / "soil_dataset"


def soil_data_dir() -> Path:
    return soil_dataset_dir() / "data"


def diagram2graph_dataset_dir() -> Path:
    return repo_root() / "diagram2graph_dataset"


def diagram2graph_data_dir() -> Path:
    return diagram2graph_dataset_dir() / "data"
