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
    return vistext_turtle_dir("test")


def vistext_split_dir(split: str = "test") -> Path:
    return vistext_data_dir() / split


def vistext_images_dir(split: str = "test") -> Path:
    return vistext_split_dir(split) / "images"


def vistext_labels_dir(split: str = "test") -> Path:
    return vistext_split_dir(split) / "labels"


def vistext_turtle_dir(split: str = "test") -> Path:
    return vistext_split_dir(split) / "turtle"


def vistext_json2ttl_converter_path() -> Path:
    return vistext_data_dir() / "json2ttl_converter.py"


def soil_dataset_dir() -> Path:
    return repo_root() / "soil_dataset"


def soil_data_dir() -> Path:
    return soil_dataset_dir() / "data"


def soil_health_dir() -> Path:
    return repo_root() / "soil_health"


def soil_health_data_dir() -> Path:
    return soil_health_dir() / "data"


def soil_health_figures_dir() -> Path:
    return soil_health_data_dir() / "figures"


def soil_health_tables_dir() -> Path:
    return soil_health_data_dir() / "tables"


def diagram2graph_dataset_dir() -> Path:
    return repo_root() / "diagram2graph_dataset"


def diagram2graph_data_dir() -> Path:
    return diagram2graph_dataset_dir() / "data"
