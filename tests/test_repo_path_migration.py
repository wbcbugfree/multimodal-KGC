from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path, PurePosixPath

from scripts.repo_path_migration import (
    canonicalize_relative_path,
    canonicalize_segment,
    rewrite_notebook_payload,
    safe_move_file,
)


class RepoPathMigrationTests(unittest.TestCase):
    def test_canonicalize_segment_fixes_case_spaces_and_typos(self) -> None:
        self.assertEqual(canonicalize_segment("Extract RDF ttl"), "extract_rdf_ttl")
        self.assertEqual(canonicalize_segment("Onehot_outputs"), "oneshot_outputs")
        self.assertEqual(
            canonicalize_segment("vistext_Oneshot_sta_outputs"),
            "vistext_oneshot_static_outputs",
        )
        self.assertEqual(
            canonicalize_segment("diagram2graph-inferance.ipynb", is_file=True),
            "diagram2graph_inference.ipynb",
        )

    def test_canonicalize_relative_path_normalizes_dataset_tree(self) -> None:
        original = PurePosixPath(
            "diagram2graph Dataset/Extrcat RDF json/diagram2graph-inferance.ipynb"
        )
        expected = PurePosixPath(
            "diagram2graph_dataset/extract_rdf_json/diagram2graph_inference.ipynb"
        )
        self.assertEqual(canonicalize_relative_path(original), expected)

    def test_rewrite_notebook_payload_updates_paths_and_clears_outputs(self) -> None:
        payload = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["Path('VisText') / 'Prompt Text' / 'dynamic_OneShot'"],
                    "outputs": [{"output_type": "stream", "text": "stale"}],
                    "execution_count": 7,
                },
                {
                    "cell_type": "markdown",
                    "source": ["See diagram2graph Dataset/Extrcat RDF json for details."],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        rewritten = rewrite_notebook_payload(
            payload,
            full_replacements=[
                (
                    "diagram2graph Dataset/Extrcat RDF json",
                    "diagram2graph_dataset/extract_rdf_json",
                )
            ],
            segment_replacements=[
                ("VisText", "vistext"),
                ("Prompt Text", "prompt_text"),
                ("dynamic_OneShot", "dynamic_oneshot"),
            ],
        )

        self.assertEqual(
            rewritten["cells"][0]["source"],
            ["Path('vistext') / 'prompt_text' / 'dynamic_oneshot'"],
        )
        self.assertEqual(rewritten["cells"][0]["outputs"], [])
        self.assertIsNone(rewritten["cells"][0]["execution_count"])
        self.assertEqual(
            rewritten["cells"][1]["source"],
            ["See diagram2graph_dataset/extract_rdf_json for details."],
        )

    def test_safe_move_file_is_resumable_when_target_already_exists(self) -> None:
        repo_root = Path(__file__).resolve().parents[1] / ".tmp" / f"migration-test-{uuid.uuid4().hex}"
        repo_root.mkdir(parents=True, exist_ok=True)
        try:
            source = repo_root / "VisText" / "Prompt Text" / "note.txt"
            target = repo_root / "vistext" / "prompt_text" / "note.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("migrated", encoding="utf-8")

            safe_move_file(
                repo_root,
                PurePosixPath("VisText/Prompt Text/note.txt"),
                PurePosixPath("vistext/prompt_text/note.txt"),
            )

            self.assertFalse(source.exists())
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "migrated")
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
