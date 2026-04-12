from __future__ import annotations

import unittest
from pathlib import Path
import shutil
from uuid import uuid4
from unittest.mock import patch

import gemini_vistext_runner_core as core
from gemini_vistext_runner_core import PromptBuilderContext, PromptPackage, RuntimeConfig


class ProcessImageRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        repo_tmp = Path(__file__).resolve().parents[2] / ".tmp"
        repo_tmp.mkdir(exist_ok=True)
        self.root = repo_tmp / f"vistext-runner-test-{uuid4().hex}"
        self.root.mkdir()
        self.images_root = self.root / "images"
        self.labels_root = self.root / "labels"
        self.output_dir = self.root / "outputs"
        self.images_root.mkdir()
        self.labels_root.mkdir()
        self.output_dir.mkdir()

        self.image_path = self.images_root / "1046.png"
        self.label_path = self.labels_root / "1046.json"
        self.image_path.write_bytes(b"fake image bytes")
        self.label_path.write_text("{}", encoding="utf-8")

        self.config = RuntimeConfig(
            model="gemini-3-flash-preview",
            images_root=self.images_root,
            labels_root=self.labels_root,
            output_dir=self.output_dir,
            manifest_path=self.output_dir / "manifest.json",
            request_delay=0.0,
            sample_mode="ids",
            sample_count=None,
            seed=42,
            ids=["1046"],
            parallel_workers=1,
            skip_existing=False,
            dry_run=False,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def prompt_builder(self, _: Path, __: PromptBuilderContext) -> PromptPackage:
        return PromptPackage(system_prompt="system", examples=[], metadata={"strategy": "test"})

    def test_retries_invalid_turtle_until_success(self) -> None:
        with (
            patch.object(core, "create_gemini_client", return_value=object()),
            patch.object(core, "call_gemini", side_effect=["bad one", "bad two", "good ttl"]),
            patch.object(core, "validate_turtle", side_effect=["parse error 1", "parse error 2", None]),
        ):
            item = core.process_image(
                client=None,
                config=self.config,
                prompt_builder=self.prompt_builder,
                image_path=self.image_path,
            )

        self.assertEqual(item["status"], "saved")
        self.assertEqual(item["attempt_count"], 3)
        self.assertEqual(len(item["attempt_failures"]), 2)
        self.assertEqual(item["attempt_failures"][0]["status"], "invalid_ttl")
        self.assertTrue((self.output_dir / "1046.ttl").exists())

    def test_returns_retry_exhausted_for_mixed_failures(self) -> None:
        with (
            patch.object(core, "create_gemini_client", return_value=object()),
            patch.object(
                core,
                "call_gemini",
                side_effect=[RuntimeError("api fail 1"), "bad ttl", RuntimeError("api fail 2")],
            ),
            patch.object(core, "validate_turtle", return_value="parse error"),
        ):
            item = core.process_image(
                client=None,
                config=self.config,
                prompt_builder=self.prompt_builder,
                image_path=self.image_path,
            )

        self.assertEqual(item["status"], "retry_exhausted")
        self.assertEqual(item["attempt_count"], 3)
        self.assertEqual(len(item["attempt_failures"]), 3)
        self.assertEqual(item["attempt_failures"][1]["status"], "invalid_ttl")
        self.assertEqual(item["error"], "api fail 2")
        self.assertFalse((self.output_dir / "1046.ttl").exists())

    def test_retries_prompt_builder_api_error(self) -> None:
        prompt_builder_calls = {"count": 0}

        def flaky_prompt_builder(_: Path, __: PromptBuilderContext) -> PromptPackage:
            prompt_builder_calls["count"] += 1
            if prompt_builder_calls["count"] == 1:
                raise RuntimeError("classification failed")
            return PromptPackage(system_prompt="system", examples=[], metadata={"chart_type": "bar"})

        with (
            patch.object(core, "create_gemini_client", return_value=object()),
            patch.object(core, "call_gemini", return_value="good ttl"),
            patch.object(core, "validate_turtle", return_value=None),
        ):
            item = core.process_image(
                client=None,
                config=self.config,
                prompt_builder=flaky_prompt_builder,
                image_path=self.image_path,
            )

        self.assertEqual(item["status"], "saved")
        self.assertEqual(item["attempt_count"], 2)
        self.assertEqual(len(item["attempt_failures"]), 1)
        self.assertEqual(item["attempt_failures"][0]["stage"], "prompt_builder")
        self.assertEqual(item["attempt_failures"][0]["status"], "api_error")


if __name__ == "__main__":
    unittest.main()
