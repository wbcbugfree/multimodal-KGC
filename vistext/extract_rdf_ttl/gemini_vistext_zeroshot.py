from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gemini_vistext_runner_core import PROMPT_ROOT, PromptPackage, load_text, run_strategy  # noqa: E402


OUTPUT_DIR = SCRIPT_DIR / "vistext_zeroshot_outputs"
SYSTEM_PROMPT_PATH = PROMPT_ROOT / "zeroshot_system.md"
DESCRIPTION = "Call Gemini on vistext chart images with zero-shot prompting and save only valid Turtle outputs."


def build_prompt_package(_: Path, __: object | None = None) -> PromptPackage:
    return PromptPackage(
        system_prompt=load_text(SYSTEM_PROMPT_PATH),
        examples=[],
    )


def main(argv: list[str] | None = None) -> int:
    return run_strategy(
        argv=argv,
        description=DESCRIPTION,
        default_output_dir=OUTPUT_DIR,
        prompt_builder=build_prompt_package,
    )


if __name__ == "__main__":
    raise SystemExit(main())
