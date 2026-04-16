from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gemini_json_runner_core import (  # noqa: E402
    OUTPUTS_ROOT,
    SYSTEM_PROMPT_PATH,
    PromptPackage,
    ground_truth_example,
    load_text,
    run_strategy,
)


OUTPUT_DIR = OUTPUTS_ROOT / "gemini_oneshot_json"
DESCRIPTION = "Call Gemini on the Diagram2Graph ablation images with static one-shot JSON prompting."
ONESHOT_EXAMPLE_ID = "3"


def build_prompt_package(_: Path, __: object | None = None) -> PromptPackage:
    return PromptPackage(
        system_prompt=load_text(SYSTEM_PROMPT_PATH),
        examples=[ground_truth_example(ONESHOT_EXAMPLE_ID)],
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
