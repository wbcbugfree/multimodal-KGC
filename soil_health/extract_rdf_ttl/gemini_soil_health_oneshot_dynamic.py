from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gemini_soil_health_runner_core import (  # noqa: E402
    DEFAULT_MODEL,
    PROMPT_ROOT,
    PromptBuilderContext,
    PromptPackage,
    ground_truth_example,
    load_text,
    run_strategy,
)


DYNAMIC_PROMPT_DIR = PROMPT_ROOT / "dynamic_oneshot"
if str(DYNAMIC_PROMPT_DIR) not in sys.path:
    sys.path.insert(0, str(DYNAMIC_PROMPT_DIR))

from categorize_image_type import classify_image_type  # noqa: E402


OUTPUT_DIR = SCRIPT_DIR / "oneshot_dynamic"
DESCRIPTION = "Call Gemini on soil-health images with dynamic one-shot prompting and save only valid Turtle outputs."
SYSTEM_PROMPT_BY_TYPE = {
    "figure": DYNAMIC_PROMPT_DIR / "figure_system.md",
    "table": DYNAMIC_PROMPT_DIR / "table_system.md",
}


def build_prompt_package(
    image_path: Path,
    context: PromptBuilderContext | None = None,
    classifier: Callable[..., str] = classify_image_type,
) -> PromptPackage:
    effective_context = context or PromptBuilderContext(model=DEFAULT_MODEL, client=None)
    image_type = classifier(
        image_path,
        model=effective_context.model,
        client=effective_context.client,
    )
    if image_type not in SYSTEM_PROMPT_BY_TYPE:
        raise ValueError(f"Unsupported image type: {image_type}")

    return PromptPackage(
        system_prompt=load_text(SYSTEM_PROMPT_BY_TYPE[image_type]),
        examples=[ground_truth_example(image_type)],
        metadata={"predicted_image_type": image_type},
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
