from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gemini_vistext_runner_core import (  # noqa: E402
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

from categorize_viz import classify_chart_type  # noqa: E402


OUTPUT_DIR = SCRIPT_DIR / "vistext_oneshot_dynamic_outputs"
DESCRIPTION = "Call Gemini on vistext chart images with dynamic one-shot prompting and save only valid Turtle outputs."
SYSTEM_PROMPT_BY_TYPE = {
    "bar": DYNAMIC_PROMPT_DIR / "bar_system.md",
    "line": DYNAMIC_PROMPT_DIR / "line_system.md",
    "area": DYNAMIC_PROMPT_DIR / "area_system.md",
}


def chart_type_from_label(image_path: Path) -> str:
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Cannot infer dry-run chart type without label: {label_path}")
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    properties = payload.get("L1_properties")
    if not isinstance(properties, list) or not properties:
        raise ValueError(f"Cannot infer chart type from L1_properties: {label_path}")
    chart_type = str(properties[0]).strip().lower()
    if chart_type not in SYSTEM_PROMPT_BY_TYPE:
        raise ValueError(f"Unsupported chart type in label {label_path}: {chart_type}")
    return chart_type


def build_prompt_package(
    image_path: Path,
    context: PromptBuilderContext | None = None,
    classifier: Callable[..., str] = classify_chart_type,
) -> PromptPackage:
    effective_context = context or PromptBuilderContext(model=DEFAULT_MODEL, client=None)
    if effective_context.dry_run:
        chart_type = chart_type_from_label(image_path)
    else:
        chart_type = classifier(
            image_path,
            model=effective_context.model,
            client=effective_context.client,
        )
    if chart_type not in SYSTEM_PROMPT_BY_TYPE:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    return PromptPackage(
        system_prompt=load_text(SYSTEM_PROMPT_BY_TYPE[chart_type]),
        examples=[ground_truth_example(chart_type)],
        metadata={"chart_type": chart_type},
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
