from __future__ import annotations

from llm_as_judge.cli import run_dataset_cli


def main() -> int:
    return run_dataset_cli(dataset="soil_health", validate_with_vistext_metrics=False)


if __name__ == "__main__":
    raise SystemExit(main())
