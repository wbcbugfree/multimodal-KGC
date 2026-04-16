from __future__ import annotations

from llm_as_judge.cli import run_dataset_cli


def main() -> int:
    return run_dataset_cli(dataset="diagram2graph", validate_with_traditional_metrics=True)


if __name__ == "__main__":
    raise SystemExit(main())
