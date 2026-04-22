from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from .datasets import collect_ttl_records, group_by_item, result_path, sample_records, strategy_dirs
from .judge_core import JudgeRunner
from .openai_provider import DEFAULT_OPENAI_JUDGE_MODEL, OpenAIJudgeProvider
from .validation import (
    compare_pairwise_to_metrics,
    load_traditional_metrics,
    select_validation_strategy_pair,
    summarize_direct_alignment,
    summarize_overall_alignment,
    summarize_pairwise_alignment,
    validate_direct_against_metrics,
)


DEFAULT_METRICS_PATHS = {
    "vistext": Path("vistext/evaluation/vistext_llm_evaluation_results.json"),
    "diagram2graph": Path("diagram2graph/evaluation/diagram2graph_llm_evaluation_results.json"),
}


def build_parser(*, dataset: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--modes", nargs="+", choices=["direct", "pairwise"], default=["direct", "pairwise"])
    parser.add_argument("--strategies", nargs="+", default=list(strategy_dirs(dataset).keys()))
    parser.add_argument("--strategy-selection", choices=["all", "widest_pair"], default="widest_pair")
    parser.add_argument(
        "--min-strategy-gap",
        type=float,
        default=0.02,
        help="Skip judge validation if the widest available strategy pair differs by less than this composite gap.",
    )
    parser.add_argument("--sample-mode", choices=["all", "random", "ids"], default="all")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-provider", choices=["openai"], default="openai")
    parser.add_argument("--judge-model", default=DEFAULT_OPENAI_JUDGE_MODEL)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of parallel judge API worker threads to use.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=result_path(dataset))
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATHS.get(dataset))
    return parser


def _pairwise_count(records: list[Any]) -> int:
    total = 0
    for item_records in group_by_item(records).values():
        total += sum(1 for _ in combinations(item_records, 2))
    return total


def _provider(args: argparse.Namespace) -> OpenAIJudgeProvider:
    if args.judge_provider == "openai":
        return OpenAIJudgeProvider(model=args.judge_model)
    raise ValueError(f"Unsupported judge provider: {args.judge_provider}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dataset_cli(
    *,
    dataset: str,
    argv: list[str] | None = None,
    validate_with_vistext_metrics: bool = False,
    validate_with_traditional_metrics: bool = False,
) -> int:
    parser = build_parser(dataset=dataset, description=f"Run LLM-as-a-judge evaluation for {dataset}.")
    args = parser.parse_args(argv)
    if args.parallel_workers <= 0:
        parser.error("--parallel-workers must be greater than 0")

    should_validate = validate_with_vistext_metrics or validate_with_traditional_metrics
    if should_validate and args.metrics_path is None:
        raise ValueError(f"No traditional metrics path is configured for dataset: {dataset}")
    if should_validate and not args.metrics_path.exists():
        raise FileNotFoundError(
            f"Traditional metrics report not found: {args.metrics_path}. "
            "Run the dataset's traditional evaluator first, or pass --metrics-path."
        )

    metrics = load_traditional_metrics(args.metrics_path) if should_validate else None
    effective_strategy_selection = args.strategy_selection if should_validate else "all"
    selection: dict[str, Any] | None = None
    if effective_strategy_selection == "widest_pair":
        if metrics is None:
            raise ValueError("--strategy-selection widest_pair requires a traditional metrics report.")
        selection = select_validation_strategy_pair(
            metrics,
            candidate_strategies=args.strategies,
            min_gap=args.min_strategy_gap,
        )
        if selection["status"] == "skipped":
            if args.dry_run:
                print(f"Dataset: {dataset}")
                print(f"Strategy selection: {args.strategy_selection}")
                print(f"Validation skipped: {selection['reason']}")
                if selection.get("best_pair"):
                    print(f"Best available pair: {', '.join(selection['best_pair']['strategies'])}")
                    print(f"Best pair composite gap: {selection['best_pair']['composite_gap']:.6f}")
                print(f"Traditional metrics path: {args.metrics_path}")
                return 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            validation = {
                "dataset": dataset,
                "generated_at_utc": _utc_now(),
                "strategy_selection": selection,
                "validation_summary": {
                    "alignment_strength": "skipped",
                    "alignment_conclusion": selection["reason"],
                },
            }
            _write_json(args.output_dir / f"{dataset}_llm_judge_validation.json", validation)
            print(f"Skipped judge validation for {dataset}: {selection['reason']}")
            return 0
        args.strategies = list(selection["best_pair"]["strategies"])

    ids = args.ids or None
    records = collect_ttl_records(dataset, strategies=args.strategies, ids=ids)
    records = sample_records(
        records,
        sample_mode=args.sample_mode,
        sample_count=args.sample_count,
        ids=ids,
        seed=args.seed,
    )
    direct_count = len(records)
    pairwise_count = _pairwise_count(records)

    if args.dry_run:
        print(f"Dataset: {dataset}")
        print(f"Strategies: {', '.join(args.strategies)}")
        print(f"Modes: {', '.join(args.modes)}")
        print(f"Direct items: {direct_count}")
        print(f"Pairwise comparisons: {pairwise_count}")
        print(f"Strategy selection: {effective_strategy_selection}")
        if selection is not None and selection.get("status") == "selected":
            print(f"Selected strategy pair: {', '.join(selection['best_pair']['strategies'])}")
            print(f"Selected pair composite gap: {selection['best_pair']['composite_gap']:.6f}")
        print(f"Parallel workers: {args.parallel_workers}")
        print(f"Output directory: {args.output_dir}")
        if should_validate:
            print(f"Traditional metrics path: {args.metrics_path}")
        return 0

    runner = JudgeRunner(
        provider=_provider(args),
        results_root=args.output_dir,
        parallel_workers=args.parallel_workers,
    )
    direct_report: dict[str, Any] | None = None
    pairwise_report: dict[str, Any] | None = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "direct" in args.modes:
        direct_report = runner.run_direct(
            records,
            output_path=args.output_dir / "direct_judge_results.json",
            skip_existing=args.skip_existing,
        )
    if "pairwise" in args.modes:
        pairwise_report = runner.run_pairwise(
            records,
            output_path=args.output_dir / "pairwise_judge_results.json",
            skip_existing=args.skip_existing,
        )

    if should_validate:
        assert metrics is not None
        validation: dict[str, Any] = {
            "dataset": dataset,
            "generated_at_utc": _utc_now(),
            "strategy_selection": selection
            or {
                "status": "selected",
                "best_pair": {"strategies": args.strategies},
            },
        }
        direct_key = "direct_vs_content_only_metrics" if dataset == "vistext" else "direct_vs_traditional_metrics"
        pairwise_key = "pairwise_vs_content_only_metrics" if dataset == "vistext" else "pairwise_vs_traditional_metrics"
        if direct_report is not None:
            validation[direct_key] = validate_direct_against_metrics(direct_report, metrics)
        if pairwise_report is not None:
            validation[pairwise_key] = compare_pairwise_to_metrics(pairwise_report, metrics)

        validation_summary: dict[str, Any] = {}
        if direct_key in validation:
            validation_summary["direct"] = summarize_direct_alignment(validation[direct_key])
        if pairwise_key in validation:
            validation_summary["pairwise"] = summarize_pairwise_alignment(validation[pairwise_key])
        validation_summary["overall"] = summarize_overall_alignment(validation_summary)
        validation["validation_summary"] = validation_summary
        _write_json(args.output_dir / f"{dataset}_llm_judge_validation.json", validation)

    print(f"Wrote judge results under: {args.output_dir}")
    return 0
