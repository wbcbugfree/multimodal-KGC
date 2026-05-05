from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from .datasets import GOLD_STRATEGY, collect_gold_records, collect_ttl_records, group_by_item, result_path, sample_records, strategy_dirs
from .judge_core import JudgeRunner, _include_pair
from .openai_batch import (
    DEFAULT_BATCH_COMPLETION_WINDOW,
    DEFAULT_BATCH_MAX_FILE_MB,
    OpenAIBatchJudgeRunner,
    build_batch_jobs,
)
from .openai_provider import DEFAULT_OPENAI_JUDGE_MODEL, OpenAIJudgeProvider
from .validation import (
    compare_pairwise_gold_preference,
    compare_pairwise_to_metrics,
    load_traditional_metrics,
    select_top_margin_items,
    select_validation_strategy_pair,
    summarize_direct_gold_preference,
    summarize_direct_alignment,
    summarize_pairwise_gold_preference,
    summarize_overall_alignment,
    summarize_pairwise_alignment,
    traditional_metric_snapshot,
    validate_direct_gold_preference,
    validate_direct_against_metrics,
)


DEFAULT_METRICS_PATHS = {
    "vistext": Path("vistext/evaluation/vistext_prompting_strategy_evaluation_results.json"),
    "diagram2graph": Path("diagram2graph/evaluation/diagram2graph_llm_evaluation_results.json"),
}


def build_parser(*, dataset: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--modes", nargs="+", choices=["direct", "pairwise"], default=["direct", "pairwise"])
    parser.add_argument("--strategies", nargs="+", default=list(strategy_dirs(dataset).keys()))
    parser.add_argument(
        "--validation-design",
        choices=["strategy_gap", "strategy_margin_top_n", "gold_vs_generated"],
        default="strategy_gap",
        help="Validation sampling design for labelled datasets.",
    )
    parser.add_argument("--strategy-selection", choices=["all", "widest_pair"], default="widest_pair")
    parser.add_argument(
        "--min-strategy-gap",
        type=float,
        default=0.02,
        help="Skip judge validation if the widest available strategy pair differs by less than this composite gap.",
    )
    parser.add_argument("--sample-mode", choices=["all", "random", "ids", "ascend"], default="all")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument(
        "--top-margin-count",
        type=int,
        default=100,
        help="Number of image IDs to keep for --validation-design strategy_margin_top_n.",
    )
    parser.add_argument(
        "--gold-sample-count",
        type=int,
        default=None,
        help="Optional item-ID limit for --validation-design gold_vs_generated after normal sampling.",
    )
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
    parser.add_argument(
        "--batch-action",
        choices=["submit", "status", "collect", "cancel"],
        default=None,
        help="Use OpenAI Batch API instead of immediate synchronous judge calls.",
    )
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        default=None,
        help="Path to the OpenAI Batch manifest. Defaults to <output-dir>/openai_batch_manifest.json.",
    )
    parser.add_argument(
        "--batch-completion-window",
        default=DEFAULT_BATCH_COMPLETION_WINDOW,
        help="OpenAI Batch completion window. OpenAI currently supports 24h.",
    )
    parser.add_argument(
        "--batch-max-file-mb",
        type=float,
        default=DEFAULT_BATCH_MAX_FILE_MB,
        help="Maximum JSONL size per uploaded batch input file. Large jobs are split into multiple batches.",
    )
    return parser


def _pairwise_count(records: list[Any], *, pairing_mode: str = "all") -> int:
    total = 0
    for item_records in group_by_item(records).values():
        total += sum(1 for record_a, record_b in combinations(item_records, 2) if _include_pair(record_a, record_b, pairing_mode))
    return total


def _provider(args: argparse.Namespace) -> OpenAIJudgeProvider:
    if args.judge_provider == "openai":
        return OpenAIJudgeProvider(model=args.judge_model)
    raise ValueError(f"Unsupported judge provider: {args.judge_provider}")


def _batch_manifest_path(args: argparse.Namespace) -> Path:
    return args.batch_manifest or (args.output_dir / "openai_batch_manifest.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _limit_item_ids(item_ids: list[str], *, count: int | None, seed: int) -> list[str]:
    if count is None or count >= len(item_ids):
        return item_ids
    if count <= 0:
        return []
    rng = random.Random(seed)
    selected = set(rng.sample(item_ids, count))
    return [item_id for item_id in item_ids if item_id in selected]


def _select_ascending_metric_records(
    records: list[Any],
    *,
    metrics: Any,
    count: int,
) -> tuple[list[Any], dict[str, Any]]:
    ranked: list[tuple[float, str, str, Any, dict[str, Any]]] = []
    missing_metric_records: list[dict[str, str]] = []
    for record in records:
        metric = metrics.per_image.get((record.strategy, record.item_id))
        snapshot = traditional_metric_snapshot(metric)
        quality_score = snapshot.get("quality_score") if isinstance(snapshot, dict) else None
        if not isinstance(quality_score, int | float):
            missing_metric_records.append(
                {
                    "strategy": record.strategy,
                    "item_id": record.item_id,
                }
            )
            continue
        ranked.append((float(quality_score), record.item_id, record.strategy, record, snapshot))

    selected = ranked[:0] if count <= 0 else sorted(ranked, key=lambda item: (item[0], item[1], item[2]))[:count]
    selected_records = [item[3] for item in selected]
    selection = {
        "status": "selected" if selected_records else "skipped",
        "selection_method": "gold_vs_generated_ascending_metric",
        "sample_mode": "ascend",
        "sample_count": count,
        "available_generated_record_count": len(ranked),
        "missing_metric_record_count": len(missing_metric_records),
        "missing_metric_records": missing_metric_records,
        "selected_item_ids": [item[1] for item in selected],
        "selected_generated_records": [
            {
                "strategy": strategy,
                "item_id": item_id,
                "generated_traditional_metrics": snapshot,
            }
            for quality_score, item_id, strategy, _record, snapshot in selected
        ],
    }
    if not selected_records:
        selection["reason"] = "No generated outputs had traditional metrics for ascending sampling."
    return selected_records, selection


def _build_validation_report(
    *,
    dataset: str,
    validation_design: str,
    selection: dict[str, Any] | None,
    strategies: list[str],
    direct_report: dict[str, Any] | None,
    pairwise_report: dict[str, Any] | None,
    metrics: Any,
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "dataset": dataset,
        "generated_at_utc": _utc_now(),
        "validation_design": validation_design,
        "strategy_selection": selection
        or {
            "status": "selected",
            "best_pair": {"strategies": strategies},
        },
    }
    validation_summary: dict[str, Any] = {}
    if validation_design == "gold_vs_generated":
        if direct_report is not None:
            validation["direct_gold_vs_generated"] = validate_direct_gold_preference(direct_report, metrics=metrics)
            validation_summary["direct"] = summarize_direct_gold_preference(validation["direct_gold_vs_generated"])
        if pairwise_report is not None:
            validation["pairwise_gold_vs_generated"] = compare_pairwise_gold_preference(pairwise_report, metrics=metrics)
            validation_summary["pairwise"] = summarize_pairwise_gold_preference(validation["pairwise_gold_vs_generated"])
    else:
        if metrics is None:
            raise ValueError("Traditional metrics are required for strategy-based judge validation.")
        direct_key = "direct_vs_content_only_metrics" if dataset == "vistext" else "direct_vs_traditional_metrics"
        pairwise_key = "pairwise_vs_content_only_metrics" if dataset == "vistext" else "pairwise_vs_traditional_metrics"
        if direct_report is not None:
            validation[direct_key] = validate_direct_against_metrics(direct_report, metrics)
            validation_summary["direct"] = summarize_direct_alignment(validation[direct_key])
        if pairwise_report is not None:
            validation[pairwise_key] = compare_pairwise_to_metrics(pairwise_report, metrics)
            validation_summary["pairwise"] = summarize_pairwise_alignment(validation[pairwise_key])
    validation_summary["overall"] = summarize_overall_alignment(validation_summary)
    validation["validation_summary"] = validation_summary
    return validation


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
    if args.batch_max_file_mb <= 0:
        parser.error("--batch-max-file-mb must be greater than 0")
    if args.top_margin_count <= 0:
        parser.error("--top-margin-count must be greater than 0")
    if args.gold_sample_count is not None and args.gold_sample_count <= 0:
        parser.error("--gold-sample-count must be greater than 0")
    if args.sample_mode == "ascend" and args.validation_design != "gold_vs_generated":
        parser.error("--sample-mode ascend is only supported with --validation-design gold_vs_generated")

    should_validate = validate_with_vistext_metrics or validate_with_traditional_metrics
    needs_metrics_for_ascending_sample = should_validate and args.validation_design == "gold_vs_generated" and args.sample_mode == "ascend"
    metrics_required = should_validate and (
        (
            args.validation_design == "strategy_margin_top_n"
            and args.batch_action in {None, "submit", "collect"}
        )
        or (
            args.validation_design == "strategy_gap"
            and (
                args.batch_action is None
                or args.batch_action == "collect"
                or (args.batch_action == "submit" and args.strategy_selection == "widest_pair")
            )
        )
        or needs_metrics_for_ascending_sample
    )
    metrics_optional = should_validate and args.validation_design == "gold_vs_generated" and (
        args.batch_action is None
        or args.batch_action == "collect"
    )
    if metrics_required and args.metrics_path is None:
        raise ValueError(f"No traditional metrics path is configured for dataset: {dataset}")
    if metrics_required and not args.metrics_path.exists():
        raise FileNotFoundError(
            f"Traditional metrics report not found: {args.metrics_path}. "
            "Run the dataset's traditional evaluator first, or pass --metrics-path."
        )

    metrics = (
        load_traditional_metrics(args.metrics_path)
        if (metrics_required or metrics_optional) and args.metrics_path is not None and args.metrics_path.exists()
        else None
    )
    effective_strategy_selection = (
        args.strategy_selection if should_validate and args.validation_design == "strategy_gap" and metrics is not None else "all"
    )
    selection: dict[str, Any] | None = None
    selected_top_margin_ids: list[str] | None = None
    if should_validate and args.validation_design == "strategy_margin_top_n":
        if metrics is None:
            raise ValueError("--validation-design strategy_margin_top_n requires a traditional metrics report.")
        selection = select_top_margin_items(
            metrics,
            candidate_strategies=args.strategies,
            top_n=args.top_margin_count,
        )
        if selection["status"] == "skipped" or not selection.get("selected_item_ids"):
            reason = selection.get("reason") or "No per-image metric gaps were available for the selected strategies."
            if args.dry_run:
                print(f"Dataset: {dataset}")
                print(f"Validation design: {args.validation_design}")
                print(f"Validation skipped: {reason}")
                return 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            validation = {
                "dataset": dataset,
                "generated_at_utc": _utc_now(),
                "validation_design": args.validation_design,
                "strategy_selection": selection,
                "validation_summary": {
                    "alignment_strength": "skipped",
                    "alignment_conclusion": reason,
                },
            }
            _write_json(args.output_dir / f"{dataset}_llm_judge_validation.json", validation)
            print(f"Skipped judge validation for {dataset}: {reason}")
            return 0
        args.strategies = list(selection["best_pair"]["strategies"])
        selected_top_margin_ids = list(selection["selected_item_ids"])
    elif effective_strategy_selection == "widest_pair":
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

    pairing_mode = "gold_vs_generated" if should_validate and args.validation_design == "gold_vs_generated" else "all"
    ids = selected_top_margin_ids or args.ids or None
    records = collect_ttl_records(dataset, strategies=args.strategies, ids=ids)
    if args.sample_mode == "ascend":
        if pairing_mode != "gold_vs_generated":
            raise ValueError("--sample-mode ascend is only supported for gold_vs_generated validation.")
        if metrics is None:
            raise ValueError("--sample-mode ascend requires a traditional metrics report.")
        records, selection = _select_ascending_metric_records(
            records,
            metrics=metrics,
            count=args.gold_sample_count or args.sample_count,
        )
        if selection.get("status") == "skipped":
            reason = selection.get("reason") or "No generated outputs were selected by ascending traditional metrics."
            if args.dry_run:
                print(f"Dataset: {dataset}")
                print(f"Validation design: {args.validation_design}")
                print(f"Validation skipped: {reason}")
                return 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            validation = {
                "dataset": dataset,
                "generated_at_utc": _utc_now(),
                "validation_design": args.validation_design,
                "strategy_selection": selection,
                "validation_summary": {
                    "alignment_strength": "skipped",
                    "alignment_conclusion": reason,
                },
            }
            _write_json(args.output_dir / f"{dataset}_llm_judge_validation.json", validation)
            print(f"Skipped judge validation for {dataset}: {reason}")
            return 0
    else:
        records = sample_records(
            records,
            sample_mode="ids" if selected_top_margin_ids else args.sample_mode,
            sample_count=args.sample_count,
            ids=ids,
            seed=args.seed,
        )
    if pairing_mode == "gold_vs_generated":
        generated_item_ids = sorted({record.item_id for record in records})
        if args.sample_mode != "ascend":
            generated_item_ids = _limit_item_ids(generated_item_ids, count=args.gold_sample_count, seed=args.seed)
        records = [record for record in records if record.item_id in set(generated_item_ids)]
        gold_records = collect_gold_records(dataset, ids=generated_item_ids, strategy_name=GOLD_STRATEGY)
        records.extend(gold_records)
    direct_count = len(records)
    pairwise_count = _pairwise_count(records, pairing_mode=pairing_mode)
    batch_jobs = (
        build_batch_jobs(
            records,
            modes=args.modes,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            pairing_mode=pairing_mode,
        )
        if args.batch_action in {"submit"} or (args.dry_run and args.batch_action == "submit")
        else []
    )

    if args.dry_run:
        print(f"Dataset: {dataset}")
        print(f"Strategies: {', '.join(args.strategies)}")
        print(f"Modes: {', '.join(args.modes)}")
        print(f"Validation design: {args.validation_design}")
        print(f"Direct items: {direct_count}")
        print(f"Pairwise comparisons: {pairwise_count}")
        print(f"Strategy selection: {effective_strategy_selection}")
        if selection is not None and selection.get("status") == "selected":
            if selection.get("best_pair"):
                print(f"Selected strategy pair: {', '.join(selection['best_pair']['strategies'])}")
                print(f"Selected pair composite gap: {selection['best_pair']['composite_gap']:.6f}")
            if selection.get("selection_method") == "strategy_margin_top_n":
                print(f"Top-margin selected IDs: {len(selection.get('selected_item_ids', []))}")
                print(f"Top-margin available IDs: {selection.get('available_item_count')}")
            if selection.get("selection_method") == "gold_vs_generated_ascending_metric":
                print(f"Ascending-metric selected generated outputs: {len(selection.get('selected_generated_records', []))}")
                print(f"Ascending-metric available generated outputs: {selection.get('available_generated_record_count')}")
        if pairing_mode == "gold_vs_generated":
            print("Pairing mode: gold_vs_generated")
            print(f"Ground-truth strategy alias: {GOLD_STRATEGY}")
        print(f"Parallel workers: {args.parallel_workers}")
        if args.batch_action:
            print(f"Batch action: {args.batch_action}")
            print(f"Batch manifest: {_batch_manifest_path(args)}")
            if args.batch_action == "submit":
                print(f"Batch requests: {len(batch_jobs)}")
                print(f"Batch completion window: {args.batch_completion_window}")
                print(f"Batch max JSONL file size: {args.batch_max_file_mb} MB")
        print(f"Output directory: {args.output_dir}")
        if should_validate:
            print(f"Traditional metrics path: {args.metrics_path}")
        return 0

    if args.batch_action:
        provider = _provider(args)
        batch_runner = OpenAIBatchJudgeRunner(
            provider=provider,
            output_dir=args.output_dir,
            manifest_path=_batch_manifest_path(args),
            direct_prompt=JudgeRunner(provider=provider, results_root=args.output_dir).direct_prompt,
            pairwise_prompt=JudgeRunner(provider=provider, results_root=args.output_dir).pairwise_prompt,
            completion_window=args.batch_completion_window,
            max_file_mb=args.batch_max_file_mb,
        )
        direct_report: dict[str, Any] | None = None
        pairwise_report: dict[str, Any] | None = None
        if args.batch_action == "submit":
            manifest = batch_runner.submit(
                dataset=dataset,
                records=records,
                modes=args.modes,
                strategy_selection=selection,
                validation_design=args.validation_design,
                skip_existing=args.skip_existing,
                pairing_mode=pairing_mode,
                dry_run=False,
            )
            batch_count = len(manifest.get("batches", []))
            print(f"Submitted {manifest.get('job_count', 0)} judge requests in {batch_count} OpenAI batch job(s).")
            print(f"Wrote batch manifest: {_batch_manifest_path(args)}")
            return 0
        if args.batch_action == "status":
            manifest = batch_runner.status()
            for batch_entry in manifest.get("batches", []):
                batch = batch_entry.get("batch") or {}
                print(
                    f"Batch {batch_entry.get('batch_index')}: "
                    f"{batch.get('id')} status={batch.get('status')} counts={batch.get('request_counts')}"
                )
            print(f"Updated batch manifest: {_batch_manifest_path(args)}")
            return 0
        if args.batch_action == "cancel":
            manifest = batch_runner.cancel()
            for batch_entry in manifest.get("batches", []):
                batch = batch_entry.get("batch") or {}
                print(f"Batch {batch_entry.get('batch_index')}: {batch.get('id')} status={batch.get('status')}")
            print(f"Updated batch manifest: {_batch_manifest_path(args)}")
            return 0
        if args.batch_action == "collect":
            direct_report, pairwise_report, _manifest = batch_runner.collect()
            print(f"Collected batch results from: {_batch_manifest_path(args)}")
        else:
            raise ValueError(f"Unsupported batch action: {args.batch_action}")

        if should_validate and direct_report is None and pairwise_report is None:
            print("No completed batch results were collected; validation summary was not updated.")
            print(f"Wrote judge results under: {args.output_dir}")
            return 0

        if should_validate:
            validation = _build_validation_report(
                dataset=dataset,
                validation_design=args.validation_design,
                selection=selection,
                strategies=args.strategies,
                direct_report=direct_report,
                pairwise_report=pairwise_report,
                metrics=metrics,
            )
            _write_json(args.output_dir / f"{dataset}_llm_judge_validation.json", validation)
        print(f"Wrote judge results under: {args.output_dir}")
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
            pairing_mode=pairing_mode,
        )

    if should_validate:
        validation = _build_validation_report(
            dataset=dataset,
            validation_design=args.validation_design,
            selection=selection,
            strategies=args.strategies,
            direct_report=direct_report,
            pairwise_report=pairwise_report,
            metrics=metrics,
        )
        _write_json(args.output_dir / f"{dataset}_llm_judge_validation.json", validation)

    print(f"Wrote judge results under: {args.output_dir}")
    return 0
