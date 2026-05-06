from __future__ import annotations

import argparse
import importlib.util
import json
import random
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from .datasets import (
    GOLD_STRATEGY,
    GOLD_TTL_DIRS,
    collect_gold_records,
    collect_ttl_records,
    group_by_item,
    repo_root,
    result_path,
    sample_records,
    strategy_dirs,
)
from .judge_core import JudgeRunner, _include_pair
from .openai_provider import DEFAULT_OPENAI_JUDGE_MODEL, OpenAIJudgeProvider
from .validation import (
    ContentOnlyMetrics,
    compare_pairwise_gold_preference,
    compare_pairwise_to_metrics,
    load_traditional_metrics,
    select_top_margin_items,
    select_validation_strategy_pair,
    summarize_direct_gold_preference,
    summarize_direct_alignment,
    summarize_pairwise_gold_preference,
    summarize_pairwise_alignment,
    traditional_metric_snapshot,
    validate_direct_gold_preference,
    validate_direct_against_metrics,
)


DEFAULT_METRICS_PATHS = {
    "vistext": Path("vistext/evaluation/vistext_prompting_strategy_evaluation_results.json"),
    "diagram2graph": Path("diagram2graph/evaluation/diagram2graph_llm_evaluation_results.json"),
}
_EVALUATOR_MODULES: dict[str, Any] = {}
_GRAPH_MATCHING_MODULES: dict[str, Any] = {}
_PER_IMAGE_F1_CACHE: dict[tuple[str, str, str, float | None], tuple[float, float, float]] = {}


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
        "--top-margin-threshold",
        type=float,
        default=None,
        help="Optional per-image gap threshold for --validation-design strategy_margin_top_n.",
    )
    parser.add_argument(
        "--top-margin-threshold-mode",
        choices=["any", "all"],
        default="any",
        help=(
            "Threshold filter for --top-margin-threshold. 'any' keeps images where either F1 or normalized GED gap "
            "exceeds the threshold; 'all' requires both gaps to exceed it."
        ),
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_evaluator_module(dataset: str) -> Any:
    if dataset in _EVALUATOR_MODULES:
        return _EVALUATOR_MODULES[dataset]
    module_paths = {
        "vistext": Path("vistext/evaluation/evaluate_vistext_llm_outputs.py"),
        "diagram2graph": Path("diagram2graph/evaluation/evaluate_diagram2graph_llm_outputs.py"),
    }
    module_path = module_paths.get(dataset)
    if module_path is None:
        raise ValueError(f"On-the-fly per-image F1 is not configured for dataset: {dataset}")
    absolute_path = repo_root() / module_path
    spec = importlib.util.spec_from_file_location(f"llm_as_judge_{dataset}_evaluator", absolute_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator module: {absolute_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EVALUATOR_MODULES[dataset] = module
    return module


def _graph_matching_module(dataset: str) -> Any:
    if dataset not in _GRAPH_MATCHING_MODULES:
        _GRAPH_MATCHING_MODULES[dataset] = _load_evaluator_module(dataset).load_graph_matching_module()
    return _GRAPH_MATCHING_MODULES[dataset]


def _vistext_numeric_tolerance(metrics: ContentOnlyMetrics) -> float:
    metadata = metrics.metadata if isinstance(metrics.metadata, dict) else {}
    metric_parameters = metadata.get("metric_parameters") if isinstance(metadata, dict) else None
    numeric_tolerance = metric_parameters.get("numeric_tolerance") if isinstance(metric_parameters, dict) else None
    relative_tolerance = numeric_tolerance.get("relative_tolerance") if isinstance(numeric_tolerance, dict) else None
    return float(relative_tolerance) if isinstance(relative_tolerance, int | float) else 0.01


def _triple_match_f1_from_metric(metric: Any) -> float | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get("triple_match_f1")
    if isinstance(value, int | float):
        return float(value)
    triple_match = metric.get("triple_match")
    if isinstance(triple_match, dict) and isinstance(triple_match.get("f1"), int | float):
        return float(triple_match["f1"])
    return None


def _compute_per_image_triple_match_prf(
    dataset: str,
    strategy: str,
    item_id: str,
    *,
    vistext_numeric_tolerance: float,
) -> tuple[float, float, float]:
    cache_key = (
        dataset,
        strategy,
        item_id,
        vistext_numeric_tolerance if dataset == "vistext" else None,
    )
    if cache_key in _PER_IMAGE_F1_CACHE:
        return _PER_IMAGE_F1_CACHE[cache_key]

    root = repo_root()
    gold_dir = GOLD_TTL_DIRS.get(dataset)
    if gold_dir is None:
        raise ValueError(f"No ground-truth TTL directory is configured for dataset: {dataset}")
    pred_dir = strategy_dirs(dataset).get(strategy)
    if pred_dir is None:
        raise ValueError(f"Unknown {dataset} strategy: {strategy}")

    gold_path = root / gold_dir / f"{item_id}.ttl"
    pred_path = root / pred_dir / f"{item_id}.ttl"
    if not gold_path.exists():
        raise FileNotFoundError(f"Ground-truth TTL not found: {gold_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predicted TTL not found: {pred_path}")

    evaluator = _load_evaluator_module(dataset)
    graph_metrics = _graph_matching_module(dataset)
    if dataset == "vistext":
        gold_graph = evaluator.ttl_to_webnlg_graph(gold_path, graph_mode="content_only")
        pred_graph = evaluator.ttl_to_webnlg_graph(pred_path, graph_mode="content_only")
        gold_graphs, pred_graphs, _metadata = evaluator.prepare_structural_graphs(
            [gold_graph],
            [pred_graph],
            vistext_numeric_tolerance,
            [item_id],
        )
    elif dataset == "diagram2graph":
        gold_graphs = [evaluator.ttl_to_webnlg_graph(gold_path)]
        pred_graphs = [evaluator.ttl_to_webnlg_graph(pred_path)]
    else:
        raise ValueError(f"On-the-fly per-image F1 is not configured for dataset: {dataset}")

    precision, recall, f1 = graph_metrics.get_triple_match_prf(gold_graphs, pred_graphs)
    result = (float(precision), float(recall), float(f1))
    _PER_IMAGE_F1_CACHE[cache_key] = result
    return result


def _fill_missing_top_margin_f1(
    *,
    dataset: str,
    metrics: ContentOnlyMetrics,
    candidate_strategies: list[str],
) -> ContentOnlyMetrics:
    pair_selection = select_validation_strategy_pair(
        metrics,
        candidate_strategies=candidate_strategies,
        min_gap=0.0,
    )
    if pair_selection.get("status") != "selected":
        return metrics
    first, second = pair_selection["best_pair"]["strategies"]
    common_ids = sorted(
        {
            item_id
            for strategy, item_id in metrics.per_image
            if strategy == first and (second, item_id) in metrics.per_image
        },
        key=lambda value: (int(value) if value.isdigit() else value),
    )
    missing_keys = [
        (strategy, item_id)
        for item_id in common_ids
        for strategy in (first, second)
        if _triple_match_f1_from_metric(metrics.per_image.get((strategy, item_id))) is None
    ]
    if not missing_keys:
        return metrics

    print(
        "Computing missing per-image triple-match F1 for "
        f"{len(missing_keys)} {dataset} strategy/item rows used by top-margin sampling."
    )
    vistext_tolerance = _vistext_numeric_tolerance(metrics)
    updated_per_image: dict[tuple[str, str], Any] = dict(metrics.per_image)
    for strategy, item_id in missing_keys:
        precision, recall, f1 = _compute_per_image_triple_match_prf(
            dataset,
            strategy,
            item_id,
            vistext_numeric_tolerance=vistext_tolerance,
        )
        metric = dict(updated_per_image[(strategy, item_id)])
        metric["triple_match"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        updated_per_image[(strategy, item_id)] = metric
    return ContentOnlyMetrics(
        per_image=updated_per_image,
        strategy_summary=metrics.strategy_summary,
        metadata=metrics.metadata,
    )


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
    if args.top_margin_count <= 0:
        parser.error("--top-margin-count must be greater than 0")
    if args.top_margin_threshold is not None and args.top_margin_threshold < 0:
        parser.error("--top-margin-threshold must be non-negative")
    if args.gold_sample_count is not None and args.gold_sample_count <= 0:
        parser.error("--gold-sample-count must be greater than 0")
    if args.sample_mode == "ascend" and args.validation_design != "gold_vs_generated":
        parser.error("--sample-mode ascend is only supported with --validation-design gold_vs_generated")

    should_validate = validate_with_vistext_metrics or validate_with_traditional_metrics
    needs_metrics_for_ascending_sample = should_validate and args.validation_design == "gold_vs_generated" and args.sample_mode == "ascend"
    metrics_required = should_validate and (
        args.validation_design in {"strategy_margin_top_n", "strategy_gap"}
        or needs_metrics_for_ascending_sample
    )
    metrics_optional = should_validate and args.validation_design == "gold_vs_generated"
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
        metrics = _fill_missing_top_margin_f1(
            dataset=dataset,
            metrics=metrics,
            candidate_strategies=args.strategies,
        )
        selection = select_top_margin_items(
            metrics,
            candidate_strategies=args.strategies,
            top_n=args.top_margin_count,
            gap_threshold=args.top_margin_threshold,
            gap_threshold_mode=args.top_margin_threshold_mode,
        )
        gap_availability = selection.get("candidate_gap_component_availability") or {}
        f1_items = gap_availability.get("triple_match_f1_gap_items")
        total_items = gap_availability.get("items")
        if isinstance(f1_items, int) and isinstance(total_items, int) and f1_items < total_items:
            raise ValueError(
                "Per-image triple-match F1 is required for --validation-design strategy_margin_top_n, but the "
                f"traditional metrics report only has F1 gaps for {f1_items}/{total_items} candidate items. "
                "Regenerate the traditional metrics report with the current evaluator, or pass --metrics-path "
                "to a report that includes per-image triple_match.f1."
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
                    "status": "skipped",
                    "reason": reason,
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
                    "status": "skipped",
                    "reason": selection["reason"],
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
                    "status": "skipped",
                    "reason": reason,
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
                print(f"Top-margin candidate IDs: {selection.get('candidate_item_count')}")
                print(f"Top-margin available IDs: {selection.get('available_item_count')}")
                if selection.get("gap_threshold") is not None:
                    print(
                        "Top-margin threshold: "
                        f"{selection.get('gap_threshold')} ({selection.get('gap_threshold_mode')})"
                    )
                selected_gap_summary = selection.get("selected_per_image_gap_summary") or {}
                available_gap_summary = selection.get("available_per_image_gap_summary") or {}
                candidate_gap_availability = selection.get("candidate_gap_component_availability") or {}
                if candidate_gap_availability:
                    print(
                        "Top-margin gap availability: "
                        f"F1={candidate_gap_availability.get('triple_match_f1_gap_items')}/"
                        f"{candidate_gap_availability.get('items')}, "
                        f"GED={candidate_gap_availability.get('normalized_ged_gap_items')}/"
                        f"{candidate_gap_availability.get('items')}"
                    )
                if isinstance(selected_gap_summary.get("mean"), int | float):
                    print(f"Selected top-margin mean per-image gap: {selected_gap_summary['mean']:.6f}")
                if isinstance(available_gap_summary.get("mean"), int | float):
                    print(f"Available top-margin mean per-image gap: {available_gap_summary['mean']:.6f}")
            if selection.get("selection_method") == "gold_vs_generated_ascending_metric":
                print(f"Ascending-metric selected generated outputs: {len(selection.get('selected_generated_records', []))}")
                print(f"Ascending-metric available generated outputs: {selection.get('available_generated_record_count')}")
        if pairing_mode == "gold_vs_generated":
            print("Pairing mode: gold_vs_generated")
            print(f"Ground-truth strategy alias: {GOLD_STRATEGY}")
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
