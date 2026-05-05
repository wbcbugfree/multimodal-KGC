from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from .datasets import GOLD_STRATEGY


@dataclass(frozen=True)
class ContentOnlyMetrics:
    per_image: dict[tuple[str, str], Mapping[str, Any]]
    strategy_summary: dict[str, Mapping[str, Any]]
    metadata: Mapping[str, Any] | None = None


def _as_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _triple_match_f1(metric: Mapping[str, Any] | None) -> float | None:
    if not isinstance(metric, Mapping):
        return None
    value = _as_float(metric.get("triple_match_f1"))
    if value is not None:
        return value
    triple_match = metric.get("triple_match")
    if isinstance(triple_match, Mapping):
        return _as_float(triple_match.get("f1"))
    return None


def traditional_metric_snapshot(metric: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metric, Mapping):
        return None

    triple_match_f1 = _triple_match_f1(metric)
    triple_match_accuracy = _as_float(metric.get("triple_match_accuracy"))
    normalized_ged = _as_float(metric.get("normalized_ged"))
    inverse_normalized_ged = 1.0 - normalized_ged if normalized_ged is not None else None

    quality_components: list[float] = []
    quality_basis: list[str] = []
    if triple_match_f1 is not None:
        quality_components.append(triple_match_f1)
        quality_basis.append("triple_match_f1")
    elif triple_match_accuracy is not None:
        quality_components.append(triple_match_accuracy)
        quality_basis.append("triple_match_accuracy")
    if inverse_normalized_ged is not None:
        quality_components.append(inverse_normalized_ged)
        quality_basis.append("inverse_normalized_ged")

    return {
        "triple_match_f1": triple_match_f1,
        "triple_match_accuracy": triple_match_accuracy,
        "normalized_ged": normalized_ged,
        "inverse_normalized_ged": inverse_normalized_ged,
        "quality_score": _mean(quality_components),
        "quality_score_basis": quality_basis,
    }


def _summarize_generated_metrics_by_outcome(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome_key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        outcome = row.get(outcome_key)
        metrics = row.get("generated_traditional_metrics")
        if isinstance(outcome, str) and isinstance(metrics, Mapping):
            grouped.setdefault(outcome, []).append(metrics)

    summaries: dict[str, Any] = {}
    metric_keys = [
        "triple_match_f1",
        "triple_match_accuracy",
        "normalized_ged",
        "inverse_normalized_ged",
        "quality_score",
    ]
    for outcome, metric_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {"count": len(metric_rows)}
        for metric_key in metric_keys:
            values = [float(item[metric_key]) for item in metric_rows if isinstance(item.get(metric_key), int | float)]
            summary[f"{metric_key}_mean"] = _mean(values)
        summaries[outcome] = summary
    return summaries


def load_traditional_metrics(path: Path) -> ContentOnlyMetrics:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "graph_modes" in data:
        strategies = data["graph_modes"]["content_only"]["strategies"]
    else:
        strategies = data["strategies"]
    per_image: dict[tuple[str, str], Mapping[str, Any]] = {}
    summary: dict[str, Mapping[str, Any]] = {}
    for strategy, strategy_data in strategies.items():
        raw_summary = strategy_data.get("summary", {})
        triple_match = raw_summary.get("triple_match_micro", {})
        summary[strategy] = {
            "triple_match_micro_f1": triple_match.get("f1"),
            "normalized_ged_mean": raw_summary.get("normalized_ged_mean"),
            "triple_match_accuracy_mean": raw_summary.get("triple_match_accuracy_mean"),
        }
        for item in strategy_data.get("per_image", []):
            item_id = str(item.get("img_id") or item.get("item_id"))
            per_image[(strategy, item_id)] = item
    return ContentOnlyMetrics(per_image=per_image, strategy_summary=summary, metadata=data)


def load_content_only_metrics(path: Path) -> ContentOnlyMetrics:
    return load_traditional_metrics(path)


def select_validation_strategy_pair(
    metrics: ContentOnlyMetrics,
    *,
    candidate_strategies: Sequence[str] | None = None,
    min_gap: float = 0.02,
) -> dict[str, Any]:
    strategies = sorted(set(candidate_strategies or metrics.strategy_summary.keys()))
    available = [strategy for strategy in strategies if strategy in metrics.strategy_summary]
    if len(available) < 2:
        return {
            "status": "skipped",
            "reason": "Fewer than two strategies are available for validation.",
            "candidate_strategies": strategies,
            "available_strategies": available,
            "min_gap": min_gap,
            "gap_metric_components": ["triple_match_f1", "normalized_ged"],
        }

    best_pair: dict[str, Any] | None = None
    for first, second in combinations(available, 2):
        first_summary = metrics.strategy_summary[first]
        second_summary = metrics.strategy_summary[second]
        triple_f1_gap = abs(
            (_as_float(first_summary.get("triple_match_micro_f1")) or 0.0)
            - (_as_float(second_summary.get("triple_match_micro_f1")) or 0.0)
        )
        normalized_ged_gap = abs(
            (_as_float(first_summary.get("normalized_ged_mean")) or 0.0)
            - (_as_float(second_summary.get("normalized_ged_mean")) or 0.0)
        )
        pair_summary = {
            "strategies": [first, second],
            "triple_match_micro_f1_gap": triple_f1_gap,
            "normalized_ged_gap": normalized_ged_gap,
            "composite_gap": _mean([triple_f1_gap, normalized_ged_gap]) or 0.0,
        }
        if best_pair is None or (
            pair_summary["composite_gap"],
            pair_summary["triple_match_micro_f1_gap"],
            pair_summary["normalized_ged_gap"],
        ) > (
            best_pair["composite_gap"],
            best_pair["triple_match_micro_f1_gap"],
            best_pair["normalized_ged_gap"],
        ):
            best_pair = pair_summary

    assert best_pair is not None
    if best_pair["composite_gap"] < min_gap:
        return {
            "status": "skipped",
            "reason": "The best-vs-worst strategy gap is too small for meaningful judge validation.",
            "candidate_strategies": strategies,
            "available_strategies": available,
            "min_gap": min_gap,
            "gap_metric_components": ["triple_match_f1", "normalized_ged"],
            "best_pair": best_pair,
        }

    return {
        "status": "selected",
        "candidate_strategies": strategies,
        "available_strategies": available,
        "min_gap": min_gap,
        "gap_metric_components": ["triple_match_f1", "normalized_ged"],
        "best_pair": best_pair,
    }


def _per_image_metric_gap_components(metric_a: Mapping[str, Any], metric_b: Mapping[str, Any]) -> dict[str, float | None]:
    values: list[float] = []
    score_a = _triple_match_f1(metric_a)
    score_b = _triple_match_f1(metric_b)
    triple_match_f1_gap = None
    if isinstance(score_a, int | float) and isinstance(score_b, int | float):
        triple_match_f1_gap = abs(float(score_a) - float(score_b))
        values.append(triple_match_f1_gap)
    ged_a = metric_a.get("normalized_ged")
    ged_b = metric_b.get("normalized_ged")
    normalized_ged_gap = None
    if isinstance(ged_a, int | float) and isinstance(ged_b, int | float):
        normalized_ged_gap = abs(float(ged_a) - float(ged_b))
        values.append(normalized_ged_gap)
    return {
        "triple_match_f1_gap": triple_match_f1_gap,
        "normalized_ged_gap": normalized_ged_gap,
        "per_image_gap": _mean(values),
    }


def _passes_gap_threshold(
    gap_components: Mapping[str, float | None],
    *,
    gap_threshold: float | None,
    gap_threshold_mode: str,
) -> bool:
    if gap_threshold is None:
        return True
    component_values = [
        gap_components.get("triple_match_f1_gap"),
        gap_components.get("normalized_ged_gap"),
    ]
    comparisons = [float(value) > gap_threshold for value in component_values if isinstance(value, int | float)]
    if gap_threshold_mode == "any":
        return any(comparisons)
    if gap_threshold_mode == "all":
        return len(comparisons) == len(component_values) and all(comparisons)
    raise ValueError(f"Unsupported gap threshold mode: {gap_threshold_mode}")


def _gap_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gaps = [float(row["per_image_gap"]) for row in rows if isinstance(row.get("per_image_gap"), int | float)]
    return {
        "count": len(gaps),
        "mean": _mean(gaps),
        "min": min(gaps) if gaps else None,
        "max": max(gaps) if gaps else None,
    }


def _gap_component_availability(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "items": len(rows),
        "triple_match_f1_gap_items": sum(1 for row in rows if isinstance(row.get("triple_match_f1_gap"), int | float)),
        "normalized_ged_gap_items": sum(1 for row in rows if isinstance(row.get("normalized_ged_gap"), int | float)),
    }


def select_top_margin_items(
    metrics: ContentOnlyMetrics,
    *,
    candidate_strategies: Sequence[str] | None = None,
    top_n: int = 100,
    gap_threshold: float | None = None,
    gap_threshold_mode: str = "any",
) -> dict[str, Any]:
    pair_selection = select_validation_strategy_pair(
        metrics,
        candidate_strategies=candidate_strategies,
        min_gap=0.0,
    )
    if pair_selection["status"] != "selected":
        return pair_selection
    first, second = pair_selection["best_pair"]["strategies"]
    common_ids = sorted(
        {
            item_id
            for strategy, item_id in metrics.per_image
            if strategy == first and (second, item_id) in metrics.per_image
        }
    )
    candidate_rows: list[dict[str, Any]] = []
    for item_id in common_ids:
        metric_a = metrics.per_image[(first, item_id)]
        metric_b = metrics.per_image[(second, item_id)]
        gap_components = _per_image_metric_gap_components(metric_a, metric_b)
        gap = gap_components.get("per_image_gap")
        if gap is None:
            continue
        candidate_rows.append(
            {
                "item_id": item_id,
                "strategy_a": first,
                "strategy_b": second,
                "per_image_gap": gap,
                "triple_match_f1_gap": gap_components.get("triple_match_f1_gap"),
                "normalized_ged_gap": gap_components.get("normalized_ged_gap"),
                "strategy_a_triple_match_f1": _triple_match_f1(metric_a),
                "strategy_b_triple_match_f1": _triple_match_f1(metric_b),
                "strategy_a_triple_match_accuracy": metric_a.get("triple_match_accuracy"),
                "strategy_b_triple_match_accuracy": metric_b.get("triple_match_accuracy"),
                "strategy_a_normalized_ged": metric_a.get("normalized_ged"),
                "strategy_b_normalized_ged": metric_b.get("normalized_ged"),
            }
        )
    rows = [
        row
        for row in candidate_rows
        if _passes_gap_threshold(row, gap_threshold=gap_threshold, gap_threshold_mode=gap_threshold_mode)
    ]
    rows = sorted(rows, key=lambda row: (-float(row["per_image_gap"]), str(row["item_id"])))
    selected_rows = rows[: max(top_n, 0)]
    return {
        **pair_selection,
        "selection_method": "strategy_margin_top_n",
        "top_n": top_n,
        "gap_threshold": gap_threshold,
        "gap_threshold_mode": gap_threshold_mode,
        "candidate_item_count": len(candidate_rows),
        "candidate_gap_component_availability": _gap_component_availability(candidate_rows),
        "candidate_per_image_gap_summary": _gap_summary(candidate_rows),
        "available_item_count": len(rows),
        "available_gap_component_availability": _gap_component_availability(rows),
        "available_per_image_gap_summary": _gap_summary(rows),
        "selected_item_ids": [str(row["item_id"]) for row in selected_rows],
        "selected_per_image_gap_summary": _gap_summary(selected_rows),
        "selected_items": selected_rows,
    }


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    try:
        from scipy.stats import spearmanr

        value = float(spearmanr(xs, ys).statistic)
    except Exception:
        return None
    if math.isnan(value):
        return None
    return value


def _score(record: Mapping[str, Any], key: str) -> float | None:
    scores = record.get("scores", {})
    if not isinstance(scores, Mapping):
        return None
    value = scores.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _classify_direct_alignment(value: float | None) -> str:
    if value is None:
        return "inconclusive"
    if value >= 0.7:
        return "strong"
    if value >= 0.4:
        return "moderate"
    if value >= 0.2:
        return "weak"
    return "poor"


def _classify_pairwise_alignment(value: float | None) -> str:
    if value is None:
        return "inconclusive"
    if value >= 0.8:
        return "strong"
    if value >= 0.6:
        return "moderate"
    if value >= 0.5:
        return "weak"
    return "poor"


def summarize_direct_alignment(validation: Mapping[str, Any]) -> dict[str, Any]:
    spearman = validation.get("spearman", {})
    if not isinstance(spearman, Mapping):
        spearman = {}
    values = [float(value) for value in spearman.values() if isinstance(value, int | float)]
    mean_value = _mean(values)
    strength = _classify_direct_alignment(mean_value)
    if strength == "strong":
        conclusion = "Direct-judge scores align strongly with the traditional metrics."
    elif strength == "moderate":
        conclusion = "Direct-judge scores align moderately with the traditional metrics."
    elif strength == "weak":
        conclusion = "Direct-judge scores show only weak alignment with the traditional metrics."
    elif strength == "poor":
        conclusion = "Direct-judge scores do not align well with the traditional metrics."
    else:
        conclusion = "Direct-judge alignment is inconclusive because the available statistics are insufficient."
    return {
        "mode": "direct",
        "matched_items": validation.get("matched_items"),
        "mean_spearman": mean_value,
        "alignment_strength": strength,
        "alignment_conclusion": conclusion,
    }


def summarize_pairwise_alignment(validation: Mapping[str, Any]) -> dict[str, Any]:
    agreement_rate = validation.get("agreement_rate")
    agreement_value = float(agreement_rate) if isinstance(agreement_rate, int | float) else None
    strength = _classify_pairwise_alignment(agreement_value)
    if strength == "strong":
        conclusion = "Pairwise judge preferences align strongly with the traditional metrics."
    elif strength == "moderate":
        conclusion = "Pairwise judge preferences align moderately with the traditional metrics."
    elif strength == "weak":
        conclusion = "Pairwise judge preferences show only weak alignment with the traditional metrics."
    elif strength == "poor":
        conclusion = "Pairwise judge preferences do not align well with the traditional metrics."
    else:
        conclusion = "Pairwise alignment is inconclusive because there are no comparable items."
    return {
        "mode": "pairwise",
        "comparable_items": validation.get("comparable_items"),
        "agreement_rate": agreement_value,
        "alignment_strength": strength,
        "alignment_conclusion": conclusion,
    }


def summarize_direct_gold_preference(validation: Mapping[str, Any]) -> dict[str, Any]:
    rate = validation.get("overall_gold_not_lower_rate")
    rate_value = float(rate) if isinstance(rate, int | float) else None
    strength = _classify_pairwise_alignment(rate_value)
    if strength == "strong":
        conclusion = "Direct-judge scores usually assign the ground-truth RDF graph a score at least as high as the generated RDF graph."
    elif strength == "moderate":
        conclusion = "Direct-judge scores moderately prefer the ground-truth RDF graph over the generated RDF graph."
    elif strength == "weak":
        conclusion = "Direct-judge scores show only weak preference for the ground-truth RDF graph."
    elif strength == "poor":
        conclusion = "Direct-judge scores do not reliably prefer the ground-truth RDF graph."
    else:
        conclusion = "Direct gold-vs-generated validation is inconclusive because no comparable items are available."
    return {
        "mode": "direct",
        "comparison_type": "gold_vs_generated",
        "comparable_items": validation.get("comparable_items"),
        "gold_not_lower_rate": rate_value,
        "alignment_strength": strength,
        "alignment_conclusion": conclusion,
    }


def summarize_pairwise_gold_preference(validation: Mapping[str, Any]) -> dict[str, Any]:
    rate = validation.get("gold_win_or_tie_rate")
    rate_value = float(rate) if isinstance(rate, int | float) else None
    strength = _classify_pairwise_alignment(rate_value)
    if strength == "strong":
        conclusion = "Pairwise judge preferences usually select the ground-truth RDF graph or mark it as tied."
    elif strength == "moderate":
        conclusion = "Pairwise judge preferences moderately favor the ground-truth RDF graph."
    elif strength == "weak":
        conclusion = "Pairwise judge preferences show only weak preference for the ground-truth RDF graph."
    elif strength == "poor":
        conclusion = "Pairwise judge preferences do not reliably favor the ground-truth RDF graph."
    else:
        conclusion = "Pairwise gold-vs-generated validation is inconclusive because no comparable items are available."
    return {
        "mode": "pairwise",
        "comparison_type": "gold_vs_generated",
        "comparable_items": validation.get("comparable_items"),
        "gold_win_or_tie_rate": rate_value,
        "alignment_strength": strength,
        "alignment_conclusion": conclusion,
    }


def summarize_overall_alignment(mode_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    strength_rank = {"poor": 0, "weak": 1, "moderate": 2, "strong": 3}
    strengths = [
        str(summary.get("alignment_strength"))
        for summary in mode_summaries.values()
        if isinstance(summary, Mapping) and str(summary.get("alignment_strength")) in strength_rank
    ]
    if not strengths:
        return {
            "alignment_strength": "inconclusive",
            "alignment_conclusion": "Overall judge alignment is inconclusive because no comparable validation statistics are available.",
        }

    min_strength = min(strengths, key=lambda value: strength_rank[value])
    if all(strength_rank[value] >= strength_rank["moderate"] for value in strengths):
        conclusion = "The LLM judge aligns well with the traditional metrics on the selected validation setting."
    elif any(strength_rank[value] >= strength_rank["moderate"] for value in strengths):
        conclusion = "The LLM judge shows partial alignment with the traditional metrics, but the evidence is mixed across validation modes."
    else:
        conclusion = "The LLM judge does not align well with the traditional metrics on the selected validation setting."
    return {
        "alignment_strength": min_strength,
        "alignment_conclusion": conclusion,
    }


def validate_direct_against_metrics(direct_report: Mapping[str, Any], metrics: ContentOnlyMetrics) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in direct_report.get("items", []):
        if not isinstance(item, Mapping) or item.get("status", "success") != "success":
            continue
        strategy = str(item.get("strategy"))
        item_id = str(item.get("item_id"))
        metric = metrics.per_image.get((strategy, item_id))
        if metric is None:
            continue
        metric_snapshot = traditional_metric_snapshot(metric) or {}
        rows.append(
            {
                "strategy": strategy,
                "item_id": item_id,
                "overall_score": _score(item, "overall_score"),
                "criteria_mean": _score(item, "criteria_mean"),
                "triple_match_f1": metric_snapshot.get("triple_match_f1"),
                "triple_match_accuracy": metric_snapshot.get("triple_match_accuracy"),
                "inverse_normalized_ged": metric_snapshot.get("inverse_normalized_ged"),
            }
        )

    overall_f1 = [row["overall_score"] for row in rows if row["overall_score"] is not None and row["triple_match_f1"] is not None]
    f1_for_overall = [row["triple_match_f1"] for row in rows if row["overall_score"] is not None and row["triple_match_f1"] is not None]
    criteria_f1 = [row["criteria_mean"] for row in rows if row["criteria_mean"] is not None and row["triple_match_f1"] is not None]
    f1_for_criteria = [
        row["triple_match_f1"] for row in rows if row["criteria_mean"] is not None and row["triple_match_f1"] is not None
    ]
    ged_overall = [row["overall_score"] for row in rows if row["overall_score"] is not None and row["inverse_normalized_ged"] is not None]
    inverse_ged = [
        row["inverse_normalized_ged"] for row in rows if row["overall_score"] is not None and row["inverse_normalized_ged"] is not None
    ]
    ged_criteria = [row["criteria_mean"] for row in rows if row["criteria_mean"] is not None and row["inverse_normalized_ged"] is not None]
    inverse_ged_for_criteria = [
        row["inverse_normalized_ged"] for row in rows if row["criteria_mean"] is not None and row["inverse_normalized_ged"] is not None
    ]

    return {
        "matched_items": len(rows),
        "metric_availability": {
            "triple_match_f1_rows": sum(1 for row in rows if row.get("triple_match_f1") is not None),
            "inverse_normalized_ged_rows": sum(1 for row in rows if row.get("inverse_normalized_ged") is not None),
        },
        "spearman": {
            "overall_vs_triple_match_f1": _spearman(overall_f1, f1_for_overall),
            "criteria_mean_vs_triple_match_f1": _spearman(criteria_f1, f1_for_criteria),
            "overall_vs_inverse_normalized_ged": _spearman(ged_overall, inverse_ged),
            "criteria_mean_vs_inverse_normalized_ged": _spearman(ged_criteria, inverse_ged_for_criteria),
        },
        "rows": rows,
    }


def _metric_winner(metric_a: Mapping[str, Any], metric_b: Mapping[str, Any], *, tolerance: float = 1e-9) -> str:
    score_a = metric_a.get("triple_match_accuracy")
    score_b = metric_b.get("triple_match_accuracy")
    if isinstance(score_a, int | float) and isinstance(score_b, int | float):
        if abs(float(score_a) - float(score_b)) <= tolerance:
            return "tie"
        return "A" if float(score_a) > float(score_b) else "B"
    ged_a = metric_a.get("normalized_ged")
    ged_b = metric_b.get("normalized_ged")
    if isinstance(ged_a, int | float) and isinstance(ged_b, int | float):
        if abs(float(ged_a) - float(ged_b)) <= tolerance:
            return "tie"
        return "A" if float(ged_a) < float(ged_b) else "B"
    return "unknown"


def compare_pairwise_to_metrics(pairwise_report: Mapping[str, Any], metrics: ContentOnlyMetrics) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    agreements = 0
    for item in pairwise_report.get("items", []):
        if not isinstance(item, Mapping) or item.get("status", "success") != "success":
            continue
        item_id = str(item.get("item_id"))
        strategy_a = str(item.get("strategy_a"))
        strategy_b = str(item.get("strategy_b"))
        metric_a = metrics.per_image.get((strategy_a, item_id))
        metric_b = metrics.per_image.get((strategy_b, item_id))
        if metric_a is None or metric_b is None:
            continue
        metric_winner = _metric_winner(metric_a, metric_b)
        if metric_winner == "unknown":
            continue
        judge = item.get("judge", {})
        judge_winner = judge.get("winner") if isinstance(judge, Mapping) else None
        agreed = judge_winner == metric_winner
        agreements += 1 if agreed else 0
        rows.append(
            {
                "item_id": item_id,
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "judge_winner": judge_winner,
                "metric_winner": metric_winner,
                "agreed": agreed,
            }
        )
    return {
        "comparable_items": len(rows),
        "agreement_rate": agreements / len(rows) if rows else None,
        "rows": rows,
    }


def validate_direct_gold_preference(
    direct_report: Mapping[str, Any],
    *,
    metrics: ContentOnlyMetrics | None = None,
    gold_strategy: str = GOLD_STRATEGY,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in direct_report.get("items", []):
        if isinstance(item, Mapping) and item.get("status", "success") == "success":
            grouped.setdefault(str(item.get("item_id")), []).append(item)

    rows: list[dict[str, Any]] = []
    overall_gold_higher = 0
    overall_gold_not_lower = 0
    criteria_gold_higher = 0
    criteria_gold_not_lower = 0
    for item_id, items in sorted(grouped.items()):
        gold_items = [item for item in items if item.get("strategy") == gold_strategy]
        generated_items = [item for item in items if item.get("strategy") != gold_strategy]
        if not gold_items:
            continue
        gold_item = gold_items[0]
        gold_overall = _score(gold_item, "overall_score")
        gold_criteria = _score(gold_item, "criteria_mean")
        for generated in generated_items:
            generated_strategy = str(generated.get("strategy"))
            generated_overall = _score(generated, "overall_score")
            generated_criteria = _score(generated, "criteria_mean")
            row = {
                "item_id": item_id,
                "generated_strategy": generated_strategy,
                "gold_overall_score": gold_overall,
                "generated_overall_score": generated_overall,
                "gold_criteria_mean": gold_criteria,
                "generated_criteria_mean": generated_criteria,
            }
            if metrics is not None:
                metric_snapshot = traditional_metric_snapshot(metrics.per_image.get((generated_strategy, item_id)))
                if metric_snapshot is not None:
                    row["generated_traditional_metrics"] = metric_snapshot
            if gold_overall is not None and generated_overall is not None:
                row["gold_overall_higher"] = gold_overall > generated_overall
                row["gold_overall_not_lower"] = gold_overall >= generated_overall
                if gold_overall > generated_overall:
                    row["overall_outcome"] = "gold_higher"
                elif gold_overall == generated_overall:
                    row["overall_outcome"] = "tie"
                else:
                    row["overall_outcome"] = "generated_higher"
                overall_gold_higher += 1 if gold_overall > generated_overall else 0
                overall_gold_not_lower += 1 if gold_overall >= generated_overall else 0
            if gold_criteria is not None and generated_criteria is not None:
                row["gold_criteria_higher"] = gold_criteria > generated_criteria
                row["gold_criteria_not_lower"] = gold_criteria >= generated_criteria
                if gold_criteria > generated_criteria:
                    row["criteria_outcome"] = "gold_higher"
                elif gold_criteria == generated_criteria:
                    row["criteria_outcome"] = "tie"
                else:
                    row["criteria_outcome"] = "generated_higher"
                criteria_gold_higher += 1 if gold_criteria > generated_criteria else 0
                criteria_gold_not_lower += 1 if gold_criteria >= generated_criteria else 0
            rows.append(row)

    overall_comparable = sum(1 for row in rows if "gold_overall_higher" in row)
    criteria_comparable = sum(1 for row in rows if "gold_criteria_higher" in row)
    return {
        "comparison_type": "gold_vs_generated",
        "gold_strategy": gold_strategy,
        "comparable_items": len(rows),
        "overall_gold_higher_rate": overall_gold_higher / overall_comparable if overall_comparable else None,
        "overall_gold_not_lower_rate": overall_gold_not_lower / overall_comparable if overall_comparable else None,
        "criteria_gold_higher_rate": criteria_gold_higher / criteria_comparable if criteria_comparable else None,
        "criteria_gold_not_lower_rate": criteria_gold_not_lower / criteria_comparable if criteria_comparable else None,
        "generated_metric_summary_by_overall_outcome": _summarize_generated_metrics_by_outcome(
            rows,
            outcome_key="overall_outcome",
        ),
        "generated_metric_summary_by_criteria_outcome": _summarize_generated_metrics_by_outcome(
            rows,
            outcome_key="criteria_outcome",
        ),
        "rows": rows,
    }


def compare_pairwise_gold_preference(
    pairwise_report: Mapping[str, Any],
    *,
    metrics: ContentOnlyMetrics | None = None,
    gold_strategy: str = GOLD_STRATEGY,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    gold_wins = 0
    gold_wins_or_ties = 0
    for item in pairwise_report.get("items", []):
        if not isinstance(item, Mapping) or item.get("status", "success") != "success":
            continue
        strategy_a = str(item.get("strategy_a"))
        strategy_b = str(item.get("strategy_b"))
        if (strategy_a == gold_strategy) == (strategy_b == gold_strategy):
            continue
        gold_side = "A" if strategy_a == gold_strategy else "B"
        generated_strategy = strategy_b if strategy_a == gold_strategy else strategy_a
        item_id = str(item.get("item_id"))
        judge = item.get("judge", {})
        judge_winner = judge.get("winner") if isinstance(judge, Mapping) else None
        gold_selected = judge_winner == gold_side
        gold_not_lost = gold_selected or judge_winner == "tie"
        if gold_selected:
            outcome = "gold_selected"
        elif judge_winner == "tie":
            outcome = "tie"
        else:
            outcome = "generated_selected"
        gold_wins += 1 if gold_selected else 0
        gold_wins_or_ties += 1 if gold_not_lost else 0
        row = {
            "item_id": item_id,
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "generated_strategy": generated_strategy,
            "gold_side": gold_side,
            "judge_winner": judge_winner,
            "gold_selected": gold_selected,
            "gold_not_lost": gold_not_lost,
            "outcome": outcome,
        }
        if metrics is not None:
            metric_snapshot = traditional_metric_snapshot(metrics.per_image.get((generated_strategy, item_id)))
            if metric_snapshot is not None:
                row["generated_traditional_metrics"] = metric_snapshot
        rows.append(row)
    return {
        "comparison_type": "gold_vs_generated",
        "gold_strategy": gold_strategy,
        "comparable_items": len(rows),
        "gold_win_rate": gold_wins / len(rows) if rows else None,
        "gold_win_or_tie_rate": gold_wins_or_ties / len(rows) if rows else None,
        "generated_metric_summary_by_outcome": _summarize_generated_metrics_by_outcome(
            rows,
            outcome_key="outcome",
        ),
        "rows": rows,
    }
