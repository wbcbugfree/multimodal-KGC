from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ContentOnlyMetrics:
    per_image: dict[tuple[str, str], Mapping[str, Any]]
    strategy_summary: dict[str, Mapping[str, Any]]


def _as_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


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
    return ContentOnlyMetrics(per_image=per_image, strategy_summary=summary)


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
        }

    best_pair: dict[str, Any] | None = None
    for first, second in combinations(available, 2):
        first_summary = metrics.strategy_summary[first]
        second_summary = metrics.strategy_summary[second]
        triple_accuracy_gap = abs(
            (_as_float(first_summary.get("triple_match_accuracy_mean")) or 0.0)
            - (_as_float(second_summary.get("triple_match_accuracy_mean")) or 0.0)
        )
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
            "triple_match_accuracy_gap": triple_accuracy_gap,
            "triple_match_micro_f1_gap": triple_f1_gap,
            "normalized_ged_gap": normalized_ged_gap,
            "composite_gap": _mean([triple_accuracy_gap, triple_f1_gap, normalized_ged_gap]) or 0.0,
        }
        if best_pair is None or (
            pair_summary["composite_gap"],
            pair_summary["triple_match_accuracy_gap"],
            pair_summary["normalized_ged_gap"],
        ) > (
            best_pair["composite_gap"],
            best_pair["triple_match_accuracy_gap"],
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
            "best_pair": best_pair,
        }

    return {
        "status": "selected",
        "candidate_strategies": strategies,
        "available_strategies": available,
        "min_gap": min_gap,
        "best_pair": best_pair,
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
        rows.append(
            {
                "strategy": strategy,
                "item_id": item_id,
                "overall_score": _score(item, "overall_score"),
                "criteria_mean": _score(item, "criteria_mean"),
                "triple_match_accuracy": metric.get("triple_match_accuracy"),
                "inverse_normalized_ged": (
                    1.0 - float(metric["normalized_ged"])
                    if isinstance(metric.get("normalized_ged"), int | float)
                    else None
                ),
            }
        )

    overall = [row["overall_score"] for row in rows if row["overall_score"] is not None and row["triple_match_accuracy"] is not None]
    triple = [row["triple_match_accuracy"] for row in rows if row["overall_score"] is not None and row["triple_match_accuracy"] is not None]
    criteria = [row["criteria_mean"] for row in rows if row["criteria_mean"] is not None and row["triple_match_accuracy"] is not None]
    triple_for_criteria = [
        row["triple_match_accuracy"] for row in rows if row["criteria_mean"] is not None and row["triple_match_accuracy"] is not None
    ]
    ged_overall = [row["overall_score"] for row in rows if row["overall_score"] is not None and row["inverse_normalized_ged"] is not None]
    inverse_ged = [
        row["inverse_normalized_ged"] for row in rows if row["overall_score"] is not None and row["inverse_normalized_ged"] is not None
    ]

    return {
        "matched_items": len(rows),
        "spearman": {
            "overall_vs_triple_match_accuracy": _spearman(overall, triple),
            "criteria_mean_vs_triple_match_accuracy": _spearman(criteria, triple_for_criteria),
            "overall_vs_inverse_normalized_ged": _spearman(ged_overall, inverse_ged),
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
