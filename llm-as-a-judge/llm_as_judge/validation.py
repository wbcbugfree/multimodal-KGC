from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ContentOnlyMetrics:
    per_image: dict[tuple[str, str], Mapping[str, Any]]
    strategy_summary: dict[str, Mapping[str, Any]]


def load_content_only_metrics(path: Path) -> ContentOnlyMetrics:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    strategies = data["graph_modes"]["content_only"]["strategies"]
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
