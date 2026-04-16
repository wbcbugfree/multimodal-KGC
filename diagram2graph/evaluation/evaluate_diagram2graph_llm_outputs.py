#!/usr/bin/env python3
"""Evaluate Diagram2Graph LLM-generated TTL graphs against ground-truth TTL graphs."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional, Sequence

from rdflib import Graph, Literal, URIRef


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_GOLD_DIR = Path("diagram2graph/data/turtle")
DEFAULT_EXTRACT_ROOT = Path("diagram2graph/extract_rdf_ttl")
DEFAULT_OUTPUT = Path("diagram2graph/evaluation/diagram2graph_llm_evaluation_results.json")
GED_CHECK_INTERVAL = 10.0
GED_STABLE_THRESHOLD = 5
GED_MAX_TIME = 300.0
DEFAULT_GED_WORKERS = 5
DEFAULT_STRATEGIES = {
    "zeroshot": "zeroshot_outputs",
    "oneshot": "oneshot_outputs",
    "fewshot": "fewshot_outputs",
}


def load_graph_matching_module(offline_bert: bool = True):
    if offline_bert:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from diagram2graph.evaluation import graph_matching

    return graph_matching


def normalize_rdf_term(term: Any) -> str:
    if isinstance(term, Literal):
        return str(term)
    if isinstance(term, URIRef):
        text = str(term)
        if "#" in text:
            return text.rsplit("#", 1)[1]
        return text.rstrip("/").rsplit("/", 1)[-1]
    return str(term)


def ttl_to_webnlg_graph(ttl_path: Path) -> list[list[str]]:
    graph = Graph().parse(ttl_path, format="turtle")
    triples = [
        [normalize_rdf_term(subject), normalize_rdf_term(predicate), normalize_rdf_term(obj)]
        for subject, predicate, obj in graph
    ]
    triples.sort(key=lambda triple: tuple(part.lower() for part in triple))
    return triples


def sort_id(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def collect_intersection_ids(
    gold_dir: Path,
    pred_dir: Path,
    requested_ids: Optional[Sequence[str]] = None,
) -> list[str]:
    gold_ids = {path.stem for path in gold_dir.glob("*.ttl")}
    pred_ids = {path.stem for path in pred_dir.glob("*.ttl")}
    intersection = gold_ids & pred_ids
    if requested_ids is not None:
        requested = {Path(raw_id).stem for raw_id in requested_ids}
        intersection &= requested
    return sorted(intersection, key=sort_id)


def _float(value: Any) -> float:
    return float(value)


def _mean(values: Iterable[Any]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(float(value) for value in values))


def _ged_task(payload: tuple[list[list[str]], list[list[str]], float, int, float]) -> float:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from diagram2graph.evaluation import graph_matching

    gold_graph, pred_graph, check_interval, stable_threshold, max_time = payload
    return float(
        graph_matching.get_ged(
            gold_graph,
            pred_graph,
            check_interval=check_interval,
            stable_threshold=stable_threshold,
            max_time=max_time,
        )
    )


def compute_ged_scores(
    gold_graphs: list[list[list[str]]],
    pred_graphs: list[list[list[str]]],
    metrics_module=None,
    ged_workers: int = DEFAULT_GED_WORKERS,
) -> list[float]:
    metrics = metrics_module or load_graph_matching_module()
    if ged_workers > 1:
        try:
            return compute_ged_scores_parallel(gold_graphs, pred_graphs, ged_workers=ged_workers)
        except Exception as exc:
            print(f"  Parallel GED unavailable ({exc}); falling back to serial GED.")

    scores = []
    for index, (gold_graph, pred_graph) in enumerate(zip(gold_graphs, pred_graphs), start=1):
        print(f"  GED graph {index}/{len(gold_graphs)}")
        score = metrics.get_ged(
            gold_graph,
            pred_graph,
            check_interval=GED_CHECK_INTERVAL,
            stable_threshold=GED_STABLE_THRESHOLD,
            max_time=GED_MAX_TIME,
        )
        scores.append(float(score))
    return scores


def compute_ged_scores_parallel(
    gold_graphs: Sequence[list[list[str]]],
    pred_graphs: Sequence[list[list[str]]],
    ged_workers: int,
) -> list[float]:
    print(f"  GED parallel mode with {ged_workers} workers")
    payloads = [
        (gold_graph, pred_graph, GED_CHECK_INTERVAL, GED_STABLE_THRESHOLD, GED_MAX_TIME)
        for gold_graph, pred_graph in zip(gold_graphs, pred_graphs)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=ged_workers, mp_context=context) as executor:
        scores = list(executor.map(_ged_task, payloads))
    return [float(score) for score in scores]


def evaluate_strategy(
    strategy_name: str,
    gold_dir: Path,
    pred_dir: Path,
    metrics_module=None,
    bert_model_type: Optional[str] = None,
    offline_bert: bool = True,
    ged_workers: int = DEFAULT_GED_WORKERS,
    requested_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    metrics = metrics_module or load_graph_matching_module(offline_bert=offline_bert)
    intersection_ids = collect_intersection_ids(gold_dir, pred_dir, requested_ids=requested_ids)
    print(f"[{strategy_name}] evaluating {len(intersection_ids)} overlapping files")

    gold_graphs = [ttl_to_webnlg_graph(gold_dir / f"{img_id}.ttl") for img_id in intersection_ids]
    pred_graphs = [ttl_to_webnlg_graph(pred_dir / f"{img_id}.ttl") for img_id in intersection_ids]

    print(f"[{strategy_name}] exact-match metrics")
    gold_edges = metrics.split_to_edges(gold_graphs)
    pred_edges = metrics.split_to_edges(pred_graphs)
    gold_tokens, pred_tokens = metrics.get_tokens(gold_edges, pred_edges)

    triple_precision, triple_recall, triple_f1 = metrics.get_triple_match_prf(gold_graphs, pred_graphs)
    print(f"[{strategy_name}] BLEU/ROUGE")
    rouge_p, rouge_r, rouge_f, bleu_p, bleu_r, bleu_f = metrics.get_bleu_rouge(
        gold_tokens,
        pred_tokens,
        gold_edges,
        pred_edges,
    )
    print(f"[{strategy_name}] BERTScore")
    bert_p, bert_r, bert_f = metrics.get_bert_score(gold_edges, pred_edges, model_type=bert_model_type)
    triple_accs = [
        metrics.get_triple_match_accuracy(pred_graph, gold_graph)
        for pred_graph, gold_graph in zip(pred_graphs, gold_graphs)
    ]
    print(f"[{strategy_name}] GED")
    ged_scores = compute_ged_scores(
        gold_graphs,
        pred_graphs,
        metrics_module=metrics,
        ged_workers=ged_workers,
    )

    per_image = []
    for index, img_id in enumerate(intersection_ids):
        per_image.append(
            {
                "img_id": img_id,
                "triple_match_accuracy": _float(triple_accs[index]),
                "rouge": {
                    "precision": _float(rouge_p[index]),
                    "recall": _float(rouge_r[index]),
                    "f1": _float(rouge_f[index]),
                },
                "bleu": {
                    "precision": _float(bleu_p[index]),
                    "recall": _float(bleu_r[index]),
                    "f1": _float(bleu_f[index]),
                },
                "bert_score": {
                    "precision": _float(bert_p[index]),
                    "recall": _float(bert_r[index]),
                    "f1": _float(bert_f[index]),
                },
                "normalized_ged": _float(ged_scores[index]),
            }
        )

    return {
        "strategy": strategy_name,
        "prediction_dir": str(pred_dir.resolve()),
        "intersection_ids": intersection_ids,
        "intersection_size": len(intersection_ids),
        "summary": {
            "triple_match_micro": {
                "precision": _float(triple_precision),
                "recall": _float(triple_recall),
                "f1": _float(triple_f1),
            },
            "triple_match_accuracy_mean": _mean(triple_accs),
            "rouge": {
                "precision": _mean(rouge_p),
                "recall": _mean(rouge_r),
                "f1": _mean(rouge_f),
            },
            "bleu": {
                "precision": _mean(bleu_p),
                "recall": _mean(bleu_r),
                "f1": _mean(bleu_f),
            },
            "bert_score": {
                "precision": _mean(bert_p),
                "recall": _mean(bert_r),
                "f1": _mean(bert_f),
            },
            "normalized_ged_mean": _mean(ged_scores),
        },
        "per_image": per_image,
    }


def build_report(
    gold_dir: Path,
    extract_root: Path,
    bert_model_type: Optional[str] = None,
    offline_bert: bool = True,
    strategy_names: Optional[list[str]] = None,
    metrics_module=None,
    output_path: Optional[Path] = None,
    ged_workers: int = DEFAULT_GED_WORKERS,
    requested_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    selected_strategies = strategy_names or list(DEFAULT_STRATEGIES.keys())
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_dir": str(gold_dir.resolve()),
        "extract_root": str(extract_root.resolve()),
        "requested_ids": list(requested_ids) if requested_ids else None,
        "metrics_used": [
            "triple_match_micro_prf",
            "triple_match_accuracy",
            "rouge",
            "bleu",
            "bert_score",
            "normalized_ged",
        ],
        "excluded_metrics": [
            "graph_match_accuracy",
            "optimal_edit_path",
            "hallucination",
            "omission",
        ],
        "metric_parameters": {
            "ged": {
                "check_interval": GED_CHECK_INTERVAL,
                "stable_threshold": GED_STABLE_THRESHOLD,
                "max_time": GED_MAX_TIME,
                "workers": ged_workers,
            },
            "graph_scope": "full_graph",
            "ordinal_normalization": False,
        },
        "strategies": {},
    }

    for strategy_name in selected_strategies:
        directory_name = DEFAULT_STRATEGIES[strategy_name]
        pred_dir = extract_root / directory_name
        strategy_report = evaluate_strategy(
            strategy_name=strategy_name,
            gold_dir=gold_dir,
            pred_dir=pred_dir,
            metrics_module=metrics_module,
            bert_model_type=bert_model_type,
            offline_bert=offline_bert,
            ged_workers=ged_workers,
            requested_ids=requested_ids,
        )
        report["strategies"][strategy_name] = strategy_report
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Diagram2Graph LLM-generated TTL outputs against ground truth."
    )
    parser.add_argument("--gold-dir", default=str(DEFAULT_GOLD_DIR))
    parser.add_argument("--extract-root", default=str(DEFAULT_EXTRACT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bert-model-type", default=None)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=sorted(DEFAULT_STRATEGIES.keys()),
        default=None,
        help="Subset of strategies to evaluate. Defaults to all three strategies.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Optional diagram ids to evaluate. Defaults to each strategy/gold intersection.",
    )
    parser.add_argument(
        "--allow-online-model-download",
        action="store_true",
        help="Allow BERTScore to query/download Hugging Face models instead of forcing offline cache usage.",
    )
    parser.add_argument(
        "--ged-workers",
        type=int,
        default=DEFAULT_GED_WORKERS,
        help="Number of worker processes for GED. Use 1 for serial GED; values >1 enable process-based parallel GED.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ged_workers < 1:
        raise SystemExit("--ged-workers must be at least 1")

    report = build_report(
        gold_dir=Path(args.gold_dir),
        extract_root=Path(args.extract_root),
        bert_model_type=args.bert_model_type,
        offline_bert=not args.allow_online_model_download,
        strategy_names=args.strategies,
        output_path=Path(args.output),
        ged_workers=args.ged_workers,
        requested_ids=args.ids,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
