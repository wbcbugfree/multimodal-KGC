#!/usr/bin/env python3
"""Evaluate Diagram2Graph JSON ablation outputs against JSON ground truth."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_GOLD_DIR = SCRIPT_DIR / "eval_gt"
DEFAULT_OUTPUTS_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT = SCRIPT_DIR / "json_ablation_evaluation_results.json"
GED_CHECK_INTERVAL = 10.0
GED_STABLE_THRESHOLD = 5
GED_MAX_TIME = 300.0
DEFAULT_GED_WORKERS = 5
DEFAULT_BERT_DEVICE = "auto"
DEFAULT_STRATEGIES = {
    "qwen_it1_json": "it1_json",
    "qwen_it2_json": "it2_json",
    "gemini_zeroshot_json": "gemini_zeroshot_json",
    "gemini_oneshot_json": "gemini_oneshot_json",
    "gemini_fewshot_json": "gemini_fewshot_json",
}
NODE_FIELDS = ["type_of_node", "shape", "label"]
EDGE_FIELDS = [
    "source",
    "source_type",
    "source_label",
    "target",
    "target_type",
    "target_label",
    "type_of_edge",
    "relationship_value",
    "relationship_type",
]


def load_graph_matching_module(offline_bert: bool = True):
    if offline_bert:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from diagram2graph.evaluation import graph_matching

    return graph_matching


def resolve_bert_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if normalized == "cuda":
        try:
            import torch
        except Exception as exc:
            raise SystemExit(f"--bert-device cuda requires PyTorch with CUDA support: {exc}") from exc
        if not torch.cuda.is_available():
            raise SystemExit("--bert-device cuda was requested, but torch.cuda.is_available() is False")
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise SystemExit("--bert-device must be one of: auto, cpu, cuda")


def sort_id(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def collect_intersection_ids(
    gold_dir: Path,
    pred_dir: Path,
    requested_ids: Optional[Sequence[str]] = None,
) -> list[str]:
    gold_ids = {path.stem for path in gold_dir.glob("*.json")}
    pred_ids = {path.stem for path in pred_dir.glob("*.json")}
    intersection = gold_ids & pred_ids
    if requested_ids is not None:
        requested = {Path(raw_id).stem for raw_id in requested_ids}
        intersection &= requested
    return sorted(intersection, key=sort_id)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def node_subject(node: dict[str, Any], index: int) -> str:
    node_id = string_value(node.get("id")).strip()
    return f"Node{node_id or index}"


def edge_source(edge: dict[str, Any]) -> str:
    return string_value(edge.get("source", edge.get("source_", ""))).strip()


def edge_target(edge: dict[str, Any]) -> str:
    return string_value(edge.get("target")).strip()


def edge_subject(edge: dict[str, Any], index: int) -> str:
    source = edge_source(edge)
    target = edge_target(edge)
    if source and target:
        return f"Edge{source}{target}"
    return f"EdgeUnknown{index}"


def json_to_webnlg_graph(json_path: Path) -> list[list[str]]:
    data = load_json(json_path)
    triples: list[list[str]] = []

    for index, node in enumerate(data.get("nodes", []), start=1):
        if not isinstance(node, dict):
            triples.append([f"NodeInvalid{index}", "invalid_node", string_value(node)])
            continue
        subject = node_subject(node, index)
        for field in NODE_FIELDS:
            triples.append([subject, field, string_value(node.get(field))])
        for extra_key in sorted(set(node) - {"id", *NODE_FIELDS}):
            triples.append([subject, extra_key, string_value(node.get(extra_key))])

    for index, edge in enumerate(data.get("edges", []), start=1):
        if not isinstance(edge, dict):
            triples.append([f"EdgeInvalid{index}", "invalid_edge", string_value(edge)])
            continue
        subject = edge_subject(edge, index)
        values = {
            "source": edge_source(edge),
            "source_type": string_value(edge.get("source_type")),
            "source_label": string_value(edge.get("source_label")),
            "target": edge_target(edge),
            "target_type": string_value(edge.get("target_type")),
            "target_label": string_value(edge.get("target_label")),
            "type_of_edge": string_value(edge.get("type_of_edge")),
            "relationship_value": string_value(edge.get("relationship_value")),
            "relationship_type": string_value(edge.get("relationship_type")),
        }
        for field in EDGE_FIELDS:
            obj = values[field]
            if field in {"source", "target"} and obj:
                obj = f"Node{obj}"
            triples.append([subject, field, obj])
        for extra_key in sorted(set(edge) - {*EDGE_FIELDS, "source_"}):
            triples.append([subject, extra_key, string_value(edge.get(extra_key))])

    triples.sort(key=lambda triple: tuple(part.lower() for part in triple))
    return triples


def _float(value: Any) -> float:
    return float(value)


def _mean(values: Iterable[Any]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(float(value) for value in values))


def empty_strategy_report(strategy_name: str, pred_dir: Path, intersection_ids: list[str]) -> dict[str, Any]:
    return {
        "strategy": strategy_name,
        "prediction_dir": str(pred_dir.resolve()),
        "intersection_ids": intersection_ids,
        "intersection_size": len(intersection_ids),
        "status": "no_overlap",
        "summary": {
            "triple_match_micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "triple_match_accuracy_mean": 0.0,
            "rouge": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "bleu": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "bert_score": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "normalized_ged_mean": 0.0,
        },
        "per_image": [],
    }


def _ged_task(payload: tuple[list[list[str]], list[list[str]], float, int, float]) -> float:
    repo_root = Path(__file__).resolve().parents[3]
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
    bert_device: str = "cpu",
    bert_batch_size: Optional[int] = None,
    offline_bert: bool = True,
    ged_workers: int = DEFAULT_GED_WORKERS,
    requested_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    intersection_ids = collect_intersection_ids(gold_dir, pred_dir, requested_ids=requested_ids)
    print(f"[{strategy_name}] evaluating {len(intersection_ids)} overlapping files")
    if not intersection_ids:
        return empty_strategy_report(strategy_name, pred_dir, intersection_ids)

    metrics = metrics_module or load_graph_matching_module(offline_bert=offline_bert)
    gold_graphs = [json_to_webnlg_graph(gold_dir / f"{img_id}.json") for img_id in intersection_ids]
    pred_graphs = [json_to_webnlg_graph(pred_dir / f"{img_id}.json") for img_id in intersection_ids]

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
    bert_p, bert_r, bert_f = metrics.get_bert_score(
        gold_edges,
        pred_edges,
        model_type=bert_model_type,
        device=bert_device,
        batch_size=bert_batch_size,
    )
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
        "status": "evaluated",
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
    outputs_root: Path,
    bert_model_type: Optional[str] = None,
    bert_device: str = "cpu",
    bert_batch_size: Optional[int] = None,
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
        "outputs_root": str(outputs_root.resolve()),
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
            "bert_score": {
                "model_type": bert_model_type,
                "device": bert_device,
                "batch_size": bert_batch_size,
            },
            "graph_scope": "full_graph",
            "ordinal_normalization": False,
            "input_format": "json",
        },
        "strategies": {},
    }

    for strategy_name in selected_strategies:
        directory_name = DEFAULT_STRATEGIES[strategy_name]
        pred_dir = outputs_root / directory_name
        strategy_report = evaluate_strategy(
            strategy_name=strategy_name,
            gold_dir=gold_dir,
            pred_dir=pred_dir,
            metrics_module=metrics_module,
            bert_model_type=bert_model_type,
            bert_device=bert_device,
            bert_batch_size=bert_batch_size,
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
        description="Evaluate Diagram2Graph JSON ablation outputs against JSON ground truth."
    )
    parser.add_argument("--gold-dir", default=str(DEFAULT_GOLD_DIR))
    parser.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bert-model-type", default=None)
    parser.add_argument(
        "--bert-device",
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_BERT_DEVICE,
        help="Device for BERTScore inference. Use cuda for GPU acceleration, cpu for CPU, or auto to prefer CUDA when available.",
    )
    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=None,
        help="Optional BERTScore batch size. Lower this if cuda runs out of memory.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=sorted(DEFAULT_STRATEGIES.keys()),
        default=None,
        help="Subset of strategies to evaluate. Defaults to Qwen and Gemini JSON folders.",
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
    if args.bert_batch_size is not None and args.bert_batch_size < 1:
        raise SystemExit("--bert-batch-size must be at least 1")
    bert_device = resolve_bert_device(args.bert_device)

    report = build_report(
        gold_dir=Path(args.gold_dir),
        outputs_root=Path(args.outputs_root),
        bert_model_type=args.bert_model_type,
        bert_device=bert_device,
        bert_batch_size=args.bert_batch_size,
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
