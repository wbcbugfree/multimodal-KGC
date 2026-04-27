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
    "qwen_json_average": ["it1_json", "it2_json"],
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


def ordinal_node_subject(index: int) -> str:
    return f"Node{index}"


def add_node_id_aliases(node_id_map: dict[str, str], raw_node_id: Any, subject: str) -> None:
    node_id = string_value(raw_node_id).strip()
    if not node_id:
        return

    aliases = [node_id]
    stripped = node_id.strip("_-")
    if stripped and stripped not in aliases:
        aliases.append(stripped)
    if node_id.isdigit():
        aliases.append(f"Node{node_id}")
    elif node_id.lower().startswith("node") and node_id[4:].isdigit():
        aliases.append(node_id[4:])

    for alias in aliases:
        node_id_map.setdefault(alias, subject)


def build_node_maps(nodes: Sequence[Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    node_id_map: dict[str, str] = {}
    node_label_map: dict[str, list[str]] = {}
    for index, node in enumerate(nodes, start=1):
        subject = ordinal_node_subject(index)
        if isinstance(node, dict):
            label = string_value(node.get("label")).strip()
            if label:
                node_label_map.setdefault(label, []).append(subject)
            add_node_id_aliases(node_id_map, node.get("id"), subject)
    for index, _node in enumerate(nodes, start=1):
        subject = ordinal_node_subject(index)
        node_id_map.setdefault(str(index), subject)
        node_id_map.setdefault(subject, subject)
    return node_id_map, node_label_map


def edge_source(edge: dict[str, Any]) -> str:
    source = string_value(edge.get("source")).strip()
    if source:
        return source
    return string_value(edge.get("source_")).strip()


def edge_target(edge: dict[str, Any]) -> str:
    return string_value(edge.get("target")).strip()


def edge_subject(source: str, target: str, index: int) -> str:
    if source and target:
        return f"Edge{source.removeprefix('Node')}{target.removeprefix('Node')}"
    return f"EdgeUnknown{index}"


def resolve_node_reference(
    raw_node_id: str,
    node_id_map: dict[str, str],
    node_label_map: dict[str, list[str]],
    edge: dict[str, Any],
    field: str,
) -> str:
    node_id = raw_node_id.strip()
    if not node_id:
        return ""
    if node_id in node_id_map:
        return node_id_map[node_id]
    stripped = node_id.strip("_-")
    if stripped in node_id_map:
        return node_id_map[stripped]

    endpoint_label = string_value(edge.get(f"{field}_label")).strip()
    label_matches = node_label_map.get(endpoint_label, [])
    if len(label_matches) == 1:
        return label_matches[0]

    if node_id.isdigit():
        return f"Node{node_id}"
    if node_id.lower().startswith("node") and node_id[4:].isdigit():
        return f"Node{node_id[4:]}"
    return f"Node{node_id}"


def json_to_webnlg_graph(json_path: Path) -> list[list[str]]:
    data = load_json(json_path)
    triples: list[list[str]] = []
    nodes = data.get("nodes", [])
    node_id_map, node_label_map = build_node_maps(nodes)

    for index, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            triples.append([f"NodeInvalid{index}", "invalid_node", string_value(node)])
            continue
        subject = ordinal_node_subject(index)
        for field in NODE_FIELDS:
            triples.append([subject, field, string_value(node.get(field))])
        for extra_key in sorted(set(node) - {"id", *NODE_FIELDS}):
            triples.append([subject, extra_key, string_value(node.get(extra_key))])

    for index, edge in enumerate(data.get("edges", []), start=1):
        if not isinstance(edge, dict):
            triples.append([f"EdgeInvalid{index}", "invalid_edge", string_value(edge)])
            continue
        source = resolve_node_reference(edge_source(edge), node_id_map, node_label_map, edge, "source")
        target = resolve_node_reference(edge_target(edge), node_id_map, node_label_map, edge, "target")
        subject = edge_subject(source, target, index)
        values = {
            "source": source,
            "source_type": string_value(edge.get("source_type")),
            "source_label": string_value(edge.get("source_label")),
            "target": target,
            "target_type": string_value(edge.get("target_type")),
            "target_label": string_value(edge.get("target_label")),
            "type_of_edge": string_value(edge.get("type_of_edge")),
            "relationship_value": string_value(edge.get("relationship_value")),
            "relationship_type": string_value(edge.get("relationship_type")),
        }
        for field in EDGE_FIELDS:
            obj = values[field]
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


def mean_nested_numeric_dict(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}

    averaged: dict[str, Any] = {}
    for key, first_value in items[0].items():
        values = [item[key] for item in items if key in item]
        if not values:
            continue
        if isinstance(first_value, dict):
            averaged[key] = mean_nested_numeric_dict([value for value in values if isinstance(value, dict)])
        elif isinstance(first_value, (int, float)):
            averaged[key] = _mean(values)
    return averaged


def common_strategy_ids(
    gold_dir: Path,
    outputs_root: Path,
    directory_names: Sequence[str],
    requested_ids: Optional[Sequence[str]] = None,
) -> list[str]:
    common_ids: Optional[set[str]] = None
    for directory_name in directory_names:
        ids = set(collect_intersection_ids(gold_dir, outputs_root / directory_name, requested_ids=requested_ids))
        common_ids = ids if common_ids is None else common_ids & ids
    return sorted(common_ids or set(), key=sort_id)


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


def average_strategy_reports(strategy_name: str, run_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not run_reports:
        return {
            "strategy": strategy_name,
            "status": "no_runs",
            "intersection_ids": [],
            "intersection_size": 0,
            "summary": {},
            "per_image": [],
            "runs": [],
        }

    common_ids = set(run_reports[0].get("intersection_ids", []))
    for report in run_reports[1:]:
        common_ids &= set(report.get("intersection_ids", []))
    intersection_ids = sorted(common_ids, key=sort_id)

    per_image_by_run: list[dict[str, dict[str, Any]]] = []
    for report in run_reports:
        per_image_by_run.append({item["img_id"]: item for item in report.get("per_image", [])})

    averaged_per_image = []
    for img_id in intersection_ids:
        image_items = [run_items[img_id] for run_items in per_image_by_run if img_id in run_items]
        averaged_per_image.append(
            {
                "img_id": img_id,
                "triple_match_accuracy": _mean(item["triple_match_accuracy"] for item in image_items),
                "rouge": mean_nested_numeric_dict([item["rouge"] for item in image_items]),
                "bleu": mean_nested_numeric_dict([item["bleu"] for item in image_items]),
                "bert_score": mean_nested_numeric_dict([item["bert_score"] for item in image_items]),
                "normalized_ged": _mean(item["normalized_ged"] for item in image_items),
            }
        )

    return {
        "strategy": strategy_name,
        "status": "averaged" if intersection_ids else "no_overlap",
        "averaged_from": [
            {
                "run": report.get("strategy"),
                "prediction_dir": report.get("prediction_dir"),
                "intersection_size": report.get("intersection_size"),
            }
            for report in run_reports
        ],
        "num_runs": len(run_reports),
        "intersection_ids": intersection_ids,
        "intersection_size": len(intersection_ids),
        "summary": mean_nested_numeric_dict([report["summary"] for report in run_reports]),
        "per_image": averaged_per_image,
        "runs": list(run_reports),
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
        "strategy_definitions": {
            name: {
                "type": "average" if isinstance(spec, list) else "single_run",
                "folders": spec if isinstance(spec, list) else [spec],
            }
            for name, spec in DEFAULT_STRATEGIES.items()
            if name in selected_strategies
        },
        "strategies": {},
    }

    for strategy_name in selected_strategies:
        strategy_spec = DEFAULT_STRATEGIES[strategy_name]
        if isinstance(strategy_spec, str):
            pred_dir = outputs_root / strategy_spec
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
        else:
            averaged_ids = common_strategy_ids(
                gold_dir=gold_dir,
                outputs_root=outputs_root,
                directory_names=strategy_spec,
                requested_ids=requested_ids,
            )
            run_reports = []
            for run_index, directory_name in enumerate(strategy_spec, start=1):
                run_reports.append(
                    evaluate_strategy(
                        strategy_name=f"{strategy_name}_run{run_index}",
                        gold_dir=gold_dir,
                        pred_dir=outputs_root / directory_name,
                        metrics_module=metrics_module,
                        bert_model_type=bert_model_type,
                        bert_device=bert_device,
                        bert_batch_size=bert_batch_size,
                        offline_bert=offline_bert,
                        ged_workers=ged_workers,
                        requested_ids=averaged_ids,
                    )
                )
            strategy_report = average_strategy_reports(strategy_name, run_reports)
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
        help="Subset of strategies to evaluate. Defaults to the averaged Qwen baseline and Gemini JSON folders.",
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
