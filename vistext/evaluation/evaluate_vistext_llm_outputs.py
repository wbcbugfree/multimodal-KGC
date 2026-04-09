#!/usr/bin/env python3
"""Evaluate VisText LLM-generated TTL graphs against ground-truth TTL graphs."""

from __future__ import annotations

import argparse
import multiprocessing
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rdflib import Graph, Literal, URIRef


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_GOLD_DIR = Path("vistext/data/json2ttl/ground_truth_ttl")
DEFAULT_EXTRACT_ROOT = Path("vistext/extract_rdf_ttl")
DEFAULT_LABELS_DIR = Path("vistext/data/labels")
DEFAULT_OUTPUT = Path("vistext/evaluation/vistext_llm_evaluation_results.json")
GED_CHECK_INTERVAL = 10.0
GED_STABLE_THRESHOLD = 5
GED_MAX_TIME = 300.0
DEFAULT_GED_WORKERS = 5
DEFAULT_GRAPH_MODES = ("full_graph", "content_only")
DEFAULT_NUMERIC_TOLERANCE = 0.0
DEFAULT_STRATEGIES = {
    "zeroshot": "vistext_zeroshot_outputs",
    "oneshot_static": "vistext_oneshot_static_outputs",
    "oneshot_dynamic": "vistext_oneshot_dynamic_outputs",
    "fewshot": "vistext_fewshot_outputs",
}
PURE_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
TIME_AXIS_TITLE_RE = re.compile(r"\b(year|years|date|dates|month|months|quarter|quarters|week|weeks|day|days|fy|financial year)\b", re.IGNORECASE)


def load_graph_matching_module(offline_bert: bool = True):
    if offline_bert:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vistext.evaluation import graph_matching

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


def is_datapoint_label(label: str) -> bool:
    lowered = label.lower()
    if lowered == "datapoint":
        return True
    if lowered.startswith("datapoint") and lowered[len("datapoint") :].isdigit():
        return True
    return False


def canonicalize_datapoint_subjects(triples: List[List[str]]) -> List[List[str]]:
    datapoint_subjects = {subject for subject, _, _ in triples if is_datapoint_label(subject)}
    if not datapoint_subjects:
        return triples

    grouped: Dict[str, List[List[str]]] = {subject: [] for subject in datapoint_subjects}
    for triple in triples:
        subject = triple[0]
        if subject in grouped:
            grouped[subject].append(triple)

    ordered_subjects = sorted(
        grouped,
        key=lambda subject: (
            tuple((predicate.lower(), obj.lower()) for _, predicate, obj in sorted(grouped[subject], key=lambda row: (row[1].lower(), row[2].lower()))),
            subject.lower(),
        ),
    )
    subject_map = {subject: f"DataPoint{index}" for index, subject in enumerate(ordered_subjects, start=1)}

    canonical_triples = []
    for subject, predicate, obj in triples:
        canonical_subject = subject_map.get(subject, subject)
        canonical_object = subject_map.get(obj, obj)
        canonical_predicate = "type" if predicate == "22-rdf-syntax-ns#type" else predicate
        if canonical_predicate == "type" and is_datapoint_label(obj):
            canonical_object = "DataPoint"
        canonical_triples.append([canonical_subject, canonical_predicate, canonical_object])
    canonical_triples.sort(key=lambda triple: tuple(part.lower() for part in triple))
    return canonical_triples


def _is_chart_type_triple(subject: Any, predicate: Any, obj: Any) -> bool:
    normalized_subject = normalize_rdf_term(subject).lower()
    normalized_predicate = normalize_rdf_term(predicate).lower()
    return normalized_subject == "chart" and normalized_predicate in {"22-rdf-syntax-ns#type", "type"}


def _keep_triple(subject: Any, predicate: Any, obj: Any, graph_mode: str) -> bool:
    if graph_mode == "full_graph":
        return True
    if graph_mode == "content_only":
        return isinstance(obj, Literal) or _is_chart_type_triple(subject, predicate, obj)
    raise ValueError(f"Unsupported graph mode: {graph_mode}")


def ttl_to_webnlg_graph(ttl_path: Path, graph_mode: str = "full_graph") -> List[List[str]]:
    graph = Graph().parse(ttl_path, format="turtle")
    triples = []
    for subject, predicate, obj in graph:
        if not _keep_triple(subject, predicate, obj, graph_mode):
            continue
        triples.append([normalize_rdf_term(subject), normalize_rdf_term(predicate), normalize_rdf_term(obj)])
    return canonicalize_datapoint_subjects(triples)


def is_pure_numeric_literal(value: str) -> bool:
    return bool(PURE_NUMERIC_RE.fullmatch(value.strip()))


def is_time_axis_title(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(TIME_AXIS_TITLE_RE.search(value))


def extract_axis_titles(graph: List[List[str]]) -> Dict[str, Optional[str]]:
    titles = {"xValue": None, "yValue": None}
    for subject, predicate, obj in graph:
        if predicate != "title":
            continue
        if subject == "XAxis":
            titles["xValue"] = obj
        elif subject == "YAxis":
            titles["yValue"] = obj
    return titles


def extract_datapoint_rows(graph: List[List[str]]) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    for subject, predicate, obj in graph:
        if not is_datapoint_label(subject):
            continue
        if predicate not in {"xValue", "yValue"}:
            continue
        rows.setdefault(subject, {})[predicate] = obj
    return rows


def axis_is_fully_numeric(rows: Dict[str, Dict[str, str]], predicate: str) -> bool:
    values = [row[predicate] for row in rows.values() if predicate in row]
    return bool(values) and all(is_pure_numeric_literal(value) for value in values)


def determine_tolerance_predicate(
    gold_graph: List[List[str]],
    pred_graph: List[List[str]],
) -> Tuple[Optional[str], Dict[str, Any]]:
    gold_rows = extract_datapoint_rows(gold_graph)
    pred_rows = extract_datapoint_rows(pred_graph)
    titles = extract_axis_titles(gold_graph)

    gold_x_numeric = axis_is_fully_numeric(gold_rows, "xValue")
    gold_y_numeric = axis_is_fully_numeric(gold_rows, "yValue")
    pred_x_numeric = axis_is_fully_numeric(pred_rows, "xValue")
    pred_y_numeric = axis_is_fully_numeric(pred_rows, "yValue")

    x_numeric = gold_x_numeric and pred_x_numeric
    y_numeric = gold_y_numeric and pred_y_numeric

    metadata = {
        "x_axis_title": titles["xValue"],
        "y_axis_title": titles["yValue"],
        "gold_x_numeric": gold_x_numeric,
        "gold_y_numeric": gold_y_numeric,
        "pred_x_numeric": pred_x_numeric,
        "pred_y_numeric": pred_y_numeric,
        "eligible_predicate": None,
        "reason": None,
    }

    if x_numeric and not y_numeric:
        metadata["eligible_predicate"] = "xValue"
        metadata["reason"] = "only_x_axis_fully_numeric"
        return "xValue", metadata
    if y_numeric and not x_numeric:
        metadata["eligible_predicate"] = "yValue"
        metadata["reason"] = "only_y_axis_fully_numeric"
        return "yValue", metadata
    if x_numeric and y_numeric:
        x_is_time = is_time_axis_title(titles["xValue"])
        y_is_time = is_time_axis_title(titles["yValue"])
        metadata["x_axis_is_time_like"] = x_is_time
        metadata["y_axis_is_time_like"] = y_is_time
        if x_is_time and not y_is_time:
            metadata["eligible_predicate"] = "yValue"
            metadata["reason"] = "both_numeric_x_axis_time_like"
            return "yValue", metadata
        if y_is_time and not x_is_time:
            metadata["eligible_predicate"] = "xValue"
            metadata["reason"] = "both_numeric_y_axis_time_like"
            return "xValue", metadata
        metadata["reason"] = "both_axes_numeric_without_single_time_like_axis"
        return None, metadata

    metadata["reason"] = "no_fully_numeric_axis"
    return None, metadata


def within_relative_tolerance(gold_value: float, predicted_value: float, tolerance: float) -> bool:
    if gold_value == 0:
        return predicted_value == 0
    lower = min(gold_value * (1 - tolerance), gold_value * (1 + tolerance))
    upper = max(gold_value * (1 - tolerance), gold_value * (1 + tolerance))
    if gold_value > 0:
        lower = max(0.0, lower)
    elif gold_value < 0:
        upper = min(0.0, upper)
    return lower <= predicted_value <= upper


def apply_numeric_tolerance_to_graph(
    gold_graph: List[List[str]],
    pred_graph: List[List[str]],
    numeric_tolerance: float,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    if numeric_tolerance <= 0:
        return pred_graph, {"enabled": False, "reason": "tolerance_disabled"}

    tolerant_predicate, metadata = determine_tolerance_predicate(gold_graph, pred_graph)
    metadata = dict(metadata)
    metadata["enabled"] = bool(tolerant_predicate)
    metadata["numeric_tolerance"] = numeric_tolerance
    metadata["matched_datapoints"] = 0
    metadata["normalized_literals"] = 0

    if not tolerant_predicate:
        return pred_graph, metadata

    counterpart_predicate = "yValue" if tolerant_predicate == "xValue" else "xValue"
    gold_rows = extract_datapoint_rows(gold_graph)
    pred_rows = extract_datapoint_rows(pred_graph)

    gold_key_map: Dict[str, str] = {}
    pred_key_map: Dict[str, str] = {}
    for subject, row in gold_rows.items():
        key = row.get(counterpart_predicate)
        if key is None or key in gold_key_map:
            metadata["enabled"] = False
            metadata["reason"] = f"duplicate_or_missing_gold_{counterpart_predicate}"
            return pred_graph, metadata
        gold_key_map[key] = subject
    for subject, row in pred_rows.items():
        key = row.get(counterpart_predicate)
        if key is None or key in pred_key_map:
            metadata["enabled"] = False
            metadata["reason"] = f"duplicate_or_missing_pred_{counterpart_predicate}"
            return pred_graph, metadata
        pred_key_map[key] = subject

    replacement_map: Dict[Tuple[str, str], str] = {}
    for key, gold_subject in gold_key_map.items():
        pred_subject = pred_key_map.get(key)
        if pred_subject is None:
            continue
        gold_value = gold_rows[gold_subject].get(tolerant_predicate)
        pred_value = pred_rows[pred_subject].get(tolerant_predicate)
        if gold_value is None or pred_value is None:
            continue
        if not (is_pure_numeric_literal(gold_value) and is_pure_numeric_literal(pred_value)):
            continue
        metadata["matched_datapoints"] += 1
        if within_relative_tolerance(float(gold_value), float(pred_value), numeric_tolerance):
            replacement_map[(pred_subject, tolerant_predicate)] = gold_value

    tolerant_graph = []
    for subject, predicate, obj in pred_graph:
        normalized_obj = replacement_map.get((subject, predicate), obj)
        if normalized_obj != obj:
            metadata["normalized_literals"] += 1
        tolerant_graph.append([subject, predicate, normalized_obj])
    return tolerant_graph, metadata


def prepare_structural_graphs(
    gold_graphs: List[List[List[str]]],
    pred_graphs: List[List[List[str]]],
    numeric_tolerance: float,
    intersection_ids: List[str],
) -> Tuple[List[List[List[str]]], List[List[List[str]]], List[Dict[str, Any]]]:
    tolerant_pred_graphs: List[List[List[str]]] = []
    tolerance_metadata: List[Dict[str, Any]] = []
    for img_id, gold_graph, pred_graph in zip(intersection_ids, gold_graphs, pred_graphs):
        tolerant_pred_graph, metadata = apply_numeric_tolerance_to_graph(
            gold_graph,
            pred_graph,
            numeric_tolerance=numeric_tolerance,
        )
        metadata["img_id"] = img_id
        tolerant_pred_graphs.append(tolerant_pred_graph)
        tolerance_metadata.append(metadata)
    return gold_graphs, tolerant_pred_graphs, tolerance_metadata


def collect_intersection_ids(gold_dir: Path, pred_dir: Path) -> List[str]:
    gold_ids = {path.stem for path in gold_dir.glob("*.ttl")}
    pred_ids = {path.stem for path in pred_dir.glob("*.ttl")}
    return sorted(gold_ids & pred_ids, key=lambda value: (int(value) if value.isdigit() else value))


def _float(value: Any) -> float:
    return float(value)


def _mean(values: Iterable[Any]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(float(v) for v in values))


def _confusion_template() -> Dict[str, Dict[str, int]]:
    chart_types = ("bar", "line", "area")
    return {gold: {pred: 0 for pred in chart_types} for gold in chart_types}


def compute_dynamic_chart_accuracy(manifest_path: Path, labels_dir: Path, intersection_ids: List[str]) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_id = {item["img_id"]: item for item in manifest.get("items", [])}
    per_image = []
    confusion = _confusion_template()
    correct = 0

    for img_id in intersection_ids:
        predicted = manifest_by_id.get(img_id, {}).get("chart_type")
        label_data = json.loads((labels_dir / f"{img_id}.json").read_text(encoding="utf-8"))
        gold = label_data.get("L1_properties", [None])[0]
        is_correct = predicted == gold
        if gold in confusion and predicted in confusion[gold]:
            confusion[gold][predicted] += 1
        if is_correct:
            correct += 1
        per_image.append(
            {
                "img_id": img_id,
                "predicted_chart_type": predicted,
                "gold_chart_type": gold,
                "correct": is_correct,
            }
        )

    total = len(intersection_ids)
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "confusion_matrix": confusion,
        "per_image": per_image,
    }


def _ged_task(payload: Tuple[List[List[str]], List[List[str]], float, int, float]) -> float:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from vistext.evaluation import graph_matching

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
    gold_graphs: List[List[List[str]]],
    pred_graphs: List[List[List[str]]],
    metrics_module=None,
    ged_workers: int = DEFAULT_GED_WORKERS,
) -> List[float]:
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
    gold_graphs: Sequence[List[List[str]]],
    pred_graphs: Sequence[List[List[str]]],
    ged_workers: int,
) -> List[float]:
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
    graph_mode: str = "full_graph",
    metrics_module=None,
    bert_model_type: Optional[str] = None,
    offline_bert: bool = True,
    ged_workers: int = DEFAULT_GED_WORKERS,
    numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> Dict[str, Any]:
    metrics = metrics_module or load_graph_matching_module(offline_bert=offline_bert)
    intersection_ids = collect_intersection_ids(gold_dir, pred_dir)
    print(f"[{strategy_name}:{graph_mode}] evaluating {len(intersection_ids)} overlapping files")

    gold_graphs = [ttl_to_webnlg_graph(gold_dir / f"{img_id}.ttl", graph_mode=graph_mode) for img_id in intersection_ids]
    pred_graphs = [ttl_to_webnlg_graph(pred_dir / f"{img_id}.ttl", graph_mode=graph_mode) for img_id in intersection_ids]
    structural_gold_graphs, structural_pred_graphs, tolerance_metadata = prepare_structural_graphs(
        gold_graphs,
        pred_graphs,
        numeric_tolerance=numeric_tolerance,
        intersection_ids=intersection_ids,
    )

    print(f"[{strategy_name}:{graph_mode}] exact-match metrics")
    gold_edges = metrics.split_to_edges(gold_graphs)
    pred_edges = metrics.split_to_edges(pred_graphs)
    gold_tokens, pred_tokens = metrics.get_tokens(gold_edges, pred_edges)

    triple_precision, triple_recall, triple_f1 = metrics.get_triple_match_prf(structural_gold_graphs, structural_pred_graphs)
    print(f"[{strategy_name}:{graph_mode}] BLEU/ROUGE")
    rouge_p, rouge_r, rouge_f, bleu_p, bleu_r, bleu_f = metrics.get_bleu_rouge(
        gold_tokens, pred_tokens, gold_edges, pred_edges
    )
    print(f"[{strategy_name}:{graph_mode}] BERTScore")
    bert_p, bert_r, bert_f = metrics.get_bert_score(gold_edges, pred_edges, model_type=bert_model_type)
    triple_accs = [
        metrics.get_triple_match_accuracy(pred_graph, gold_graph)
        for pred_graph, gold_graph in zip(structural_pred_graphs, structural_gold_graphs)
    ]
    print(f"[{strategy_name}:{graph_mode}] GED")
    ged_scores = compute_ged_scores(
        structural_gold_graphs,
        structural_pred_graphs,
        metrics_module=metrics_module,
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
                "tolerance": tolerance_metadata[index],
            }
        )

    return {
        "strategy": strategy_name,
        "graph_mode": graph_mode,
        "numeric_tolerance": numeric_tolerance,
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
        "tolerance": {
            "enabled": numeric_tolerance > 0,
            "relative_tolerance": numeric_tolerance,
            "eligible_predicate_counts": {
                "xValue": sum(1 for item in tolerance_metadata if item.get("eligible_predicate") == "xValue" and item.get("enabled")),
                "yValue": sum(1 for item in tolerance_metadata if item.get("eligible_predicate") == "yValue" and item.get("enabled")),
            },
            "normalized_literals_total": sum(int(item.get("normalized_literals", 0)) for item in tolerance_metadata),
        },
        "per_image": per_image,
    }


def build_report(
    gold_dir: Path,
    extract_root: Path,
    labels_dir: Path,
    bert_model_type: Optional[str] = None,
    offline_bert: bool = True,
    strategy_names: Optional[List[str]] = None,
    graph_modes: Optional[List[str]] = None,
    metrics_module=None,
    output_path: Optional[Path] = None,
    ged_workers: int = DEFAULT_GED_WORKERS,
    numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> Dict[str, Any]:
    selected_strategies = strategy_names or list(DEFAULT_STRATEGIES.keys())
    selected_graph_modes = graph_modes or list(DEFAULT_GRAPH_MODES)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_dir": str(gold_dir.resolve()),
        "extract_root": str(extract_root.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "graph_modes_requested": selected_graph_modes,
        "metrics_used": [
            "triple_match_micro_prf",
            "triple_match_accuracy",
            "rouge",
            "bleu",
            "bert_score",
            "normalized_ged",
        ],
        "excluded_metrics": ["optimal_edit_path", "hallucination", "omission"],
        "metric_parameters": {
            "ged": {
                "check_interval": GED_CHECK_INTERVAL,
                "stable_threshold": GED_STABLE_THRESHOLD,
                "max_time": GED_MAX_TIME,
                "workers": ged_workers,
            },
            "numeric_tolerance": {
                "relative_tolerance": numeric_tolerance,
                "structural_metrics_only": True,
            },
        },
        "graph_mode_definitions": {
            "full_graph": "Evaluate all triples in the normalized RDF graph.",
            "content_only": "Evaluate literal-valued triples plus the Chart rdf:type triple only.",
        },
        "graph_modes": {},
    }

    for graph_mode in selected_graph_modes:
        mode_report = {"strategies": {}}
        for strategy_name in selected_strategies:
            directory_name = DEFAULT_STRATEGIES[strategy_name]
            pred_dir = extract_root / directory_name
            strategy_report = evaluate_strategy(
                strategy_name=strategy_name,
                gold_dir=gold_dir,
                pred_dir=pred_dir,
                graph_mode=graph_mode,
                metrics_module=metrics_module,
                bert_model_type=bert_model_type,
                offline_bert=offline_bert,
                ged_workers=ged_workers,
                numeric_tolerance=numeric_tolerance,
            )
            if strategy_name == "oneshot_dynamic":
                strategy_report["chart_type_classification"] = compute_dynamic_chart_accuracy(
                    pred_dir / "manifest.json",
                    labels_dir,
                    strategy_report["intersection_ids"],
                )
            mode_report["strategies"][strategy_name] = strategy_report
            if output_path is not None:
                report["graph_modes"][graph_mode] = mode_report
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["graph_modes"][graph_mode] = mode_report

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisText LLM-generated TTL outputs against ground truth.")
    parser.add_argument("--gold-dir", default=str(DEFAULT_GOLD_DIR))
    parser.add_argument("--extract-root", default=str(DEFAULT_EXTRACT_ROOT))
    parser.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bert-model-type", default=None)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=sorted(DEFAULT_STRATEGIES.keys()),
        default=None,
        help="Subset of strategies to evaluate. Defaults to all four strategies.",
    )
    parser.add_argument(
        "--graph-modes",
        nargs="+",
        choices=sorted(DEFAULT_GRAPH_MODES),
        default=None,
        help="Graph projections to evaluate. Defaults to both full_graph and content_only.",
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
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=DEFAULT_NUMERIC_TOLERANCE,
        help="Relative tolerance for structural numeric matching. Example: 0.01 means +/- 1%%. Applies only to quantity-like datapoint x/y literals.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ged_workers < 1:
        raise SystemExit("--ged-workers must be at least 1")
    if args.numeric_tolerance < 0:
        raise SystemExit("--numeric-tolerance must be non-negative")
    report = build_report(
        gold_dir=Path(args.gold_dir),
        extract_root=Path(args.extract_root),
        labels_dir=Path(args.labels_dir),
        bert_model_type=args.bert_model_type,
        offline_bert=not args.allow_online_model_download,
        strategy_names=args.strategies,
        graph_modes=args.graph_modes,
        output_path=Path(args.output),
        ged_workers=args.ged_workers,
        numeric_tolerance=args.numeric_tolerance,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
