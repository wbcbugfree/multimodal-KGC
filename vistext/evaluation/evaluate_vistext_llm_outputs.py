#!/usr/bin/env python3
"""Evaluate VisText LLM-generated TTL graphs against ground-truth TTL graphs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from rdflib import Graph, Literal, URIRef


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_GOLD_DIR = Path("vistext/data/json2ttl/ground_truth_ttl")
DEFAULT_EXTRACT_ROOT = Path("vistext/extract_rdf_ttl")
DEFAULT_LABELS_DIR = Path("vistext/data/labels")
DEFAULT_OUTPUT = Path("vistext/evaluation/vistext_llm_evaluation_results.json")
GED_CHECK_INTERVAL = 5.0
GED_STABLE_THRESHOLD = 3
GED_MAX_TIME = 120.0
DEFAULT_STRATEGIES = {
    "zeroshot": "vistext_zeroshot_outputs",
    "oneshot_static": "vistext_oneshot_static_outputs",
    "oneshot_dynamic": "vistext_oneshot_dynamic_outputs",
    "fewshot": "vistext_fewshot_outputs",
}


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


def ttl_to_webnlg_graph(ttl_path: Path) -> List[List[str]]:
    graph = Graph().parse(ttl_path, format="turtle")
    triples = [
        [normalize_rdf_term(subject), normalize_rdf_term(predicate), normalize_rdf_term(obj)]
        for subject, predicate, obj in graph
    ]
    return canonicalize_datapoint_subjects(triples)


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


def compute_ged_scores(
    gold_graphs: List[List[List[str]]],
    pred_graphs: List[List[List[str]]],
    metrics_module=None,
) -> List[float]:
    metrics = metrics_module or load_graph_matching_module()
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


def evaluate_strategy(
    strategy_name: str,
    gold_dir: Path,
    pred_dir: Path,
    metrics_module=None,
    bert_model_type: Optional[str] = None,
    offline_bert: bool = True,
) -> Dict[str, Any]:
    metrics = metrics_module or load_graph_matching_module(offline_bert=offline_bert)
    intersection_ids = collect_intersection_ids(gold_dir, pred_dir)
    print(f"[{strategy_name}] evaluating {len(intersection_ids)} overlapping files")

    gold_graphs = [ttl_to_webnlg_graph(gold_dir / f"{img_id}.ttl") for img_id in intersection_ids]
    pred_graphs = [ttl_to_webnlg_graph(pred_dir / f"{img_id}.ttl") for img_id in intersection_ids]

    print(f"[{strategy_name}] exact-match metrics")
    gold_edges = metrics.split_to_edges(gold_graphs)
    pred_edges = metrics.split_to_edges(pred_graphs)
    gold_tokens, pred_tokens = metrics.get_tokens(gold_edges, pred_edges)

    triple_precision, triple_recall, triple_f1 = metrics.get_triple_match_prf(gold_graphs, pred_graphs)
    graph_match_accuracy = metrics.get_graph_match_accuracy(pred_graphs, gold_graphs)
    print(f"[{strategy_name}] BLEU/ROUGE")
    rouge_p, rouge_r, rouge_f, bleu_p, bleu_r, bleu_f = metrics.get_bleu_rouge(
        gold_tokens, pred_tokens, gold_edges, pred_edges
    )
    print(f"[{strategy_name}] BERTScore")
    bert_p, bert_r, bert_f = metrics.get_bert_score(gold_edges, pred_edges, model_type=bert_model_type)
    triple_accs = [metrics.get_triple_match_accuracy(pred_graph, gold_graph) for pred_graph, gold_graph in zip(pred_graphs, gold_graphs)]
    print(f"[{strategy_name}] GED")
    ged_scores = compute_ged_scores(gold_graphs, pred_graphs, metrics_module=metrics_module)

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
            "graph_match_accuracy": _float(graph_match_accuracy),
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
    labels_dir: Path,
    bert_model_type: Optional[str] = None,
    offline_bert: bool = True,
    strategy_names: Optional[List[str]] = None,
    metrics_module=None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    selected_strategies = strategy_names or list(DEFAULT_STRATEGIES.keys())
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_dir": str(gold_dir.resolve()),
        "extract_root": str(extract_root.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "metrics_used": [
            "triple_match_micro_prf",
            "triple_match_accuracy",
            "graph_match_accuracy",
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
            }
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
        )
        if strategy_name == "oneshot_dynamic":
            strategy_report["chart_type_classification"] = compute_dynamic_chart_accuracy(
                pred_dir / "manifest.json",
                labels_dir,
                strategy_report["intersection_ids"],
            )
        report["strategies"][strategy_name] = strategy_report
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

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
        "--allow-online-model-download",
        action="store_true",
        help="Allow BERTScore to query/download Hugging Face models instead of forcing offline cache usage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        gold_dir=Path(args.gold_dir),
        extract_root=Path(args.extract_root),
        labels_dir=Path(args.labels_dir),
        bert_model_type=args.bert_model_type,
        offline_bert=not args.allow_online_model_download,
        strategy_names=args.strategies,
        output_path=Path(args.output),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
