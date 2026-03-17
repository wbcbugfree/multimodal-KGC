import argparse
import csv
import sys
from pathlib import Path


EDGE_RELATIONSHIPS = ("d2g:follows", "d2g:branches", "d2g:source", "d2g:target")


def sortable_image_name(image_name: str) -> tuple[int, str]:
    try:
        return (0, f"{int(image_name):012d}")
    except ValueError:
        return (1, image_name)


def parse_triples(file_path: Path) -> tuple[set[str], set[str]]:
    node_triples = []
    edge_triples = []
    content = file_path.read_text(encoding="utf-8")

    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("@") or stripped.startswith("#"):
            continue
        lines.append(line)

    for statement in "\n".join(lines).split("."):
        normalized = " ".join(statement.split())
        if not normalized:
            continue

        if "/node/" in normalized and "/edge/" not in normalized:
            if any(relationship in normalized for relationship in EDGE_RELATIONSHIPS):
                edge_triples.append(normalized)
            else:
                node_triples.append(normalized)
        elif "/edge/" in normalized or any(
            relationship in normalized for relationship in EDGE_RELATIONSHIPS
        ):
            edge_triples.append(normalized)
        else:
            node_triples.append(normalized)

    return set(node_triples), set(edge_triples)


def calculate_metrics(label_set: set[str], output_set: set[str]) -> dict[str, object]:
    overlap = label_set & output_set
    precision = len(overlap) / len(output_set) if output_set else 0
    recall = len(overlap) / len(label_set) if label_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    union = label_set | output_set
    jaccard = len(overlap) / len(union) if union else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(jaccard, 4),
        "overlap": len(overlap),
        "label_count": len(label_set),
        "output_count": len(output_set),
    }


def load_ttl_map(folder: Path) -> dict[str, Path]:
    return {path.name: path for path in folder.iterdir() if path.is_file() and path.suffix == ".ttl"}


def evaluate_folders(label_dir: Path, output_dir: Path) -> list[list[object]]:
    label_map = load_ttl_map(label_dir)
    output_map = load_ttl_map(output_dir)

    rows = []
    for name, label_path in label_map.items():
        if name not in output_map:
            continue

        output_path = output_map[name]
        label_nodes, label_edges = parse_triples(label_path)
        output_nodes, output_edges = parse_triples(output_path)

        node_metrics = calculate_metrics(label_nodes, output_nodes)
        edge_metrics = calculate_metrics(label_edges, output_edges)
        overall_metrics = calculate_metrics(label_nodes | label_edges, output_nodes | output_edges)

        rows.append(
            [
                name.removesuffix(".ttl"),
                node_metrics["label_count"],
                node_metrics["output_count"],
                node_metrics["overlap"],
                node_metrics["precision"],
                node_metrics["recall"],
                node_metrics["f1"],
                node_metrics["jaccard"],
                edge_metrics["label_count"],
                edge_metrics["output_count"],
                edge_metrics["overlap"],
                edge_metrics["precision"],
                edge_metrics["recall"],
                edge_metrics["f1"],
                edge_metrics["jaccard"],
                overall_metrics["label_count"],
                overall_metrics["output_count"],
                overall_metrics["overlap"],
                overall_metrics["precision"],
                overall_metrics["recall"],
                overall_metrics["f1"],
                overall_metrics["jaccard"],
            ]
        )

    rows.sort(key=lambda row: sortable_image_name(str(row[0])))
    return rows


def write_csv(output_csv: Path, rows: list[list[object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "node_label_triples",
                "node_output_triples",
                "node_overlap_triples",
                "node_precision",
                "node_recall",
                "node_f1",
                "node_jaccard",
                "edge_label_triples",
                "edge_output_triples",
                "edge_overlap_triples",
                "edge_precision",
                "edge_recall",
                "edge_f1",
                "edge_jaccard",
                "overall_label_triples",
                "overall_output_triples",
                "overall_overlap_triples",
                "overall_precision",
                "overall_recall",
                "overall_f1",
                "overall_jaccard",
            ]
        )
        writer.writerows(rows)


def print_summary(rows: list[list[object]]) -> None:
    if not rows:
        print("No matching TTL filenames were found.")
        return

    node_f1_avg = sum(float(str(row[6])) for row in rows) / len(rows)
    edge_f1_avg = sum(float(str(row[13])) for row in rows) / len(rows)
    overall_f1_avg = sum(float(str(row[20])) for row in rows) / len(rows)

    print("\n=== Summary Statistics ===")
    print(f"Average Node F1-Score: {node_f1_avg:.4f}")
    print(f"Average Edge F1-Score: {edge_f1_avg:.4f}")
    print(f"Average Overall F1-Score: {overall_f1_avg:.4f}")


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parents[1]

    parser = argparse.ArgumentParser(
        description="Evaluate node, edge, and overall F1 metrics between Turtle folders."
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=dataset_dir / "JSON2ttl" / "out_folder",
        help="Folder containing reference TTL files.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=dataset_dir / "prompt engineering" / "ZeroShot_outputs",
        help="Folder containing predicted TTL files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_dir / "result" / "Node_edge_sepratly" / "ZeroShot_outputs_node_edge_separate.csv",
        help="CSV file to write evaluation results to.",
    )
    return parser


def validate_dir(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {resolved}")
    return resolved


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        labels_dir = validate_dir(args.labels_dir, "Labels directory")
        outputs_dir = validate_dir(args.outputs_dir, "Outputs directory")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    rows = evaluate_folders(labels_dir, outputs_dir)
    output_csv = args.output_csv.resolve()
    write_csv(output_csv, rows)

    print(f"Saved {len(rows)} matching evaluations to {output_csv}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
