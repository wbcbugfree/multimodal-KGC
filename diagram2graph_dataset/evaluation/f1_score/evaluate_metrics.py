import argparse
import csv
import sys
from pathlib import Path


def sortable_image_name(image_name: str) -> tuple[int, str]:
    try:
        return (0, f"{int(image_name):012d}")
    except ValueError:
        return (1, image_name)


def parse_triples(file_path: Path) -> set[str]:
    triples = []
    content = file_path.read_text(encoding="utf-8")

    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("@") or stripped.startswith("#"):
            continue
        lines.append(line)

    for statement in "\n".join(lines).split("."):
        normalized = " ".join(statement.split())
        if normalized:
            triples.append(normalized)

    return set(triples)


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
        label_triples = parse_triples(label_path)
        output_triples = parse_triples(output_path)
        overlap = label_triples & output_triples

        precision = len(overlap) / len(output_triples) if output_triples else 0
        recall = len(overlap) / len(label_triples) if label_triples else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        union = label_triples | output_triples
        jaccard = len(overlap) / len(union) if union else 0

        rows.append(
            [
                name.removesuffix(".ttl"),
                len(label_triples),
                len(output_triples),
                len(overlap),
                round(precision, 4),
                round(recall, 4),
                round(f1, 4),
                round(jaccard, 4),
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
                "label_triples",
                "output_triples",
                "overlap_triples",
                "precision",
                "recall",
                "f1",
                "jaccard",
            ]
        )
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parents[1]

    parser = argparse.ArgumentParser(
        description="Evaluate graph-level F1 metrics between label and output Turtle folders."
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=dataset_dir / "json2ttl" / "out_folder",
        help="Folder containing reference TTL files.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=dataset_dir / "prompt_engineering" / "zeroshot_outputs",
        help="Folder containing predicted TTL files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_dir / "result" / "node_edge_simultaneous" / "zeroshot_outputs.csv",
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
    write_csv(args.output_csv.resolve(), rows)

    print(f"Saved {len(rows)} matching evaluations to {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
