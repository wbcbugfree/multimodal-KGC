# pyright: reportMissingImports=false

"""Create comparison plots from the repo's diagram2graph evaluation CSV files."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    evaluating_dir = script_dir.parent
    f1_dir = evaluating_dir / "F1_Score" / "result"
    ragas_dir = evaluating_dir / "RAGas"

    parser = argparse.ArgumentParser(
        description="Generate mean bar plots for node-edge and RAGas summary CSV files."
    )
    parser.add_argument(
        "--node-sep-few",
        type=Path,
        default=f1_dir / "Node_edge_sepratly" / "FewShot_outputs_node_edge_separate.csv",
        help="Few-shot node/edge separate CSV.",
    )
    parser.add_argument(
        "--node-sep-one",
        type=Path,
        default=f1_dir / "Node_edge_sepratly" / "OneShot_outputs_node_edge_separate.csv",
        help="One-shot node/edge separate CSV.",
    )
    parser.add_argument(
        "--node-sep-zero",
        type=Path,
        default=f1_dir / "Node_edge_sepratly" / "ZeroShot_outputs_node_edge_separate.csv",
        help="Zero-shot node/edge separate CSV.",
    )
    parser.add_argument(
        "--node-sim-few",
        type=Path,
        default=f1_dir / "Node_Edge_simultaneous" / "FewShot_outputs.csv",
        help="Few-shot graph-level CSV.",
    )
    parser.add_argument(
        "--node-sim-one",
        type=Path,
        default=f1_dir / "Node_Edge_simultaneous" / "OneShot_outputs.csv",
        help="One-shot graph-level CSV.",
    )
    parser.add_argument(
        "--node-sim-zero",
        type=Path,
        default=f1_dir / "Node_Edge_simultaneous" / "ZeroShot_outputs.csv",
        help="Zero-shot graph-level CSV.",
    )
    parser.add_argument(
        "--ragas-few",
        type=Path,
        default=ragas_dir / "FewShot" / "summary_Few.csv",
        help="Few-shot RAGas summary CSV.",
    )
    parser.add_argument(
        "--ragas-one",
        type=Path,
        default=ragas_dir / "Oneshot" / "summary_One.csv",
        help="One-shot RAGas summary CSV.",
    )
    parser.add_argument(
        "--ragas-zero",
        type=Path,
        default=ragas_dir / "ZeroShot" / "summary_zero.csv",
        help="Zero-shot RAGas summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "repro_plots",
        help="Folder where PNG plots will be written.",
    )
    return parser


def tidy_cols(df):
    df = df.copy()
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]
    return df


def load_three_frames(pd, few_path: Path, one_path: Path, zero_path: Path):
    frames = []
    for prompting, path in (
        ("few-shot", few_path),
        ("one-shot", one_path),
        ("zero-shot", zero_path),
    ):
        frame = tidy_cols(pd.read_csv(path))
        frame["prompting"] = prompting
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_ragas_frames(pd, few_path: Path, one_path: Path, zero_path: Path):
    frames = []
    for prompting, path in (
        ("few-shot", few_path),
        ("one-shot", one_path),
        ("zero-shot", zero_path),
    ):
        frame = tidy_cols(pd.read_csv(path))
        if {"metric", "mean_score"}.issubset(frame.columns):
            metric_rows = {str(row.metric).strip().lower(): row.mean_score for row in frame.itertuples()}
            metric_rows["mean_score"] = frame["mean_score"].mean()
            frame = pd.DataFrame([metric_rows])
        frame["prompting"] = prompting
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def choose_metrics(df, preferred=None, exclude_cols=None):
    if preferred is None:
        preferred = []
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    drop_like = {"id", "index", "sample_id", "image_id", "img_id", "row", "col", "page"}
    numeric_cols = [column for column in numeric_cols if column not in drop_like and not column.endswith("_id")]
    numeric_cols = [column for column in numeric_cols if column not in exclude_cols]

    chosen = [column for column in preferred if column in numeric_cols]
    return chosen or numeric_cols


def mean_table(pd, df, metric_cols):
    if not metric_cols:
        return pd.DataFrame()

    out = df.groupby("prompting")[metric_cols].mean(numeric_only=True).reset_index()
    order = ["zero-shot", "one-shot", "few-shot"]
    out["prompting"] = pd.Categorical(out["prompting"], categories=order, ordered=True)
    return out.sort_values("prompting").reset_index(drop=True)


def barplot(plt, mean_df, metric, outdir: Path, prefix: str):
    if mean_df.empty or metric not in mean_df.columns:
        return None

    series = mean_df.set_index("prompting")[metric].dropna()
    if series.empty:
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    series.plot(kind="bar")
    plt.title(f"{prefix}: mean {metric} by prompting")
    plt.ylabel(f"mean({metric})")
    plt.xlabel("prompting")
    plt.tight_layout()
    output_path = outdir / f"{prefix}_mean_{metric}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def validate_paths(paths: list[Path]) -> None:
    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing required CSV files:\n{joined}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_paths(
            [
                args.node_sep_few,
                args.node_sep_one,
                args.node_sep_zero,
                args.node_sim_few,
                args.node_sim_one,
                args.node_sim_zero,
                args.ragas_few,
                args.ragas_one,
                args.ragas_zero,
            ]
        )
    except FileNotFoundError as exc:
        parser.exit(1, f"{exc}\n")

    import matplotlib.pyplot as plt
    import pandas as pd

    sep_df = load_three_frames(pd, args.node_sep_few, args.node_sep_one, args.node_sep_zero)
    sim_df = load_three_frames(pd, args.node_sim_few, args.node_sim_one, args.node_sim_zero)
    ragas_df = load_ragas_frames(pd, args.ragas_few, args.ragas_one, args.ragas_zero)

    sep_node_metrics = choose_metrics(
        sep_df,
        preferred=["node_precision", "node_recall", "node_f1", "node_jaccard"],
        exclude_cols=["image", "node_label_triples", "node_output_triples", "node_overlap_triples"],
    )
    sep_edge_metrics = choose_metrics(
        sep_df,
        preferred=["edge_precision", "edge_recall", "edge_f1", "edge_jaccard"],
        exclude_cols=["image", "edge_label_triples", "edge_output_triples", "edge_overlap_triples"],
    )
    sim_graph_metrics = choose_metrics(
        sim_df,
        preferred=["precision", "recall", "f1", "jaccard"],
        exclude_cols=["image", "label_triples", "output_triples", "overlap_triples"],
    )
    ragas_metrics = choose_metrics(ragas_df, preferred=["faithful_rate", "relevance_rate", "mean_score"])

    sep_nodes_mean = mean_table(pd, sep_df, sep_node_metrics)
    sep_edges_mean = mean_table(pd, sep_df, sep_edge_metrics)
    sim_mean = mean_table(pd, sim_df, sim_graph_metrics)
    ragas_mean = mean_table(pd, ragas_df, ragas_metrics)

    output_dir = args.output_dir.resolve()
    saved = []
    for metric in sep_node_metrics:
        path = barplot(plt, sep_nodes_mean, metric, output_dir, "node_sep")
        if path is not None:
            saved.append(path)
    for metric in sep_edge_metrics:
        path = barplot(plt, sep_edges_mean, metric, output_dir, "node_sep")
        if path is not None:
            saved.append(path)
    for metric in sim_graph_metrics:
        path = barplot(plt, sim_mean, metric, output_dir, "node_sim")
        if path is not None:
            saved.append(path)
    for metric in ragas_metrics:
        path = barplot(plt, ragas_mean, metric, output_dir, "ragas")
        if path is not None:
            saved.append(path)

    print(f"Saved {len(saved)} plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
