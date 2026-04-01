"""
Reproduce summary plots for Node-Edge (separate/simultaneous) and ragas metrics.
Usage: run in an environment where the CSV files are available at the same paths used below.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("/mnt/data")

paths = {
  "node_sep": {
    "few": "/mnt/data/fewshot.csv",
    "one": "/mnt/data/oneshot.csv",
    "zero": "/mnt/data/zeroshot.csv"
  },
  "node_sim": {
    "few": "/mnt/data/fewshot_outputs.csv",
    "one": "/mnt/data/oneshot_outputs.csv",
    "zero": "/mnt/data/zeroshot_outputs.csv"
  },
  "ragas": {
    "few": "/mnt/data/summary_few.csv",
    "one": "/mnt/data/summary_one.csv",
    "zero": "/mnt/data/summary_zero.csv"
  }
}

def tidy_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def load_three(few, one, zero):
    d = []
    few = tidy_cols(pd.read_csv(few)); few["prompting"] = "few-shot"; d.append(few)
    one = tidy_cols(pd.read_csv(one)); one["prompting"] = "one-shot"; d.append(one)
    zero = tidy_cols(pd.read_csv(zero)); zero["prompting"] = "zero-shot"; d.append(zero)
    return pd.concat(d, ignore_index=True, sort=False)

def choose_metrics(df, include_keywords=None, exclude_cols=None):
    if include_keywords is None:
        include_keywords = []
    if exclude_cols is None:
        exclude_cols = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"id","index","sample_id","image_id","img_id","row","col","page"}
    num_cols = [c for c in num_cols if c not in drop_like and not c.endswith("_id")]
    num_cols = [c for c in num_cols if c not in exclude_cols]
    if include_keywords:
        filtered = [c for c in num_cols if any(k in c for k in include_keywords)]
        if filtered:
            return filtered
    return num_cols

def mean_table(df, metric_cols):
    if not metric_cols:
        return pd.dataFrame()
    out = df.groupby("prompting")[metric_cols].mean(numeric_only=True).reset_index()
    order = ["zero-shot","one-shot","few-shot"]
    out["prompting"] = pd.Categorical(out["prompting"], categories=order, ordered=True)
    out = out.sort_values("prompting").reset_index(drop=True)
    return out

def barplot(mean_df, metric, outdir, prefix):
    outdir.mkdir(parents=True, exist_ok=True)
    if mean_df is None or mean_df.empty or metric not in mean_df.columns:
        return None
    s = mean_df.set_index("prompting")[metric].dropna()
    if s.shape[0] == 0:
        return None
    plt.figure()
    s.plot(kind="bar")
    plt.title(f"{prefix}: mean {metric} by prompting")
    plt.ylabel(f"mean({metric})")
    plt.xlabel("prompting")
    plt.tight_layout()
    p = outdir/f"{prefix}_mean_{metric}.png"
    plt.savefig(p, dpi=200)
    plt.close()
    return p

# Load
sep_df = load_three(paths["node_sep"]["few"], paths["node_sep"]["one"], paths["node_sep"]["zero"])
sim_df = load_three(paths["node_sim"]["few"], paths["node_sim"]["one"], paths["node_sim"]["zero"])
ragas_df = load_three(paths["ragas"]["few"], paths["ragas"]["one"], paths["ragas"]["zero"])

# Metrics
sep_node_metrics = choose_metrics(sep_df, ["precision_nodes","recall_nodes","f1_nodes","jaccard_nodes"], ["image"])
sep_edge_metrics = choose_metrics(sep_df, ["precision_edges","recall_edges","f1_edges","jaccard_edges"], ["image"])
sim_graph_metrics = choose_metrics(sim_df, ["precision","recall","f1","jaccard"], ["image","label_triples","output_triples","overlap_triples"])
ragas_metrics = choose_metrics(ragas_df, ["relevance","faithfulness","mean_score"])

# Means
sep_nodes_mean = mean_table(sep_df, sep_node_metrics)
sep_edges_mean = mean_table(sep_df, sep_edge_metrics)
sim_mean = mean_table(sim_df, sim_graph_metrics)
ragas_mean = mean_table(ragas_df, ragas_metrics)

# Plots
out = base/"repro_plots"
out.mkdir(parents=True, exist_ok=True)

for m in (sep_node_metrics or []):
    barplot(sep_nodes_mean, m, out, "node_sep")
for m in (sep_edge_metrics or []):
    barplot(sep_edges_mean, m, out, "node_sep")
for m in (sim_graph_metrics or []):
    barplot(sim_mean, m, out, "node_sim")
for m in (ragas_metrics or []):
    barplot(ragas_mean, m, out, "ragas")

print("Saved plots to:", out)
