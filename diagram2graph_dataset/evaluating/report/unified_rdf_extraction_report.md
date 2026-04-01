# Unified evaluation Report: RDF Triple Extraction from Images

This consolidated report merges:
- **Node–Edge (separate)** metrics (nodes and edges evaluated independently),
- **Node–Edge (simultaneous)** metrics (whole-graph evaluation), and
- **ragas** metrics (e.g., multi-modal relevance, multi-modal faithfulness, or available overall scores).

**Prompting strategies compared:** zero-shot, one-shot, few-shot.

## 1) Node–Edge (Separate)
### Mean Scores – Nodes
| prompting | precision_nodes | recall_nodes | f1_nodes | jaccard_nodes |
| --- | --- | --- | --- | --- |
| zero-shot | 0.3347 | 0.3356 | 0.3351 | 0.2076 |
| one-shot | 0.3341 | 0.3350 | 0.3345 | 0.2074 |
| few-shot | 0.3367 | 0.3377 | 0.3371 | 0.2090 |


### Mean Scores – Edges
| prompting | precision_edges | recall_edges | f1_edges | jaccard_edges |
| --- | --- | --- | --- | --- |
| zero-shot | 0.7863 | 0.7902 | 0.7869 | 0.7328 |
| one-shot | 0.7779 | 0.7852 | 0.7812 | 0.7244 |
| few-shot | 0.7778 | 0.7872 | 0.7819 | 0.7260 |


### Macro Averages (Node / Edge / Graph)
| prompting | node_macro_mean | edge_macro_mean | graph_macro_mean |
| --- | --- | --- | --- |
| zero-shot | 0.3033 | 0.7741 | 0.5387 |
| one-shot | 0.3028 | 0.7672 | 0.5350 |
| few-shot | 0.3051 | 0.7682 | 0.5367 |


**Observations:**
- **precision_nodes** best by **few-shot** (mean=0.3367)
- **recall_nodes** best by **few-shot** (mean=0.3377)
- **f1_nodes** best by **few-shot** (mean=0.3371)
- **jaccard_nodes** best by **few-shot** (mean=0.2090)
- **precision_edges** best by **zero-shot** (mean=0.7863)
- **recall_edges** best by **zero-shot** (mean=0.7902)
- **f1_edges** best by **zero-shot** (mean=0.7869)
- **jaccard_edges** best by **zero-shot** (mean=0.7328)

**Charts:**
![](/mnt/data/combined_plots/node_sep_mean_precision_nodes.png)
![](/mnt/data/combined_plots/node_sep_mean_recall_nodes.png)
![](/mnt/data/combined_plots/node_sep_mean_f1_nodes.png)
![](/mnt/data/combined_plots/node_sep_mean_jaccard_nodes.png)
![](/mnt/data/combined_plots/node_sep_mean_precision_edges.png)
![](/mnt/data/combined_plots/node_sep_mean_recall_edges.png)
![](/mnt/data/combined_plots/node_sep_mean_f1_edges.png)
![](/mnt/data/combined_plots/node_sep_mean_jaccard_edges.png)

![](/mnt/data/combined_plots/macros_mean_node_macro_mean.png)
![](/mnt/data/combined_plots/macros_mean_edge_macro_mean.png)
![](/mnt/data/combined_plots/macros_mean_graph_macro_mean.png)

## 2) Node–Edge (Simultaneous / Whole-Graph)
### Mean Scores – Graph-level
| prompting | precision | recall | f1 | jaccard |
| --- | --- | --- | --- | --- |
| zero-shot | 0.5296 | 0.5042 | 0.5154 | 0.3590 |
| one-shot | 0.5910 | 0.5952 | 0.5929 | 0.4350 |
| few-shot | 0.7386 | 0.7444 | 0.7412 | 0.6194 |


**Observations:**
- **precision** best by **few-shot** (mean=0.7386)
- **recall** best by **few-shot** (mean=0.7444)
- **f1** best by **few-shot** (mean=0.7412)
- **jaccard** best by **few-shot** (mean=0.6194)

**Charts:**
![](/mnt/data/combined_plots/node_sim_mean_precision.png)
![](/mnt/data/combined_plots/node_sim_mean_recall.png)
![](/mnt/data/combined_plots/node_sim_mean_f1.png)
![](/mnt/data/combined_plots/node_sim_mean_jaccard.png)

## 3) ragas (Multi-modal)
### Mean Scores – ragas
| prompting | mean_score |
| --- | --- |
| zero-shot | 0.8653 |
| one-shot | 0.8899 |
| few-shot | 0.8203 |


**Observations:**
- **mean_score** best by **one-shot** (mean=0.8899)

**Charts:**
![](/mnt/data/combined_plots/ragas_mean_mean_score.png)

## 4) High-level Takeaways
- **Nodes / precision_nodes** → best mean by **few-shot** (=0.3367)
- **Nodes / recall_nodes** → best mean by **few-shot** (=0.3377)
- **Nodes / f1_nodes** → best mean by **few-shot** (=0.3371)
- **Nodes / jaccard_nodes** → best mean by **few-shot** (=0.2090)
- **Edges / precision_edges** → best mean by **zero-shot** (=0.7863)
- **Edges / recall_edges** → best mean by **zero-shot** (=0.7902)
- **Edges / f1_edges** → best mean by **zero-shot** (=0.7869)
- **Edges / jaccard_edges** → best mean by **zero-shot** (=0.7328)
- **Graph / precision** → best mean by **few-shot** (=0.7386)
- **Graph / recall** → best mean by **few-shot** (=0.7444)
- **Graph / f1** → best mean by **few-shot** (=0.7412)
- **Graph / jaccard** → best mean by **few-shot** (=0.6194)
- **ragas / mean_score** → best mean by **one-shot** (=0.8899)