# Node–Edge (Separately) evaluation Report

**Files analyzed:** `zeroshot.csv`, `oneshot.csv`, `fewshot.csv`.
**Prompting strategies compared:** zero-shot, one-shot, few-shot.

## Metrics Detected
Numeric columns treated as metrics:

image, gold_nodes, pred_nodes, tp_nodes, precision_nodes, recall_nodes, f1_nodes, jaccard_nodes, gold_edges, pred_edges, tp_edges, precision_edges, recall_edges, f1_edges, jaccard_edges

**Node metrics:** gold_nodes, pred_nodes, tp_nodes, precision_nodes, recall_nodes, f1_nodes, jaccard_nodes
**Edge metrics:** gold_edges, pred_edges, tp_edges, precision_edges, recall_edges, f1_edges, jaccard_edges
**Other metrics:** image

## Mean Scores by Prompting
| prompting | image | gold_nodes | pred_nodes | tp_nodes | precision_nodes | recall_nodes | f1_nodes | jaccard_nodes | gold_edges | pred_edges | tp_edges | precision_edges | recall_edges | f1_edges | jaccard_edges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| few-shot | 376.7169 | 8.0365 | 8.0685 | 2.6393 | 0.3367 | 0.3377 | 0.3371 | 0.2090 | 7.1324 | 7.0868 | 5.1370 | 0.7778 | 0.7872 | 0.7819 | 0.7260 |
| one-shot | 376.7169 | 8.0365 | 8.0639 | 2.6164 | 0.3341 | 0.3350 | 0.3345 | 0.2074 | 7.1324 | 7.0411 | 5.1233 | 0.7779 | 0.7852 | 0.7812 | 0.7244 |
| zero-shot | 376.7169 | 8.0365 | 8.0594 | 2.6301 | 0.3347 | 0.3356 | 0.3351 | 0.2076 | 7.1324 | 7.0228 | 5.1735 | 0.7863 | 0.7902 | 0.7869 | 0.7328 |


## Macro Averages (Node / Edge / Graph)
| prompting | node_macro_mean | edge_macro_mean | graph_macro_mean |
| --- | --- | --- | --- |
| few-shot | 2.8521 | 3.2042 | 3.0281 |
| one-shot | 2.8469 | 3.1951 | 3.0210 |
| zero-shot | 2.8484 | 3.2036 | 3.0260 |


## Best Prompt per Metric (by Mean)
| prompting | metric | mean |
| --- | --- | --- |
| zero-shot | f1_edges | 0.7869 |
| few-shot | f1_nodes | 0.3371 |
| few-shot | gold_edges | 7.1324 |
| few-shot | gold_nodes | 8.0365 |
| few-shot | image | 376.7169 |
| zero-shot | jaccard_edges | 0.7328 |
| few-shot | jaccard_nodes | 0.2090 |
| zero-shot | precision_edges | 0.7863 |
| few-shot | precision_nodes | 0.3367 |
| few-shot | pred_edges | 7.0868 |
| few-shot | pred_nodes | 8.0685 |
| zero-shot | recall_edges | 0.7902 |
| few-shot | recall_nodes | 0.3377 |
| zero-shot | tp_edges | 5.1735 |
| few-shot | tp_nodes | 2.6393 |


## Effect Sizes (Cohen's d)
Positive values mean the first method tends to score higher. Rough guide: 0.2=small, 0.5=medium, 0.8=large.
| metric | cohens_d_few_vs_one | cohens_d_few_vs_zero | cohens_d_one_vs_zero |
| --- | --- | --- | --- |
| image | 0.000 | 0.000 | 0.000 |
| gold_nodes | 0.000 | 0.000 | 0.000 |
| pred_nodes | 0.003 | 0.005 | 0.003 |
| tp_nodes | 0.024 | 0.010 | -0.014 |
| precision_nodes | 0.021 | 0.016 | -0.005 |
| recall_nodes | 0.023 | 0.018 | -0.005 |
| f1_nodes | 0.022 | 0.017 | -0.005 |
| jaccard_nodes | 0.017 | 0.016 | -0.002 |
| gold_edges | 0.000 | 0.000 | 0.000 |
| pred_edges | 0.021 | 0.029 | 0.008 |
| tp_edges | 0.006 | -0.016 | -0.022 |
| precision_edges | -0.000 | -0.028 | -0.028 |
| recall_edges | 0.007 | -0.010 | -0.017 |
| f1_edges | 0.002 | -0.017 | -0.019 |
| jaccard_edges | 0.004 | -0.020 | -0.024 |


## Relative Improvements (%)
Few-shot vs zero-shot and others (higher is better).
| metric | few_vs_zero_% | few_vs_one_% | one_vs_zero_% |
| --- | --- | --- | --- |
| image | 0.0 | 0.0 | 0.0 |
| gold_nodes | 0.0 | 0.0 | 0.0 |
| pred_nodes | 0.1 | 0.1 | 0.1 |
| tp_nodes | 0.3 | 0.9 | -0.5 |
| precision_nodes | 0.6 | 0.8 | -0.2 |
| recall_nodes | 0.6 | 0.8 | -0.2 |
| f1_nodes | 0.6 | 0.8 | -0.2 |
| jaccard_nodes | 0.7 | 0.8 | -0.1 |
| gold_edges | 0.0 | 0.0 | 0.0 |
| pred_edges | 0.9 | 0.6 | 0.3 |
| tp_edges | -0.7 | 0.3 | -1.0 |
| precision_edges | -1.1 | -0.0 | -1.1 |
| recall_edges | -0.4 | 0.3 | -0.6 |
| f1_edges | -0.6 | 0.1 | -0.7 |
| jaccard_edges | -0.9 | 0.2 | -1.1 |


## Charts: Mean by Prompting
![mean_f1_nodes](/mnt/data/node_edge_sep_charts/mean_f1_nodes.png)
![mean_f1_edges](/mnt/data/node_edge_sep_charts/mean_f1_edges.png)
![mean_precision_nodes](/mnt/data/node_edge_sep_charts/mean_precision_nodes.png)
![mean_precision_edges](/mnt/data/node_edge_sep_charts/mean_precision_edges.png)
![mean_recall_nodes](/mnt/data/node_edge_sep_charts/mean_recall_nodes.png)
![mean_recall_edges](/mnt/data/node_edge_sep_charts/mean_recall_edges.png)
![mean_jaccard_nodes](/mnt/data/node_edge_sep_charts/mean_jaccard_nodes.png)
![mean_jaccard_edges](/mnt/data/node_edge_sep_charts/mean_jaccard_edges.png)
![mean_image](/mnt/data/node_edge_sep_charts/mean_image.png)
![mean_gold_nodes](/mnt/data/node_edge_sep_charts/mean_gold_nodes.png)

## Charts: Distributions by Prompting
![box_f1_nodes](/mnt/data/node_edge_sep_boxplots/box_f1_nodes.png)
![box_f1_edges](/mnt/data/node_edge_sep_boxplots/box_f1_edges.png)
![box_precision_nodes](/mnt/data/node_edge_sep_boxplots/box_precision_nodes.png)
![box_precision_edges](/mnt/data/node_edge_sep_boxplots/box_precision_edges.png)
![box_recall_nodes](/mnt/data/node_edge_sep_boxplots/box_recall_nodes.png)
![box_recall_edges](/mnt/data/node_edge_sep_boxplots/box_recall_edges.png)
![box_jaccard_nodes](/mnt/data/node_edge_sep_boxplots/box_jaccard_nodes.png)
![box_jaccard_edges](/mnt/data/node_edge_sep_boxplots/box_jaccard_edges.png)
![box_image](/mnt/data/node_edge_sep_boxplots/box_image.png)
![box_gold_nodes](/mnt/data/node_edge_sep_boxplots/box_gold_nodes.png)

## Quick Observations
- **precision_nodes** best by **few-shot** (mean=0.3367)
- **recall_nodes** best by **few-shot** (mean=0.3377)
- **f1_nodes** best by **few-shot** (mean=0.3371)
- **jaccard_nodes** best by **few-shot** (mean=0.2090)
- **precision_edges** best by **zero-shot** (mean=0.7863)
- **recall_edges** best by **zero-shot** (mean=0.7902)
- **f1_edges** best by **zero-shot** (mean=0.7869)
- **jaccard_edges** best by **zero-shot** (mean=0.7328)