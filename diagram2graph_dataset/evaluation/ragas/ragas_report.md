# ragas Metrics Report

**Files analyzed:** `summary_zero.csv`, `summary_one.csv`, `summary_few.csv`.
**Prompting strategies compared:** zero-shot, one-shot, few-shot.

## Metrics Detected
The following numeric columns were treated as metrics:

mean_score

## Mean Scores by Prompting
| prompting | mean_score |
| --- | --- |
| few-shot | 0.8203 |
| one-shot | 0.8899 |
| zero-shot | 0.8653 |


## Best Prompt per Metric (by Mean)
| prompting | metric | mean |
| --- | --- | --- |
| one-shot | mean_score | 0.8899 |


## Effect Sizes (Cohen's d)
Positive values mean the first method tends to score higher. Rough guide: 0.2=small, 0.5=medium, 0.8=large.
| metric | cohens_d_few_vs_one | cohens_d_few_vs_zero | cohens_d_one_vs_zero |
| --- | --- | --- | --- |
| mean_score | -0.386 | -0.235 | 0.163 |


## Charts: Mean by Prompting
![mean_mean_score](/mnt/data/ragas_charts/mean_mean_score.png)

## Charts: Distributions by Prompting
![box_mean_score](/mnt/data/ragas_boxplots/box_mean_score.png)

## Quick Observations
- **mean_score**: best mean with **one-shot** = 0.8899