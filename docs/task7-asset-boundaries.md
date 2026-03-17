# Task 7 Asset Boundaries

This note separates reusable assets from legacy or experiment-specific areas without moving top-level dataset folders.

## Shared Reusable Helpers

- `common/` is the reusable helper area for config loading, repo-root discovery, and notebook bootstrap logic.

## Soil Dataset

- Reusable canonical assets:
  - `Soil Dataset/Data/figures/`
  - `Soil Dataset/Data/tables/`
  - `Soil Dataset/prompt engineering/Prompt Text/`
  - `Soil Dataset/Extract All RDFs/`
- Preserve as valuable outputs, but not reusable core:
  - `Soil Dataset/Build KG/`
- Legacy or staging areas:
  - `Soil Dataset/Executability/`
  - `Soil Dataset/Evaluating/`
  - `Soil Dataset/Extract RDF Method/`

## VisText

- Reusable canonical assets:
  - `VisText/Data/images/`
  - `VisText/Data/labels/`
  - `VisText/Data/Json2ttl/`
  - `VisText/Prompt Text/`
  - `VisText/Evaluation/json_to_ttl_converter_v2.py`
- Preserve as historical experiment outputs:
  - `VisText/Extract RDF ttl/vistext_Zeroshot_outputs/`
  - `VisText/Extract RDF ttl/vistext_Oneshot_outputs/`
  - `VisText/Extract RDF ttl/vistext_Fewshot_outputs/`
  - `VisText/Evaluation/gold_ttl_1000/`
- Legacy or manual staging area:
  - `VisText/Evaluation/complete_data.py`

## diagram2graph Dataset

- Reusable canonical assets:
  - `diagram2graph Dataset/Data/diagram2graph/`
  - `diagram2graph Dataset/Data/labels/`
  - `diagram2graph Dataset/JSON2ttl/Script.py`
  - `diagram2graph Dataset/Evaluating/report/plot_summaries.py`
  - `diagram2graph Dataset/prompt engineering/Prompt Text/`
- Preserve as historical experiment outputs:
  - `diagram2graph Dataset/Extract RDF ttl/`
  - `diagram2graph Dataset/Build KG/`
  - `diagram2graph Dataset/prompt engineering/ZeroShot_outputs/`
  - `diagram2graph Dataset/prompt engineering/OneShot_outputs/`
  - `diagram2graph Dataset/prompt engineering/FewShot_outputs/`
- Legacy or staging areas:
  - `diagram2graph Dataset/Extrcat RDF json/`
  - `diagram2graph Dataset/prompt engineering/Evaluation/`
  - `diagram2graph Dataset/Evaluating/report/plot_summaries Clude.py`

## Guardrails

- Do not rename typo-bearing legacy folders yet.
- Do not move preserved outputs into a new top-level archive folder in this phase.
- Prefer labeling and documenting historical areas before attempting later cleanup.
