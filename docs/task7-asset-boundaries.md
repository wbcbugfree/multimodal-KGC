# Task 7 Asset Boundaries

This note separates reusable assets from legacy or experiment-specific areas within the normalized lowercase underscore path layout.

## Shared Reusable Helpers

- `common/` is the reusable helper area for config loading, repo-root discovery, and notebook bootstrap logic.

## soil_dataset

- Reusable canonical assets:
  - `soil_dataset/data/figures/`
  - `soil_dataset/data/tables/`
  - `soil_dataset/prompt_engineering/prompt_text/`
  - `soil_dataset/extract_all_rdfs/`
- Preserve as valuable outputs, but not reusable core:
  - `soil_dataset/build_kg/`
- Legacy or staging areas:
  - `soil_dataset/executability/`
  - `soil_dataset/evaluation/`
  - `soil_dataset/extract_rdf_method/`

## vistext

- Reusable canonical assets:
  - `vistext/data/images/`
  - `vistext/data/labels/`
  - `vistext/data/json2ttl/`
  - `vistext/prompt_text/`
  - `vistext/data/json2ttl/json_to_ttl_converter_v2.py`
  - `vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py`
- Preserve as historical experiment outputs:
  - `vistext/extract_rdf_ttl/vistext_zeroshot_outputs/`
  - `vistext/extract_rdf_ttl/vistext_oneshot_static_outputs/`
  - `vistext/extract_rdf_ttl/vistext_oneshot_dynamic_outputs/`
  - `vistext/extract_rdf_ttl/vistext_fewshot_outputs/`

## diagram2graph_dataset

- Reusable canonical assets:
  - `diagram2graph_dataset/data/diagram2graph/`
  - `diagram2graph_dataset/data/labels/`
  - `diagram2graph_dataset/json2ttl/script.py`
  - `diagram2graph_dataset/evaluation/report/plot_summaries.py`
  - `diagram2graph_dataset/prompt_engineering/prompt_text/`
- Preserve as historical experiment outputs:
  - `diagram2graph_dataset/extract_rdf_ttl/`
  - `diagram2graph_dataset/build_kg/`
  - `diagram2graph_dataset/prompt_engineering/zeroshot_outputs/`
  - `diagram2graph_dataset/prompt_engineering/oneshot_outputs/`
  - `diagram2graph_dataset/prompt_engineering/fewshot_outputs/`
- Legacy or staging areas:
  - `diagram2graph_dataset/extract_rdf_json/`
  - `diagram2graph_dataset/prompt_engineering/evaluation/`
  - `diagram2graph_dataset/evaluation/report/plot_summaries_claude.py`

## Guardrails

- Do not reintroduce mixed-case or space-bearing dataset paths.
- Do not move preserved outputs into a new top-level archive folder in this phase.
- Prefer labeling and documenting historical areas before attempting later cleanup.
