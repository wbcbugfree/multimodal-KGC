# Repository Map

This map reflects the current worktree state and the approved conservative refactor direction.

## Root Layout

- `soil_dataset/` - soil tables and figures, prompt_engineering, extraction notebooks, KG building, executability checks, and evaluation outputs.
- `vistext/` - chart dataset prompts, canonical image and label inputs, JSON-to-TTL conversion, Gemini extraction runners, and preserved outputs.
- `diagram2graph_dataset/` - diagram dataset labels and outputs, extraction notebooks, conversion scripts, and evaluation reports.
- `README.md` - repository entry point to be expanded as part of the documentation refactor.
- `docs/` - planning and repository structure documentation.

## dataset-Level Notes

### `soil_dataset/`

- `data/` holds figures and tables.
- `prompt_engineering/` and nested `prompt_text/` hold prompting assets.
- `extract_rdf_method/` and `extract_all_rdfs/` hold extraction notebooks and outputs.
- `build_kg/` and `executability/` hold KG outputs and RDF validation artifacts.
- `evaluation/` holds RAGAS and other evaluation work.
- Reusable core areas are `data/`, `prompt_engineering/prompt_text/`, and `extract_all_rdfs/`.
- `executability/`, `evaluation/`, and `extract_rdf_method/` should be treated as legacy or staging areas.

### `vistext/`

- `prompt_text/` holds prompt variants.
- `data/json2ttl/` contains the canonical converter code.
- `extract_rdf_ttl/` contains Gemini runners plus preserved output folders.
- Canonical `data/` layout is:
  - `vistext/data/images/`
  - `vistext/data/labels/`
  - `vistext/data/json2ttl/`
- Historical note: these canonical folders replaced the legacy `sub-image/`, `ground truth/`, and `1000_sub data/` inputs after conservative path migration.
- Reusable core areas are the canonical `data/` folders plus `prompt_text/` and the converter scripts.
- `extract_rdf_ttl/vistext_*_outputs/` are preserved historical outputs.

### `diagram2graph_dataset/`

- `json2ttl/script.py` is the main converter script.
- `extract_rdf_ttl/` and `extract_rdf_json/` hold extraction notebooks, prompts, and generated outputs.
- `evaluation/` contains F1, RAGAS, and report-generation assets.
- Reusable core areas are `data/`, `json2ttl/script.py`, `prompt_engineering/prompt_text/`, and `evaluation/report/plot_summaries.py`.
- `extract_rdf_ttl/`, `build_kg/`, and `prompt_engineering/*_outputs/` are preserved historical outputs.
- `extract_rdf_json/` and `evaluation/report/plot_summaries_claude.py` are legacy areas retained for provenance.

## Cross-Cutting Refactor Priorities

- Keep the lowercase underscore-separated dataset directories as the canonical repo layout.
- Preserve valuable outputs and reports unless they are confirmed safe to remove.
- Remove hardcoded secret leakage before broader cleanup.
- Replace absolute machine paths with configurable or relative paths.
- Deduplicate only confirmed duplicates.
- Label reusable vs historical areas before attempting more structural cleanup.
- Review and align repository guidance files with the final refactored layout.
