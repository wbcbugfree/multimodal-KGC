# Repository Map

This map reflects the current worktree state and the approved conservative refactor direction.

## Root Layout

- `Soil Dataset/` - soil tables and figures, prompt engineering, extraction notebooks, KG building, executability checks, and evaluation outputs.
- `VisText/` - chart dataset prompts, data assets, JSON-to-TTL conversion, and evaluation scripts.
- `diagram2graph Dataset/` - diagram dataset labels and outputs, extraction notebooks, conversion scripts, and evaluation reports.
- `README.md` - repository entry point to be expanded as part of the documentation refactor.
- `docs/` - planning and repository structure documentation.

## Dataset-Level Notes

### `Soil Dataset/`

- `Data/` holds figures and tables.
- `prompt engineering/` and nested `Prompt Text/` hold prompting assets.
- `Extract RDF Method/` and `Extract All RDFs/` hold extraction notebooks and outputs.
- `Build KG/` and `Executability/` hold KG outputs and RDF validation artifacts.
- `Evaluating/` holds RAGAS and other evaluation work.
- Reusable core areas are `Data/`, `prompt engineering/Prompt Text/`, and `Extract All RDFs/`.
- `Executability/`, `Evaluating/`, and `Extract RDF Method/` should be treated as legacy or staging areas.

### `VisText/`

- `Prompt Text/` holds prompt variants.
- `Data/Json2ttl/` contains conversion code.
- `Evaluation/` contains conversion and evaluation utilities.
- Canonical `Data/` layout is:
  - `VisText/Data/images/`
  - `VisText/Data/labels/`
  - `VisText/Data/Json2ttl/`
- Historical note: these canonical folders replaced the legacy `sub-image/`, `ground truth/`, and `1000_sub data/` inputs after conservative path migration.
- Reusable core areas are the canonical `Data/` folders plus `Prompt Text/` and the converter scripts.
- `Extract RDF ttl/vistext_*_outputs/` and `Evaluation/gold_ttl_1000/` are preserved historical outputs.
- `Evaluation/complete_data.py` remains a manual staging utility.

### `diagram2graph Dataset/`

- `JSON2ttl/Script.py` is the main converter script.
- `Extract RDF ttl/` and `Extrcat RDF json/` hold extraction notebooks, prompts, and generated outputs.
- `Evaluating/` contains F1, RAGAS, and report-generation assets.
- Reusable core areas are `Data/`, `JSON2ttl/Script.py`, `prompt engineering/Prompt Text/`, and `Evaluating/report/plot_summaries.py`.
- `Extract RDF ttl/`, `Build KG/`, and `prompt engineering/*_outputs/` are preserved historical outputs.
- `Extrcat RDF json/` and `Evaluating/report/plot_summaries Clude.py` are legacy areas retained for provenance.

## Cross-Cutting Refactor Priorities

- Keep top-level dataset directories for now.
- Preserve valuable outputs and reports unless they are confirmed safe to remove.
- Remove hardcoded secret leakage before broader cleanup.
- Replace absolute machine paths with configurable or relative paths.
- Deduplicate only confirmed duplicates.
- Label reusable vs historical areas before attempting more structural cleanup.
- Review and align repository guidance files with the final refactored layout.
