# Copilot instructions for `multimodal-KGC`

## Build, test, and lint commands

This repository does **not** use a centralized build system, linter, or automated test suite.
Validation is done by running dataset-specific scripts and representative smoke tests.

Use the canonical lowercase underscore-separated dataset paths in examples and code changes.

### Representative single-run smoke tests

```bash
# vistext converter
python "vistext/data/json2ttl/json_to_ttl_converter_v2.py" \
  "vistext/data/labels/1046.json" \
  -o ".tmp/1046-v2.ttl"

# vistext Gemini runner dry-run
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" \
  --dry-run --sample-mode ids --ids 1046

# diagram2graph converter
python "diagram2graph_dataset/json2ttl/script.py" \
  "diagram2graph_dataset/data/labels/9.json" \
  ".tmp/d2g-9.ttl"
```

### Evaluation entry points (folder-to-folder checks)

```bash
python "diagram2graph_dataset/evaluation/f1_score/evaluate_metrics.py"
python "diagram2graph_dataset/evaluation/f1_score/evaluate_node_edge_sep.py"
```

### Main CLI entry points

```bash
python "vistext/data/json2ttl/json_to_ttl_converter_v2.py" --help
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --help
python "diagram2graph_dataset/json2ttl/script.py" --help
python "diagram2graph_dataset/evaluation/f1_score/evaluate_metrics.py" --help
python "diagram2graph_dataset/evaluation/f1_score/evaluate_node_edge_sep.py" --help
```

## High-level architecture

The repo is organized as three dataset-first pipelines plus a small shared helper area:

- `soil_dataset/` - soil tables and figures, prompt assets, extraction notebooks, KG build artifacts, executability checks, and evaluation packets.
- `vistext/` - chart-focused pipeline with canonical local inputs in `vistext/data/images/` and `vistext/data/labels/`, plus preserved historical output folders.
- `diagram2graph_dataset/` - flowchart and diagram pipeline with canonical images and labels under `data/`, JSON-to-TTL conversion, and F1 evaluation scripts.
- `common/` - shared helpers for repo-root discovery and secret loading (top-level ignored `config` + env overrides).

Each dataset broadly follows:

1. `data/` for canonical source assets and labels
2. `prompt_engineering/` or `prompt_text/` for prompt assets and notebooks
3. `extract_rdf_ttl/`, `extract_rdf_json/`, or `extract_all_rdfs/` for model outputs
4. `json2ttl/` or `build_kg/` for conversion and graph artifacts
5. `evaluation/` for F1, RAGAS, bridge evaluation, and reports

Most execution in this repo is notebook-driven; standalone Python scripts are primarily converters, evaluators, and data-fixing utilities.

Across scripts, there are two shared architecture patterns worth preserving:

1. VisText Gemini runners (`vistext/extract_rdf_ttl/gemini_vistext_*.py`) are thin strategy wrappers over `gemini_vistext_runner_core.py`. The core handles sampling, request execution, Turtle validation, and manifest upserts.
2. Path and secret resolution is centralized in `common/paths.py` and `common/config.py`; scripts should prefer these helpers over hardcoded machine-local paths.

## Key codebase conventions

- Prompting variants are encoded in folder and file names (`fewshot`, `oneshot`, `zeroshot`) and reused across extraction outputs, KG artifacts, and reports. Preserve this naming when adding artifacts.

- `vistext/data/json2ttl/json_to_ttl_converter_v2.py` is the canonical vistext converter in this checkout.
- The Gemini runners under `vistext/extract_rdf_ttl/` are the canonical vistext extraction entry points.

- `diagram2graph_dataset/json2ttl/script.py` is the canonical diagram2graph converter. The parameterized F1 and plotting scripts under `diagram2graph_dataset/evaluation/` are preferred over older hardcoded variants.
- `diagram2graph_dataset/evaluation/report/plot_summaries_claude.py` is retained for legacy provenance and should not be treated as the primary entry point.

- Preserve valuable generated outputs, KG render folders, evaluation CSVs, and notebooks by default. Treat them as research artifacts, not disposable build products.

- Some folders remain intentionally legacy for provenance or compatibility, including:
  - `diagram2graph_dataset/extract_rdf_json/`
  - selected Soil evaluation staging packets

- Secrets are loaded from env vars and/or the top-level gitignored `config` file; env values override file values (`common/config.py`).
- Keep output schema expectations stable when changing converters:
  - VisText converter V2 intentionally maps `datatable` + `caption_L1` into `chart:` Turtle output.
  - diagram2graph conversion/evaluation depends on node/edge field names such as `type_of_node`, `type_of_edge`, and `relationship_type`.
