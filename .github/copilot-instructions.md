# Copilot instructions for `multimodal-KGC`

## Build, test, and lint commands

This repository does **not** use a centralized build system, linter, or automated test suite.
Validation is done by running dataset-specific scripts and representative smoke tests.

Use the canonical lowercase underscore-separated dataset paths in examples and code changes.

### Representative smoke tests

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

### Main CLI entry points

```bash
python "vistext/data/json2ttl/json_to_ttl_converter_v2.py" --help
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --help
python "diagram2graph_dataset/json2ttl/script.py" --help
python "diagram2graph_dataset/evaluating/f1_score/evaluate_metrics.py" --help
python "diagram2graph_dataset/evaluating/f1_score/evaluate_node_edge_sep.py" --help
```

## High-level architecture

The repo is organized as three dataset-first pipelines plus a small shared helper area:

- `soil_dataset/` - soil tables and figures, prompt assets, extraction notebooks, KG build artifacts, executability checks, and evaluation packets.
- `vistext/` - chart-focused pipeline with canonical local inputs in `vistext/data/images/` and `vistext/data/labels/`, plus preserved historical output folders.
- `diagram2graph_dataset/` - flowchart and diagram pipeline with canonical images and labels under `data/`, JSON-to-TTL conversion, and evaluation scripts.
- `common/` - shared helpers for repo-root discovery and secret loading from environment variables or the ignored top-level `config` file.

Each dataset broadly follows:

1. `data/` for canonical source assets and labels
2. `prompt_engineering/` or `prompt_text/` for prompt assets and notebooks
3. `extract_rdf_ttl/`, `extract_rdf_json/`, or `extract_all_rdfs/` for model outputs
4. `json2ttl/` or `build_kg/` for conversion and graph artifacts
5. `evaluating/` for F1, RAGAS, bridge evaluation, and reports

Most execution in this repo is notebook-driven; standalone Python scripts are primarily converters, evaluators, and data-fixing utilities.

## Key codebase conventions

- Prompting variants are encoded in folder and file names (`fewshot`, `oneshot`, `zeroshot`) and reused across extraction outputs, KG artifacts, and reports. Preserve this naming when adding artifacts.

- `vistext/data/json2ttl/json_to_ttl_converter_v2.py` is the canonical vistext converter in this checkout.
- The Gemini runners under `vistext/extract_rdf_ttl/` are the canonical vistext extraction entry points.

- `diagram2graph_dataset/json2ttl/script.py` is the canonical diagram2graph converter. The parameterized F1 and plotting scripts under `diagram2graph_dataset/evaluating/` are preferred over older hardcoded variants.

- Preserve valuable generated outputs, KG render folders, evaluation CSVs, and notebooks by default. Treat them as research artifacts, not disposable build products.

- Some folders remain intentionally legacy for provenance or compatibility, including:
  - `diagram2graph_dataset/extract_rdf_json/`
  - `diagram2graph_dataset/evaluating/report/plot_summaries_claude.py`
  - selected Soil evaluation staging packets

- A top-level `config` file is used for API keys and is gitignored. Never commit key contents.

## Environment and dependency notes

- Use `python` in this Windows checkout; on Unix-like systems `python3` is usually equivalent.
- A top-level `requirements.txt` exists for common notebook and script dependencies.
- There is still no lockfile, package metadata, or unified automated test suite.
- Some optional notebook-only dependencies may remain commented in `requirements.txt` because they are heavier or more version-sensitive.

## Practical cautions

- Paths with spaces are common; always quote them in shell commands.
- Some notebooks still contain Kaggle or Colab path examples as workflow context.
- `diagram2graph_dataset/evaluating/report/plot_summaries_claude.py` is a legacy script and should not be treated as the preferred entry point.
