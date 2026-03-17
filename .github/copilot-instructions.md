# Copilot instructions for `multimodal-KGC`

## Build, test, and lint commands

This repository does **not** use a centralized build system, linter, or automated test suite.
Validation is done by running dataset-specific scripts and representative smoke tests.

Use quoted paths because directory names contain spaces.

### Representative smoke tests

```bash
# VisText v1 converter
python3 "VisText/Data/Json2ttl/json_to_ttl_converter.py" \
  "VisText/Data/labels/1046.json" \
  -o "/tmp/1046-v1.ttl"

# VisText v2 converter
python3 "VisText/Evaluation/json_to_ttl_converter_v2.py" \
  "VisText/Data/labels/1046.json" \
  -o "/tmp/1046-v2.ttl"

# diagram2graph converter
python3 "diagram2graph Dataset/JSON2ttl/Script.py" \
  "diagram2graph Dataset/Data/labels/9.json" \
  "/tmp/d2g-9.ttl"
```

### Main CLI entry points

```bash
python3 "VisText/Data/Json2ttl/json_to_ttl_converter.py" --help
python3 "VisText/Evaluation/json_to_ttl_converter_v2.py" --help
python3 "diagram2graph Dataset/JSON2ttl/Script.py" --help
python3 "diagram2graph Dataset/Evaluating/F1_Score/evaluate_metrics.py" --help
python3 "diagram2graph Dataset/Evaluating/F1_Score/evaluate_Node Edge Sep.py" --help
```

## High-level architecture

The repo is organized as three dataset-first pipelines plus a small shared helper area:

- `Soil Dataset/` - soil tables and figures, prompt assets, extraction notebooks, KG build artifacts, executability checks, and evaluation packets.
- `VisText/` - chart-focused pipeline with canonical local inputs in `VisText/Data/images/` and `VisText/Data/labels/`, plus preserved historical output folders.
- `diagram2graph Dataset/` - flowchart and diagram pipeline with canonical images and labels under `Data/`, JSON-to-TTL conversion, and evaluation scripts.
- `common/` - shared helpers for repo-root discovery and secret loading from environment variables or the ignored top-level `config` file.

Each dataset broadly follows:

1. `Data/` for canonical source assets and labels
2. `prompt engineering/` or `Prompt Text/` for prompt assets and notebooks
3. `Extract ... RDF .../` for model outputs
4. `JSON2ttl/` or `Build KG/` for conversion and graph artifacts
5. `Evaluating/` for F1, RAGAS, bridge evaluation, and reports

Most execution in this repo is notebook-driven; standalone Python scripts are primarily converters, evaluators, and data-fixing utilities.

## Key codebase conventions

- Prompting variants are encoded in folder and file names (`FewShot`, `OneShot`, `ZeroShot`) and reused across extraction outputs, KG artifacts, and reports. Preserve this naming when adding artifacts.

- `VisText` intentionally has two TTL schemas:
  - `VisText/Data/Json2ttl/json_to_ttl_converter.py` uses the older schema-oriented output.
  - `VisText/Evaluation/json_to_ttl_converter_v2.py` uses the evaluation-oriented `chart:` schema.
  Choose the converter based on the downstream consumer and do not mix outputs casually.

- `diagram2graph Dataset/JSON2ttl/Script.py` is the canonical diagram2graph converter. The parameterized F1 and plotting scripts under `diagram2graph Dataset/Evaluating/` are preferred over older hardcoded variants.

- Preserve valuable generated outputs, KG render folders, evaluation CSVs, and notebooks by default. Treat them as research artifacts, not disposable build products.

- Some folders remain intentionally legacy for provenance or compatibility, including:
  - `diagram2graph Dataset/Extrcat RDF json/`
  - `diagram2graph Dataset/Evaluating/report/plot_summaries Clude.py`
  - selected Soil evaluation staging packets

- A top-level `config` file is used for API keys and is gitignored. Never commit key contents.

## Environment and dependency notes

- Use `python3`; do not assume `python` exists.
- A top-level `requirements.txt` exists for common notebook and script dependencies.
- There is still no lockfile, package metadata, or unified automated test suite.
- Some optional notebook-only dependencies may remain commented in `requirements.txt` because they are heavier or more version-sensitive.

## Practical cautions

- Paths with spaces are common; always quote them in shell commands.
- Some notebooks still contain Kaggle or Colab path examples as workflow context.
- `VisText/Evaluation/complete_data.py` is interactive and mutates files in place.
- `diagram2graph Dataset/Evaluating/report/plot_summaries Clude.py` is a legacy script and should not be treated as the preferred entry point.
