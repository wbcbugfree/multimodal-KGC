# multimodal-KGC

Multimodal knowledge-graph construction workflows for three academic datasets: soil-health figures and tables, chart images, and process or flowchart diagrams.

This repository is a notebook-first research artifact. It preserves extraction outputs, evaluation notebooks, and generated RDF or Turtle files from prior experiments while making the local layout safer and easier to understand.

## What this repository contains

- `soil_dataset/` - soil-health figures and tables, prompt assets, extraction notebooks, KG-building notebooks, executability checks, and evaluation packets.
- `vistext/` - chart dataset inputs, prompt text, JSON-to-TTL converters, preserved TTL outputs, and evaluation utilities.
- `diagram2graph_dataset/` - diagram images and labels, extraction notebooks, KG outputs, JSON-to-TTL conversion, and F1 or report-generation scripts.
- `common/` - shared helpers for repo-root discovery and secret loading from environment variables or the ignored top-level `config` file.
- `docs/` - refactor notes, repository maps, and internal asset-boundary documentation.

## Current repository status

This checkout has been refactored for publication-friendly naming.

- Top-level dataset folders now use lowercase underscore-separated names.
- Valuable generated outputs are preserved by default.
- Confirmed duplicate mirrors have been removed where references were migrated safely.
- Secret-bearing notebooks were updated to load keys from environment variables or the local ignored `config` file.
- Legacy folders and historical outputs are still present where they help preserve provenance or notebook compatibility.

## Quick reviewer guide

If you are reviewing the repository for paper evaluation, start here:

- Read this `README.md` for the canonical layout and the validation commands.
- Check `docs/repo_map.md` for a concise map of the current repository structure.
- Check `docs/task7-asset-boundaries.md` for the reusable-vs-historical boundary decisions.
- Inspect one preserved output directory such as `vistext/extract_rdf_ttl/vistext_zeroshot_outputs/` or `diagram2graph_dataset/build_kg/` to see retained research artifacts.
- Use the smoke-test commands in this README instead of looking for a nonexistent repo-wide `pytest` suite.

Current canonical input counts in this checkout:

- soil_dataset: 15 figure images and 56 table images under `soil_dataset/data/`
- vistext:
  - `vistext/data/train/`: 7057 images, 7057 label JSON files, 7050 Turtle files
  - `vistext/data/eval/`: 883 images, 883 label JSON files, 882 Turtle files
  - `vistext/data/test/`: 882 images, 882 label JSON files, 882 Turtle files
- diagram2graph_dataset: 219 diagram images and 219 label JSON files under `diagram2graph_dataset/data/`

## Canonical inputs and reusable entry points

### soil_dataset

- Canonical raw inputs:
  - `soil_dataset/data/figures/`
  - `soil_dataset/data/tables/`
- Canonical extraction outputs:
  - `soil_dataset/extract_all_rdfs/`
- Reusable prompt assets:
  - `soil_dataset/prompt_engineering/prompt_text/`
- Legacy or staging areas retained for workflow compatibility:
  - `soil_dataset/executability/`
  - `soil_dataset/evaluation/`
  - `soil_dataset/extract_rdf_method/`

### vistext

- Canonical local inputs:
  - `vistext/data/train/`
  - `vistext/data/eval/`
  - `vistext/data/test/`
- Reusable converter entry points:
  - `vistext/data/json2ttl_converter.py`
  - `vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py`
- Reusable prompt assets:
  - `vistext/prompt_engineering/`
- Preserved historical outputs:
  - `vistext/extract_rdf_ttl/vistext_zeroshot_outputs/`
  - `vistext/extract_rdf_ttl/vistext_oneshot_static_outputs/`
  - `vistext/extract_rdf_ttl/vistext_oneshot_dynamic_outputs/`
  - `vistext/extract_rdf_ttl/vistext_fewshot_outputs/`

### diagram2graph_dataset

- Canonical inputs:
  - `diagram2graph_dataset/data/diagram2graph/`
  - `diagram2graph_dataset/data/labels/`
- Reusable converter and report scripts:
  - `diagram2graph_dataset/json2ttl/script.py`
  - `diagram2graph_dataset/evaluation/report/plot_summaries.py`
- Reusable prompt assets:
  - `diagram2graph_dataset/prompt_engineering/prompt_text/`
- Preserved historical outputs:
  - `diagram2graph_dataset/extract_rdf_ttl/`
  - `diagram2graph_dataset/build_kg/`
  - `diagram2graph_dataset/prompt_engineering/zeroshot_outputs/`
  - `diagram2graph_dataset/prompt_engineering/oneshot_outputs/`
  - `diagram2graph_dataset/prompt_engineering/fewshot_outputs/`

## dataset workflow overview

Although each dataset evolved independently, the repository generally follows this pattern:

1. `data/` contains source images, tables, figures, or labels.
   VisText now uses split-specific subfolders with `images/`, `labels/`, and `turtle/` under `train/`, `eval/`, and `test/`.
2. `prompt_engineering/` or `prompt_text/` contains prompt variants or notebook-driven prompt workflows.
3. `Extract RDF .../` stores multimodal model outputs.
4. `json2ttl/` or `build_kg/` converts outputs into knowledge-graph artifacts.
5. `evaluation/` stores F1 analysis, RAGAS notebooks, bridge evaluation notebooks, and summary plots.

## Local environment and secrets

- Use `python` in this Windows checkout; on Unix-like environments `python3` is typically equivalent.
- There is no lockfile, `pyproject.toml`, or centralized test suite.
- The ignored top-level `config` file is the local place for API keys.
- Environment variables override values in `config`.
- Shared secret loading helpers live in `common/config.py`.

Example `config` shape:

```json
{
  "gemini_api_key": "..."
}
```

## Installation

This repository is notebook-heavy and does not define a reproducible environment yet, but a practical local setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some notebooks may also require optional heavy packages that are intentionally left commented in `requirements.txt`.

## Representative smoke tests

There is no official repo-wide test command. Use small representative conversions instead.

### vistext converter

```bash
python "vistext/data/json2ttl_converter.py" \
  "vistext/data/test/labels/1046.json" \
  -o ".tmp/1046.ttl"
```

### vistext Gemini runner dry-run

```bash
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" \
  --dry-run --sample-mode ids --ids 1046
```

### diagram2graph converter

```bash
python "diagram2graph_dataset/json2ttl/script.py" \
  "diagram2graph_dataset/data/labels/9.json" \
  ".tmp/d2g-9.ttl"
```

## Important caveats

- The canonical repo layout no longer uses spaces in dataset paths.
- Several notebooks still contain Kaggle or Colab path examples as historical workflow context.
- `diagram2graph_dataset/evaluation/report/plot_summaries_claude.py` is retained as a legacy script, not a recommended entry point.

## Repository guidance for contributors and agents

- Internal refactor notes live in `docs/refactor_inventory.md`, `docs/repo_map.md`, and `docs/task7-asset-boundaries.md`.
- dataset-level internal notes live in:
  - `soil_dataset/readme_internal.md`
  - `vistext/readme_internal.md`
  - `diagram2graph_dataset/readme_internal.md`
- Prefer canonical paths for new work and treat legacy staging folders as compatibility areas unless you verify references first.

## License

This repository is licensed under the MIT License. See `LICENSE`.
