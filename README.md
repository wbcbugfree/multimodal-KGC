# multimodal-KGC

Multimodal knowledge-graph construction workflows for three academic datasets: soil-health figures and tables, chart images, and process or flowchart diagrams.

This repository is a notebook-first research artifact. It preserves extraction outputs, evaluation notebooks, and generated RDF or Turtle files from prior experiments while making the local layout safer and easier to understand.

## What this repository contains

- `Soil Dataset/` - soil-health figures and tables, prompt assets, extraction notebooks, KG-building notebooks, executability checks, and evaluation packets.
- `VisText/` - chart dataset inputs, prompt text, JSON-to-TTL converters, preserved TTL outputs, and evaluation utilities.
- `diagram2graph Dataset/` - diagram images and labels, extraction notebooks, KG outputs, JSON-to-TTL conversion, and F1 or report-generation scripts.
- `common/` - shared helpers for repo-root discovery and secret loading from environment variables or the ignored top-level `config` file.
- `docs/` - refactor notes, repository maps, and internal asset-boundary documentation.

## Current repository status

This checkout has been conservatively refactored.

- Top-level dataset folders stay unchanged.
- Valuable generated outputs are preserved by default.
- Confirmed duplicate mirrors have been removed where references were migrated safely.
- Secret-bearing notebooks were updated to load keys from environment variables or the local ignored `config` file.
- Legacy folders and historical outputs are still present where they help preserve provenance or notebook compatibility.

## Quick reviewer guide

If you are reviewing the repository for paper evaluation, start here:

- Read this `README.md` for the canonical layout and the validation commands.
- Check `docs/repo_map.md` for a concise map of the current repository structure.
- Check `docs/task7-asset-boundaries.md` for the reusable-vs-historical boundary decisions.
- Inspect one preserved output directory such as `VisText/Extract RDF ttl/vistext_Zeroshot_outputs/` or `diagram2graph Dataset/Build KG/` to see retained research artifacts.
- Use the smoke-test commands in this README instead of looking for a nonexistent repo-wide `pytest` suite.

Current canonical input counts in this checkout:

- Soil Dataset: 15 figure images and 56 table images under `Soil Dataset/Data/`
- VisText: 882 images and 882 label JSON files under `VisText/Data/`
- diagram2graph Dataset: 219 diagram images and 219 label JSON files under `diagram2graph Dataset/Data/`

## Canonical inputs and reusable entry points

### Soil Dataset

- Canonical raw inputs:
  - `Soil Dataset/Data/figures/`
  - `Soil Dataset/Data/tables/`
- Canonical extraction outputs:
  - `Soil Dataset/Extract All RDFs/`
- Reusable prompt assets:
  - `Soil Dataset/prompt engineering/Prompt Text/`
- Legacy or staging areas retained for workflow compatibility:
  - `Soil Dataset/Executability/`
  - `Soil Dataset/Evaluating/`
  - `Soil Dataset/Extract RDF Method/`

### VisText

- Canonical local inputs:
  - `VisText/Data/images/`
  - `VisText/Data/labels/`
- Reusable converter entry points:
  - `VisText/Data/Json2ttl/json_to_ttl_converter.py`
  - `VisText/Evaluation/json_to_ttl_converter_v2.py`
- Reusable prompt assets:
  - `VisText/Prompt Text/`
- Preserved historical outputs:
  - `VisText/Extract RDF ttl/vistext_Zeroshot_outputs/`
  - `VisText/Extract RDF ttl/vistext_Oneshot_outputs/`
  - `VisText/Extract RDF ttl/vistext_Fewshot_outputs/`

### diagram2graph Dataset

- Canonical inputs:
  - `diagram2graph Dataset/Data/diagram2graph/`
  - `diagram2graph Dataset/Data/labels/`
- Reusable converter and report scripts:
  - `diagram2graph Dataset/JSON2ttl/Script.py`
  - `diagram2graph Dataset/Evaluating/report/plot_summaries.py`
- Reusable prompt assets:
  - `diagram2graph Dataset/prompt engineering/Prompt Text/`
- Preserved historical outputs:
  - `diagram2graph Dataset/Extract RDF ttl/`
  - `diagram2graph Dataset/Build KG/`
  - `diagram2graph Dataset/prompt engineering/ZeroShot_outputs/`
  - `diagram2graph Dataset/prompt engineering/OneShot_outputs/`
  - `diagram2graph Dataset/prompt engineering/FewShot_outputs/`

## Dataset workflow overview

Although each dataset evolved independently, the repository generally follows this pattern:

1. `Data/` contains source images, tables, figures, or labels.
2. `prompt engineering/` or `Prompt Text/` contains prompt variants or notebook-driven prompt workflows.
3. `Extract RDF .../` stores multimodal model outputs.
4. `JSON2ttl/` or `Build KG/` converts outputs into knowledge-graph artifacts.
5. `Evaluating/` stores F1 analysis, RAGAS notebooks, bridge evaluation notebooks, and summary plots.

## Local environment and secrets

- Use `python3`. This environment does not provide `python`.
- There is no lockfile, `pyproject.toml`, or centralized test suite.
- The ignored top-level `config` file is the local place for API keys.
- Environment variables override values in `config`.
- Shared secret loading helpers live in `common/config.py`.

Example `config` shape:

```json
{
  "gemini_api_key": "...",
  "openai_api_key": "...",
  "avalai_api_key": "..."
}
```

## Installation

This repository is notebook-heavy and does not define a reproducible environment yet, but a practical local setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some notebooks may also require optional heavy packages that are intentionally left commented in `requirements.txt`.

## Representative smoke tests

There is no official repo-wide test command. Use small representative conversions instead.

### VisText v1 converter

```bash
python3 "VisText/Data/Json2ttl/json_to_ttl_converter.py" \
  "VisText/Data/labels/1046.json" \
  -o "/tmp/1046-v1.ttl"
```

### VisText v2 converter

```bash
python3 "VisText/Evaluation/json_to_ttl_converter_v2.py" \
  "VisText/Data/labels/1046.json" \
  -o "/tmp/1046-v2.ttl"
```

### diagram2graph converter

```bash
python3 "diagram2graph Dataset/JSON2ttl/Script.py" \
  "diagram2graph Dataset/Data/labels/9.json" \
  "/tmp/d2g-9.ttl"
```

## Important caveats

- Quote paths in shell commands because many directories contain spaces.
- Several notebooks still contain Kaggle or Colab path examples as historical workflow context.
- `VisText/Extract RDF ttl/README.md` explains why preserved `manifest.json` files still mention older Kaggle-era input paths.
- `VisText/Evaluation/complete_data.py` is interactive and mutates files in place.
- `diagram2graph Dataset/Evaluating/report/plot_summaries Clude.py` is retained as a legacy script, not a recommended entry point.

## Repository guidance for contributors and agents

- Internal refactor notes live in `docs/refactor_inventory.md`, `docs/repo_map.md`, and `docs/task7-asset-boundaries.md`.
- Dataset-level internal notes live in:
  - `Soil Dataset/README-internal.md`
  - `VisText/README-internal.md`
  - `diagram2graph Dataset/README-internal.md`
- Prefer canonical paths for new work and treat legacy staging folders as compatibility areas unless you verify references first.

## License

This repository is licensed under the MIT License. See `LICENSE`.
