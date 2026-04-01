# AGENTS.md

## Purpose
This file guides coding agents working in `multimodal-KGC`.
It summarizes repository structure, runnable commands, and code conventions for this checkout.
It folds in the repo-specific guidance from `CLAUDE.md`.

## Instruction Precedence
1. Follow direct user instructions first.
2. Then follow this file.
3. Then follow `CLAUDE.md`.
4. Then follow the local style of the file you edit.

## Rules Files Found In This Checkout
- `CLAUDE.md` exists and contains repo-specific guidance.
- No `.cursorrules` file was found.
- No `.cursor/rules/` directory was found.
- `.github/copilot-instructions.md` exists and should stay aligned with this file.

## Repository Overview
- This is a research repo for extracting RDF/Turtle knowledge from charts, tables, and diagrams with multimodal LLMs.
- Work is organized by dataset folder, not as a single Python package.
- Most experimentation and evaluation happen in Jupyter notebooks.
- Python scripts are mainly converters, evaluators, and data-fixing utilities.
- Canonical dataset paths now use lowercase underscore-separated names.
- There is no formal build system, lockfile, linter config, or unit test suite.

## Main Areas
- `soil_dataset/`: soil-health tables and figures; `she:` ontology work.
- `vistext/`: chart dataset; canonical local inputs now live in `vistext/data/images/` and `vistext/data/labels/`, with JSON-to-TTL conversion and evaluation alongside preserved historical outputs.
- `diagram2graph_dataset/`: flowchart/process diagrams; `d2g:` ontology and F1 evaluation.

## Canonical And Legacy Boundaries
- Shared reusable helpers live under `common/`.
- Treat `soil_dataset/data/`, `soil_dataset/prompt_engineering/prompt_text/`, and `soil_dataset/extract_all_rdfs/` as Soil canonical areas.
- Treat `vistext/data/images/`, `vistext/data/labels/`, and `vistext/data/json2ttl/` as vistext canonical areas.
- Treat `diagram2graph_dataset/data/`, `diagram2graph_dataset/json2ttl/`, and `diagram2graph_dataset/prompt_engineering/prompt_text/` as diagram2graph canonical areas.
- Treat preserved outputs and evaluation or staging packets as historical or compatibility areas unless a task explicitly migrates them.

## Common dataset Flow
1. `data/` for raw inputs and labels.
2. `prompt_engineering/` or `prompt_text/` for prompt design.
3. `extract_rdf_ttl/`, `extract_rdf_json/`, or `extract_all_rdfs/` for model outputs.
4. `json2ttl/` or `build_kg/` for KG generation.
5. `evaluation/` for F1, RAGAS, and notebook analysis.

## Environment
- Use `python` in this Windows checkout; on Unix-like environments `python3` is typically equivalent.
- A top-level `requirements.txt` now exists for common notebook and script dependencies, but there is still no lockfile or packaging metadata.
- Common dependencies mentioned in code and notebooks include `openai`, `google-generativeai`, `rdflib`, `ragas`, `pandas`, `numpy`, and `matplotlib`.
- The top-level `config` file is gitignored and used for API keys; never commit it.

## Working In This Repo
- Prefer minimal, local edits over broad cleanup.
- Preserve the canonical lowercase underscore-separated dataset paths unless the user asks for another naming refactor.
- Treat notebooks as first-class project assets; do not replace notebook workflows with scripts unless asked.
- Be careful with generated TTL/JSON data and label files; large directories are part of the working dataset.
- If you touch notebooks, mention whether outputs were preserved, cleared, or changed.

## Build, Lint, And Test
- Build: no official build command exists.
- Lint/format/type-check: no canonical repo-wide commands were found for `ruff`, `black`, `isort`, `flake8`, `pylint`, `mypy`, or `pyright`.
- Tests: no `pytest`, `unittest`, `tox`, or `nox` setup was found.
- There is no true single-test command because there is no unit test suite.
- When a user asks for "the test", use a verified smoke test on one representative input file.

## Verified Smoke Tests
These commands were run successfully in this checkout and are the closest equivalent to a single test:

```bash
python "vistext/data/json2ttl/json_to_ttl_converter_v2.py" \
  "vistext/data/labels/1046.json" \
  -o ".tmp/1046-v2.ttl"

python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" \
  --dry-run --sample-mode ids --ids 1046

python "diagram2graph_dataset/json2ttl/script.py" \
  "diagram2graph_dataset/data/labels/9.json" \
  ".tmp/d2g-9.ttl"
```

## Main CLI Commands
Use `--help` for full flags:

```bash
python "vistext/data/json2ttl/json_to_ttl_converter_v2.py" --help
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --help
python "diagram2graph_dataset/json2ttl/script.py" --help
```

Common command forms:

```bash
python "vistext/data/json2ttl/json_to_ttl_converter_v2.py" <input> [-o <output>]
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" [--sample-mode {all,random,ids}] [--sample-count N] [--ids ...]
python "diagram2graph_dataset/json2ttl/script.py" <input_path> <output_path> [--base IRI] [--recursive] [--pattern "*.json"]
python "diagram2graph_dataset/evaluation/f1_score/evaluate_metrics.py"
python "diagram2graph_dataset/evaluation/f1_score/evaluate_node_edge_sep.py"
```

## Command Caveats
- `diagram2graph_dataset/evaluation/f1_score/evaluate_metrics.py` has been parameterized, but downstream inputs still need to exist locally.
- `diagram2graph_dataset/evaluation/f1_score/evaluate_node_edge_sep.py` has been parameterized, but downstream inputs still need to exist locally.
- `diagram2graph_dataset/evaluation/report/plot_summaries.py` has been parameterized; `plot_summaries_claude.py` remains a legacy script with older assumptions.
- The Gemini runners under `vistext/extract_rdf_ttl/` require a local `config` file with a valid `gemini_api_key`.
- Most evaluation beyond these scripts happens in notebooks.

## If A User Asks For Build, Lint, Or Test Commands
- Say explicitly that no official build command exists.
- Say explicitly that no official lint command exists.
- Say explicitly that no unit test framework exists.
- Offer one of the verified smoke-test commands instead of inventing a `pytest` command.

## Code Style
- Use Python 3 style with 4-space indentation.
- Preserve the script-oriented style of the surrounding file.
- Prefer small helper functions and straightforward data transformations over heavy abstraction.
- Keep existing module and function docstrings when a file already uses them.
- Keep comments sparse; add them only when the logic is not obvious.
- Use readable line breaks rather than dense one-liners.
- In new or heavily edited code, prefer double quotes unless the surrounding file clearly uses single quotes.

## Imports
- Keep imports at the top of the file unless there is a strong CLI-only reason not to.
- Group standard library imports together.
- Use explicit imports rather than wildcard imports.
- In new utilities, prefer `from pathlib import Path` when practical.
- If an older file already uses `os.path`, match nearby style unless the change naturally converts the whole block.

## Types And Naming
- Add type hints to new functions when practical.
- Match the local file's typing style.
- `diagram2graph_dataset/json2ttl/script.py` uses built-in generics like `list[str]`.
- `vistext` converters use `typing.List`, `Dict`, `Tuple`, and `Optional`.
- `from __future__ import annotations` and dataclasses are good defaults for new standalone scripts when they fit nearby style.
- Use `snake_case` for functions, variables, and helpers.
- Use `PascalCase` for classes and dataclasses.
- Preserve ontology and dataset-specific field names such as `chart`, `d2g`, `caption_L1`, `type_of_node`, and `relationship_type`.

## CLI, Files, And Error Handling
- Prefer `argparse` for new command-line scripts.
- Prefer a `main() -> int` entry point and `raise SystemExit(main())` for new scripts.
- Validate input paths early and return non-zero exit codes on fatal user errors.
- Use `print(..., file=sys.stderr)` for fatal CLI errors when appropriate.
- When batch-processing many files, continue per-file where reasonable and print the failing file clearly.
- Catch broad `Exception` only around outer CLI boundaries or batch item processing.
- Open text files with `encoding="utf-8"`.
- Use `json.load` and `json.dump`; when editing human-facing JSON, prefer `indent=4`.
- Keep TTL prefixes, ordering, and schema shape stable if the existing script already does so.
- Do not silently change base URIs, ontology namespaces, or output schema details without clear user intent.

## Validation Guidance
- Do not claim "all tests pass" for this repo; there is no unified test suite.
- State exactly what you validated.
- Good evidence here is usually one sample conversion command, one evaluation script run, one notebook workflow check, or one manual output comparison.
- If you change a script with hardcoded paths, either parameterize it or say exactly what path edits are still required.

## Known Pitfalls
- Some historical docs or notebook text may still mention pre-refactor paths from earlier layouts.
- Several scripts use absolute filesystem paths from another machine.
- Notebook workflows may assume Colab, Kaggle, or local-machine paths.
- Notebook outputs may contain secrets, stale execution state, or machine-specific paths.
- This repo mixes polished converters, ad hoc research scripts, notebooks, and data artifacts; not everything is production-hardened.

## Bottom Line
Treat this repository as a notebook-first, dataset-centric research codebase.
Use `python`, prefer canonical lowercase underscore paths, avoid assuming a test framework exists, and validate changes with small real-data smoke tests.
