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
- Directory names contain spaces; always quote shell paths.
- There is no formal build system, lockfile, linter config, or unit test suite.

## Main Areas
- `Soil Dataset/`: soil-health tables and figures; `she:` ontology work.
- `VisText/`: chart dataset; canonical local inputs now live in `VisText/Data/images/` and `VisText/Data/labels/`, with JSON-to-TTL conversion and evaluation alongside preserved historical outputs.
- `diagram2graph Dataset/`: flowchart/process diagrams; `d2g:` ontology and F1 evaluation.

## Canonical And Legacy Boundaries
- Shared reusable helpers live under `common/`.
- Treat `Soil Dataset/Data/`, `Soil Dataset/prompt engineering/Prompt Text/`, and `Soil Dataset/Extract All RDFs/` as Soil canonical areas.
- Treat `VisText/Data/images/`, `VisText/Data/labels/`, and `VisText/Data/Json2ttl/` as VisText canonical areas.
- Treat `diagram2graph Dataset/Data/`, `diagram2graph Dataset/JSON2ttl/`, and `diagram2graph Dataset/prompt engineering/Prompt Text/` as diagram2graph canonical areas.
- Treat preserved outputs and evaluation or staging packets as historical or compatibility areas unless a task explicitly migrates them.

## Common Dataset Flow
1. `Data/` for raw inputs and labels.
2. `prompt engineering/` or `Prompt Text/` for prompt design.
3. `Extract RDF .../` for model outputs.
4. `JSON2ttl/` or `Build KG/` for KG generation.
5. `Evaluating/` for F1, RAGAS, and notebook analysis.

## Environment
- Use `python3`; `python` is not available in this environment.
- A top-level `requirements.txt` now exists for common notebook and script dependencies, but there is still no lockfile or packaging metadata.
- Common dependencies mentioned in code and notebooks include `openai`, `google-generativeai`, `rdflib`, `ragas`, `pandas`, `numpy`, and `matplotlib`.
- The top-level `config` file is gitignored and used for API keys; never commit it.

## Working In This Repo
- Prefer minimal, local edits over broad cleanup.
- Preserve dataset-specific folder names, typos, and workflow-specific naming unless the user asks to rename them.
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
python3 "VisText/Data/Json2ttl/json_to_ttl_converter.py" \
  "VisText/Data/labels/1046.json" \
  -o "/tmp/1046-v1.ttl"

python3 "VisText/Evaluation/json_to_ttl_converter_v2.py" \
  "VisText/Data/labels/1046.json" \
  -o "/tmp/1046-v2.ttl"

python3 "diagram2graph Dataset/JSON2ttl/Script.py" \
  "diagram2graph Dataset/Data/labels/9.json" \
  "/tmp/d2g-9.ttl"
```

## Main CLI Commands
Use `--help` for full flags:

```bash
python3 "VisText/Data/Json2ttl/json_to_ttl_converter.py" --help
python3 "VisText/Evaluation/json_to_ttl_converter_v2.py" --help
python3 "diagram2graph Dataset/JSON2ttl/Script.py" --help
```

Common command forms:

```bash
python3 "VisText/Data/Json2ttl/json_to_ttl_converter.py" <input> [-o <output>]
python3 "VisText/Evaluation/json_to_ttl_converter_v2.py" <input> [-o <output>] [--sort]
python3 "diagram2graph Dataset/JSON2ttl/Script.py" <input_path> <output_path> [--base IRI] [--recursive] [--pattern "*.json"]
python3 "diagram2graph Dataset/Evaluating/F1_Score/evaluate_metrics.py"
python3 "diagram2graph Dataset/Evaluating/F1_Score/evaluate_Node Edge Sep.py"
python3 "VisText/Evaluation/complete_data.py"
```

## Command Caveats
- `diagram2graph Dataset/Evaluating/F1_Score/evaluate_metrics.py` has been parameterized, but downstream inputs still need to exist locally.
- `diagram2graph Dataset/Evaluating/F1_Score/evaluate_Node Edge Sep.py` has been parameterized, but downstream inputs still need to exist locally.
- `diagram2graph Dataset/Evaluating/report/plot_summaries.py` has been parameterized; `plot_summaries Clude.py` remains a legacy script with older assumptions.
- `VisText/Evaluation/complete_data.py` is interactive and mutates files in place.
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
- `diagram2graph Dataset/JSON2ttl/Script.py` uses built-in generics like `list[str]`.
- `VisText` converters use `typing.List`, `Dict`, `Tuple`, and `Optional`.
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
- Paths with spaces are common; unquoted shell commands will break.
- Several scripts use absolute filesystem paths from another machine.
- Notebook workflows may assume Colab, Kaggle, or local-machine paths.
- Notebook outputs may contain secrets, stale execution state, or machine-specific paths.
- `VisText/Evaluation/complete_data.py` prompts for confirmation and rewrites JSON files.
- This repo mixes polished converters, ad hoc research scripts, notebooks, and data artifacts; not everything is production-hardened.

## Bottom Line
Treat this repository as a notebook-first, dataset-centric research codebase.
Use `python3`, quote paths, avoid assuming a test framework exists, and validate changes with small real-data smoke tests.
