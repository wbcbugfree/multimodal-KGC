# AGENTS.md

## Fast Orientation
- This is a dataset-first research repository (not a packaged Python project): no `pyproject.toml`, no lockfile, no CI workflow, no pre-commit config, and no repo-wide `pytest` suite.
- Most work is notebook-driven; standalone Python files are mainly converters/evaluation utilities.
- Treat generated TTL/CSV outputs as preserved research artifacts unless a task explicitly asks to regenerate or prune them.

## Instruction Sources In This Checkout
- Repo-local agent guidance also exists at `.github/copilot-instructions.md`.
- `CLAUDE.md`, `.cursorrules`, `.cursor/rules/`, and `opencode.json` are not present.

## Ownership And Real Entrypoints
- `vistext/`: chart pipeline. Canonical local inputs are `vistext/data/images/` and `vistext/data/labels/`.
- `vistext` converter entrypoint: `vistext/data/json2ttl/json_to_ttl_converter_v2.py`.
- `diagram2graph_dataset/`: diagram pipeline. Converter entrypoint: `diagram2graph_dataset/json2ttl/script.py`.
- `diagram2graph` evaluation CLIs: `diagram2graph_dataset/evaluation/f1_score/evaluate_metrics.py` and `diagram2graph_dataset/evaluation/f1_score/evaluate_node_edge_sep.py`.
- `soil_dataset/` is currently notebook/assets focused (no `.py` scripts found in this checkout).
- Shared path/secret helpers live in `common/paths.py` and `common/config.py`; prefer these over hardcoded machine paths.

## Commands Agents Should Use
Use `python3` in Linux/macOS sessions (`python` may be the right command in Windows shells).

- Single vistext conversion:
  `python3 "vistext/data/json2ttl/json_to_ttl_converter_v2.py" "vistext/data/labels/1046.json" -o ".tmp/1046-v2.ttl"`
- Single diagram2graph conversion:
  `python3 "diagram2graph_dataset/json2ttl/script.py" "diagram2graph_dataset/data/labels/9.json" ".tmp/d2g-9.ttl"`
- diagram2graph folder conversion (from repo script note):
  `python3 "diagram2graph_dataset/json2ttl/script.py" "diagram2graph_dataset/data/labels" "diagram2graph_dataset/json2ttl/out_folder"`
- Graph-level evaluation:
  `python3 "diagram2graph_dataset/evaluation/f1_score/evaluate_metrics.py" --output-csv ".tmp/metrics.csv"`
- Node/edge-separated evaluation:
  `python3 "diagram2graph_dataset/evaluation/f1_score/evaluate_node_edge_sep.py" --output-csv ".tmp/node-edge.csv"`
- VisText Gemini dry-run sampling:
  `python3 "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --dry-run --sample-mode ids --ids 1046`

## Dependency And Secrets Gotchas
- `requirements.txt` is minimal and does not include the Gemini SDK used by `vistext/extract_rdf_ttl/gemini_vistext_runner_core.py` (`from google import genai`); install that separately when running Gemini scripts.
- Secrets come from top-level gitignored `config` JSON and/or environment variables via `common/config.py`; environment values override config values.
- `get_api_key("gemini_api_key")` accepts multiple key-name variants, but setting `GEMINI_API_KEY` is the clearest option.

## Architecture Details Easy To Miss
- `vistext/extract_rdf_ttl/gemini_vistext_*.py` are thin strategy wrappers; core sampling, API calls, Turtle validation, and manifest upserts are centralized in `vistext/extract_rdf_ttl/gemini_vistext_runner_core.py`.
- Dynamic one-shot (`gemini_vistext_oneshot_dynamic.py`) classifies chart type first via `vistext/prompt_text/dynamic_oneshot/categorize_viz.py`, adding an extra Gemini call per image.
- Runner outputs are status-tracked in `manifest.json` under each output directory.

## Verified Quirks (Do Not "Fix" By Accident)
- `diagram2graph_dataset/json2ttl/script.py` currently uses `re.search(r"(\\d+)", input_json_path.stem)` for diagram id inference; this falls back to `diagram` in single-file runs, producing IRIs like `/diagram/diagram/...`.
- Prefer `diagram2graph_dataset/evaluation/report/plot_summaries.py`; `plot_summaries_claude.py` is legacy and hardcodes `/mnt/data`.
- `README.md` references `docs/` and dataset `readme_internal.md` files that are not present in this checkout; trust executable scripts and actual filesystem state when prose and code disagree.

## Validation Expectations For Agent Reports
- Do not claim "all tests pass"; there is no unified test harness.
- State exactly which command(s) were run and what they verified.
- For most changes, run the smallest relevant real-data smoke command above.
