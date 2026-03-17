# Refactor Inventory

## Scope And Constraints

- Approved conservative refactor for an academic, dataset-first repository.
- Keep top-level dataset directories for now: `Soil Dataset/`, `VisText/`, `diagram2graph Dataset/`.
- Preserve valuable outputs by default, including generated TTL, CSV summaries, reports, plots, and notebooks.
- Remove hardcoded secret leakage.
- Deduplicate only confirmed duplicates.
- Repository guidance files should be reviewed and updated alongside the refactor so they describe the final canonical layout accurately.

## Current Top-Level Areas

- `Soil Dataset/`: tables, figures, prompts, extraction notebooks, KG outputs, executability checks, and evaluation artifacts.
- `VisText/`: prompt text, data folders, JSON-to-TTL converters, and evaluation utilities.
- `diagram2graph Dataset/`: labels, extraction notebooks, TTL outputs, JSON-to-TTL conversion, and evaluation reports.

## Canonicalization Decisions

### VisText

- Canonical `VisText/Data/` layout is:
  - `VisText/Data/images/`
  - `VisText/Data/labels/`
  - keep `VisText/Data/Json2ttl/`
- Historical note: the canonical folders were populated from the approved `VisText/Data/1000_sub data/` source, then the duplicate legacy input directories were removed.

## Preserve-By-Default Assets

- Jupyter notebooks across all three datasets.
- Generated TTL output folders, including experimental and evaluation outputs.
- Evaluation CSVs, PNG plots, Markdown/HTML reports, and archive bundles.
- One-off research outputs that may represent costly model runs.

## Confirmed Refactor Risks And Targets

### Hardcoded Secret Leakage

- Confirmed embedded Gemini API keys appear in notebooks such as `diagram2graph Dataset/Extrcat RDF json/diagram2graph-inferance.ipynb`.
- Confirmed embedded keys also appear in `diagram2graph Dataset/Extract RDF ttl/Code.ipynb`.
- These should be replaced with environment variables or local ignored configuration before broader publication or cleanup.

### Hardcoded Machine Paths

- `diagram2graph Dataset/Evaluating/report/plot_summaries Clude.py` still uses `/mnt/data` assumptions and should be treated as legacy until it is removed or aligned.
- Multiple reports reference `/mnt/data/...` assets directly, which makes them non-portable.
- Additional notebooks include Colab or Kaggle paths such as `/content/drive` and `/kaggle/input/...`.

### Confirmed Duplicate Candidates

- `diagram2graph Dataset/Evaluating/report/plot_summaries Clude.py`
- This file now diverges from the parameterized `plot_summaries.py` and should be removed or archived in a later legacy-cleanup task.

## Task 7 Asset Boundaries

- Reusable shared helpers live under `common/`.
- Reusable dataset-level assets should stay in canonical data folders, stable prompt-text folders, and maintained converter/report scripts.
- Historical model outputs, KG renders, and evaluation reports remain preserve-by-default artifacts and should not be moved into a new archive layout yet.
- Legacy or staging areas should be labeled in-place with internal docs before any future move or rename decisions.

## Deferred Work

- `README.md` rewrite and repository-facing narrative cleanup.
- Top-level directory renaming or consolidation.
- Broad notebook standardization beyond secrets, paths, and reference fixes.
