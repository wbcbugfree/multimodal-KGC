# Conservative Academic Refactor Implementation Plan

**Goal:** Reorganize the repository conservatively so research assets stay usable while paths, secrets, duplicate files, and fragile scripts become safer and easier to maintain.

**Architecture:** Keep the current top-level dataset directories in place and refactor inside each dataset first. Prioritize path normalization, documentation, and parameterization before any broad code movement. Preserve notebooks, generated outputs, and evaluation artifacts unless they are confirmed duplicates or secret-bearing files that must be sanitized.

**Tech Stack:** Python 3 scripts, Jupyter notebooks, Markdown docs, existing dataset folders and generated RDF/CSV/PNG artifacts.

---

## Approved Direction

- This plan defines a conservative refactor for an academic, notebook-first repository.
- Keep the top-level dataset directories for now: `soil_dataset/`, `vistext/`, and `diagram2graph_dataset/`.
- Do not rewrite `README.md` in this task; README work happens later.
- Preserve valuable outputs, reports, and notebook assets unless they are confirmed safe to remove.
- Remove hardcoded secret leakage and replace it with environment-variable or local-config patterns.
- Deduplicate only confirmed duplicates; do not collapse near-duplicates without checking references and provenance.
- Repository guidance should be updated to match the final refactored layout, including any agent-instruction files present in the repository.

## Task 1: Document Current State And Guardrails

**Files:**
- Create: `docs/plans/2026-03-17-conservative-academic-refactor.md`
- Create: `docs/refactor_inventory.md`
- Create: `docs/repo_map.md`

**Step 1: Capture the current repository shape**

Document the current dataset-first layout and note that the top level remains stable during the first refactor phase.

**Step 2: Record the approved migration rules**

Capture the specific vistext decision: move `vistext/data/1000_sub data/` contents into canonical `vistext/data/images/` and `vistext/data/labels/`, then remove `vistext/data/sub-image/`, `vistext/data/ground truth/`, and `vistext/data/1000_sub data/` only after path migration and verification are complete.

**Step 3: Record preservation rules**

Mark notebooks, generated TTL outputs, evaluation CSVs, plots, and reports as preserve-by-default. Require explicit confirmation before deleting outputs that may encode experiment results.

**Step 4: Record cleanup priorities**

List hardcoded secrets, machine-specific paths, and confirmed duplicate scripts as first-wave cleanup targets.

## Task 2: Path Normalization Without Top-Level Moves

**Files:**
- Modify later: `vistext/**`
- Modify later: `diagram2graph_dataset/**`
- Modify later: `soil_dataset/**`

**Step 1: Create canonical data targets inside vistext**

Create `vistext/data/images/` and `vistext/data/labels/` as the canonical locations for image and label assets.

**Step 2: Migrate vistext inputs by content type**

Move image files from `vistext/data/sub-image/` and any image payloads in `vistext/data/1000_sub data/` into `vistext/data/images/`. Move label JSON files from `vistext/data/ground truth/` and `vistext/data/1000_sub data/label/` into `vistext/data/labels/`.

**Step 3: Update code and notebook references**

Update scripts, notebooks, and docs to point to `vistext/data/images/` and `vistext/data/labels/`.

**Step 4: Remove legacy vistext folders only after verification**

Remove `vistext/data/sub-image/`, `vistext/data/ground truth/`, and `vistext/data/1000_sub data/` only after path migration is complete and representative conversions still work.

## Task 3: Secret And Machine-Path Cleanup

**Files:**
- Modify later: notebooks and scripts with embedded keys or absolute paths

**Step 1: Remove hardcoded secret leakage**

Replace embedded API keys in notebooks and scripts with `os.getenv(...)` or a local ignored config pattern. Check notebook outputs for accidental key exposure before preserving them.

**Step 2: Replace hardcoded machine paths**

Parameterize paths such as `/mnt/data`, Colab `/content/...`, Kaggle `/kaggle/input/...`, and machine-specific home-directory references.

**Step 3: Document local execution assumptions**

Add usage notes near scripts that currently depend on manual file placement or notebook-specific environments.

## Task 4: Deduplicate Confirmed Duplicates Conservatively

**Files:**
- Modify later: confirmed duplicate scripts and reports

**Step 1: Keep both copies until reference usage is checked**

Examples already visible include `diagram2graph_dataset/evaluation/report/plot_summaries.py` and `diagram2graph_dataset/evaluation/report/plot_summaries_claude.py`, which appear identical and are strong candidates for deduplication.

**Step 2: Remove only confirmed duplicates**

If two files are byte-identical or semantically identical and one is unused, keep the canonical file and remove the duplicate after updating references.

**Step 3: Preserve distinct outputs**

Do not deduplicate generated outputs, CSVs, or reports solely by filename similarity; verify that contents and intended provenance match.

## Task 5: Soil Cleanup And Staging Clarification

**Files:**
- Modify later: `soil_dataset/**`

**Step 1: Trim only confirmed Soil staging mirrors**

Remove exact mirrored files from Soil staging folders only after confirming that active notebooks and evaluation packets do not still require the legacy staging name.

**Step 2: Keep compatibility mirrors where active consumers still depend on them**

If a staging notebook still expects a specific file name such as `rdf_extractions_fewshot.json`, retain a compatibility mirror and document why it remains.

**Step 3: Label staging-only assets clearly**

Document which Soil folders are canonical, which are evaluation-specific, and which are retained only for legacy workflow compatibility.

## Task 6: Secret Loading Cleanup Across Notebooks

**Files:**
- Modify later: secret-bearing notebooks in `soil_dataset/`, `vistext/`, and `diagram2graph_dataset/`

**Step 1: Remove embedded keys and unsafe prompts**

Replace hardcoded API keys, Kaggle-only secret lookups, and interactive key prompts with shared config or environment-variable loading.

**Step 2: Reuse shared helpers conservatively**

Use the top-level `common/` helpers where the notebook runs inside the repo checkout, while allowing environment-variable fallback when a notebook is copied elsewhere.

**Step 3: Preserve notebook outputs unless sanitization requires otherwise**

Make source-only notebook edits where possible so research outputs remain intact.

## Task 7: Separate Reusable Assets From Legacy Experiments

**Files:**
- Create: `docs/task7-asset-boundaries.md`
- Create or update: dataset-level `readme_internal.md` notes

**Step 1: Classify reusable vs historical areas in place**

Mark canonical data folders, shared scripts, and prompt assets as reusable, and label outputs, staging folders, and legacy notebooks as historical or staging areas.

**Step 2: Avoid structural moves in this phase**

Do not rename typo-bearing folders, move preserved outputs into new archive trees, or collapse legacy experiment folders yet.

**Step 3: Feed the classification into later docs**

Use these reusable-vs-historical boundaries when rewriting `README.md` and repository guidance files later in the refactor.

## Task 8: Reviewer-Facing README Rewrite

**Files:**
- Modify later: `README.md`

**Step 1: Explain the repository like an academic artifact**

Describe datasets, workflows, canonical inputs, preserved outputs, and how reviewers can run representative conversions without assuming hidden context.

**Step 2: Reflect the conservative refactor state accurately**

Document current canonical paths and clearly label historical outputs and notebook-first workflows.

## Task 9: Guidance File Alignment

**Files:**
- Modify later: `AGENTS.md`
- Modify later: `.github/copilot-instructions.md`

**Step 1: Align guidance files with the final canonical layout**

Update repository guidance so future agents and contributors see the same canonical paths, validation commands, and preserve-vs-legacy rules.

## Task 10: Final Validation Pass

**Files:**
- Modify later: docs and any touched scripts/notebooks

**Step 1: Re-run representative smoke tests**

Run the verified dataset-specific smoke tests that still apply after refactoring.

**Step 2: Verify documentation claims against the worktree**

Ensure README and internal docs match the actual folder structure, preserved outputs, and current validation commands.

## Validation Expectations For Later Tasks

- Use representative smoke tests, not invented repo-wide test commands.
- Validate at least one vistext conversion after path migration.
- Validate at least one diagram2graph conversion after path or script cleanup.
- State exactly what was validated; do not claim a unified test suite exists.

## Implementation Constraints

- README rewrite is intentionally deferred until the structural refactor is in place.
- Changes should preserve current research assets unless they are confirmed duplicate mirrors or secret-bearing sources that require sanitization.
