# VisText V2 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce converter-side ground-truth errors in `vistext/data/json2ttl/json_to_ttl_converter_v2.py` while keeping `datatable` as the canonical source and only using `scenegraph` for conservative bar-chart repairs.

**Architecture:** Keep the existing V2 datatable-first flow, then add a validation/repair layer before TTL emission. Improve datatable tokenization for numeric-containing labels, add a minimal scenegraph parser borrowed from V3 for bar-label repair only when alignment is provably safe, and write a per-image exception report that classifies clean, repaired, suspicious, and error outcomes.

**Tech Stack:** Python 3, stdlib `json`/`re`/`argparse`, existing VisText JSON labels, `rdflib` for smoke validation.

---

### Task 1: Add Regression Coverage For Known Failure Shapes

**Files:**
- Create: `vistext/data/json2ttl/test_json_to_ttl_converter_v2.py`
- Modify: `vistext/data/json2ttl/json_to_ttl_converter_v2.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_numeric_prefix_label_is_not_split():
    ...

def test_numeric_suffix_label_is_not_split():
    ...

def test_scenegraph_repairs_damaged_bar_labels_without_reordering():
    ...
```

- [ ] **Step 2: Run the focused test file and verify it fails**

Run: `python -m unittest vistext.data.json2ttl.test_json_to_ttl_converter_v2 -v`
Expected: failures around numeric-in-label parsing and damaged bar labels.

- [ ] **Step 3: Keep the failing examples minimal but real**

Use representative IDs:
- `706` for labels beginning with numbers
- `1283` for labels containing internal numeric tokens
- `193` for damaged bar labels recoverable from scenegraph
- `1358` as a guard that year/value ordering does not get “corrected” by unsafe scenegraph reordering

- [ ] **Step 4: Re-run the failing tests before implementation**

Run: `python -m unittest vistext.data.json2ttl.test_json_to_ttl_converter_v2 -v`
Expected: still failing in the same places.

### Task 2: Harden Datatable Parsing Without Changing The Source-Of-Truth Model

**Files:**
- Modify: `vistext/data/json2ttl/json_to_ttl_converter_v2.py`
- Test: `vistext/data/json2ttl/test_json_to_ttl_converter_v2.py`

- [ ] **Step 1: Improve token protection for label phrases that contain numeric words**

Add conservative preprocessing for patterns that should stay atomic in categorical labels, such as:
- `80 years and older`
- `9 years or younger`
- `10 to 19 years`
- `BBC Radio 2`
- `BBC Radio 4 Extra`
- date-like labels such as `Mar 23`

- [ ] **Step 2: Keep parse_datatable datatable-first**

Do not reorder rows based on scenegraph. Do not interpolate values. Do not change line/area parsing into scenegraph-driven parsing.

- [ ] **Step 3: Add post-parse structural checks**

Flag suspicious point sequences when:
- categorical labels are empty or contain leaked placeholders such as `None val`
- bar-chart categorical labels obviously mismatch scenegraph tick inventory
- point counts disagree with scenegraph marks for bar charts

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest vistext.data.json2ttl.test_json_to_ttl_converter_v2 -v`
Expected: parser regressions fixed without changing guarded cases like `1358`.

### Task 3: Add Conservative Bar-Only Repair Borrowed From V3

**Files:**
- Modify: `vistext/data/json2ttl/json_to_ttl_converter_v2.py`
- Test: `vistext/data/json2ttl/test_json_to_ttl_converter_v2.py`

- [ ] **Step 1: Borrow only the minimal scenegraph helpers needed for bar repair**

Add small helpers for:
- parsing bar scenegraph ticks/marks
- determining bar orientation
- extracting scenegraph categorical tick labels

Do not import V3 wholesale. Do not add interpolation-based numeric fallback logic.

- [ ] **Step 2: Repair only when alignment is safe**

For bar charts only:
- keep exact datatable numeric values as canonical
- use scenegraph categorical labels only when bar count matches and a numeric-value anchor or monotonic order makes the mapping unambiguous
- if alignment is ambiguous, emit the datatable result unchanged and mark it suspicious

- [ ] **Step 3: Keep suspicious outputs instead of dropping them**

Always write a TTL unless conversion truly crashes, but record why the output is `repaired` or `suspicious`.

- [ ] **Step 4: Re-run the focused tests**

Run: `python -m unittest vistext.data.json2ttl.test_json_to_ttl_converter_v2 -v`
Expected: all focused regression tests pass.

### Task 4: Expand V2 Exception Reporting

**Files:**
- Modify: `vistext/data/json2ttl/json_to_ttl_converter_v2.py`

- [ ] **Step 1: Change the report from “errors only” to per-image outcomes**

For each image, record:
- `img_id`
- `status` (`clean`, `repaired`, `suspicious`, `error`)
- `reasons`
- `repair_actions`
- `ttl_file` when written

- [ ] **Step 2: Keep the top-level summary compact**

Summarize counts for:
- converted clean
- converted repaired
- converted suspicious
- hard errors

- [ ] **Step 3: Verify directory conversion still writes a report**

Run: `python "vistext/data/json2ttl/json_to_ttl_converter_v2.py" "vistext/data/labels" -o ".tmp/v2-report-smoke"`
Expected: TTL files plus `exceptions_report.json`.

### Task 5: Regenerate Outputs And Validate With Real Data

**Files:**
- Modify: `vistext/data/json2ttl/ground_truth_ttl/*`

- [ ] **Step 1: Regenerate the VisText V2 ground-truth folder**

Run: `python "vistext/data/json2ttl/json_to_ttl_converter_v2.py" "vistext/data/labels" -o "vistext/data/json2ttl/ground_truth_ttl"`

- [ ] **Step 2: Run targeted smoke checks on known IDs**

Inspect:
- `193.ttl`
- `706.ttl`
- `1283.ttl`
- `1358.ttl`
- `6859` report entry

- [ ] **Step 3: Parse representative TTL files with rdflib**

Run:
`python -c "from pathlib import Path; from rdflib import Graph; [Graph().parse(f, format='turtle') for f in [Path('vistext/data/json2ttl/ground_truth_ttl/193.ttl'), Path('vistext/data/json2ttl/ground_truth_ttl/706.ttl'), Path('vistext/data/json2ttl/ground_truth_ttl/1283.ttl'), Path('vistext/data/json2ttl/ground_truth_ttl/1358.ttl')]]; print('rdflib-ok')"`

- [ ] **Step 4: Review the diff footprint**

Run:
- `git diff --stat -- vistext/data/json2ttl/json_to_ttl_converter_v2.py`
- `git diff --stat -- vistext/data/json2ttl/ground_truth_ttl`
- `git diff -- vistext/data/json2ttl/ground_truth_ttl/193.ttl`
- `git diff -- vistext/data/json2ttl/ground_truth_ttl/706.ttl`
- `git diff -- vistext/data/json2ttl/ground_truth_ttl/1283.ttl`
- `git diff -- vistext/data/json2ttl/ground_truth_ttl/1358.ttl`

- [ ] **Step 5: Iterate once if a safe additional fix is obvious**

Only make a second revision if:
- it fixes a repeated pattern
- it is source-preserving
- it does not require scenegraph interpolation or order guessing
