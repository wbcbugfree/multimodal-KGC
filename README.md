# Multimodal-KGC

Multimodal-KGC is a research repository for constructing RDF/Turtle knowledge graphs directly from infographic images. The repository focuses on scientific visual artifacts that are commonly lost during document preprocessing, including charts, flow/process diagrams, figures, and tables.

The project has three main components:

- End-to-end image-to-RDF prompting with Gemini 3 Flash.
- Two labelled image-to-RDF datasets for reference-based evaluation: VisText-derived chart-to-RDF and Diagram2Graph-derived diagram-to-RDF.
- An LLM-as-a-judge workflow for evaluating unlabelled image-to-RDF outputs, validated on the labelled datasets and applied to a soil-health use case.

This is a dataset-first research codebase, not a packaged Python library. Scripts are standalone entry points, generated outputs are preserved as experiment artifacts, and there is no repo-wide `pytest` or CI workflow.

## Workflow Overview

The figure below summarizes the repository workflow: labelled VisText and Diagram2Graph datasets support metric-based validation, Gemini 3 Flash generates RDF/Turtle directly from images under predefined schemas, the generated graphs are evaluated with graph-matching metrics when references are available, and the validated LLM-as-a-judge workflow is used for the unlabelled soil-health application.

![Overview of the image-to-RDF workflow](img/overview_figure_main.svg)

## Repository Layout

```text
common/                 Shared path, config, and Gemini batch helpers.
vistext/                Chart image-to-RDF dataset, Gemini runners, and evaluation.
diagram2graph/          Diagram image-to-RDF dataset, Gemini runners, evaluation, and JSON/RDF ablation.
soil_health/            Unlabelled soil-health figure/table use case.
llm-as-a-judge/         OpenAI-based judge validation and unlabelled evaluation workflow.
requirements.txt        Python dependencies used by converters, runners, and evaluators.
LICENSE                 MIT license.
```

## Environment Setup

Use Python 3.10 or newer. The examples below use `python`, which is the working command in this Windows checkout. On Linux or macOS, `python3` may be the correct executable.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The evaluation scripts use BERTScore through Hugging Face models. If model files are not already cached locally, the first full evaluation may need internet access unless you pass `--allow-online-model-download` according to the evaluator CLI. For faster BERTScore evaluation on a CUDA-enabled machine, use `--bert-device cuda` or `--bert-device auto`.

## API Keys

API keys are loaded by `common/config.py`. Environment variables take precedence over a local top-level `config` JSON file.

Recommended environment variables:

```bash
set GEMINI_API_KEY=...
set OPENAI_API_KEY=...
```

Equivalent local `config` file:

```json
{
  "gemini_api_key": "...",
  "openai_api_key": "..."
}
```

Do not commit `config`.

## Datasets

### VisText-Derived Chart-to-RDF

VisText is used as a labelled chart-to-RDF dataset. The target RDF graph represents:

- chart type: `:BarChart`, `:LineChart`, or `:AreaChart`
- chart title
- x-axis and y-axis titles
- one `:DataPoint` resource per plotted observation
- `:xValue` and `:yValue` literals for each datapoint

Data are organized by split:

```text
vistext/data/train/images/   vistext/data/train/labels/   vistext/data/train/turtle/
vistext/data/eval/images/    vistext/data/eval/labels/    vistext/data/eval/turtle/
vistext/data/test/images/    vistext/data/test/labels/    vistext/data/test/turtle/
```

The converted RDF release covers all three original splits. The LLM experiments use only the test split. Each split has an `exceptions_report.json` with `include` and `exclude` counts; excluded records have source-label issues and are omitted from experiments.

| Split | Raw images | Usable RDF graphs |
| --- | ---: | ---: |
| train | 7057 | 6869 |
| eval | 883 | 855 |
| test | 882 | 868 |

Converter:

```bash
python "vistext/data/json2ttl_converter.py" "vistext/data/test/labels/1046.json" -o ".tmp/1046.ttl"
```

### Diagram2Graph-Derived Diagram-to-RDF

Diagram2Graph is used as a labelled diagram-to-RDF dataset. The target RDF graph represents:

- diagram nodes as `:NodeN`
- node labels, node types, and node shapes
- directed edges as `:EdgeN`
- edge source, target, line style, relationship type, and optional edge label

Data layout:

```text
diagram2graph/data/images/
diagram2graph/data/labels/
diagram2graph/data/turtle/
```

The dataset contains 219 labelled diagrams and 219 converted RDF graphs.

Converter:

```bash
python "diagram2graph/data/json2ttl_converter.py" "diagram2graph/data/labels/9.json" ".tmp/d2g-9.ttl"
python "diagram2graph/data/json2ttl_converter.py" "diagram2graph/data/labels" "diagram2graph/data/turtle"
```

### Soil-Health Image Collection

The soil-health collection is an unlabelled use case with 71 images from the European Environment Agency report *Soil monitoring in Europe: Indicators and thresholds for soil health assessments*:

```text
soil_health/data/figures/   15 figure images
soil_health/data/tables/    56 table images
```

These images are used to test whether image-to-RDF generation can recover information from scientific figures and tables when no gold RDF graphs are available. Outputs are evaluated with the LLM-as-a-judge workflow after judge validation on the labelled datasets.

## Prompting Strategies

The repository implements the following prompting strategies:

| Dataset | Strategies |
| --- | --- |
| VisText | zero-shot, static one-shot, dynamic one-shot, few-shot |
| Diagram2Graph | zero-shot, static one-shot, few-shot |
| Soil Health | zero-shot, static one-shot, dynamic one-shot, few-shot |

Dynamic one-shot prompting first classifies the input image type, then selects a type-specific system prompt and example. VisText classifies charts as `bar`, `line`, or `area`; Soil Health classifies images as `figure` or `table`.

Prompt files and examples are under each dataset's `prompt_engineering/` folder.

## Gemini Image-to-RDF Inference

All Gemini runners default to `gemini-3-flash-preview`. They validate generated RDF/Turtle with `rdflib`, retry failed API calls or Turtle syntax failures, and write a `manifest.json` in each output folder.

### VisText

```bash
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --dry-run --sample-mode ids --ids 1046
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --sample-mode random --sample-count 5 --parallel-workers 4
python "vistext/extract_rdf_ttl/gemini_vistext_oneshot_static.py" --sample-mode all --parallel-workers 4
python "vistext/extract_rdf_ttl/gemini_vistext_oneshot_dynamic.py" --sample-mode all --parallel-workers 4
python "vistext/extract_rdf_ttl/gemini_vistext_fewshot.py" --sample-mode all --parallel-workers 4
```

Output folders:

```text
vistext/extract_rdf_ttl/vistext_zeroshot_outputs/
vistext/extract_rdf_ttl/vistext_oneshot_static_outputs/
vistext/extract_rdf_ttl/vistext_oneshot_dynamic_outputs/
vistext/extract_rdf_ttl/vistext_fewshot_outputs/
```

### Diagram2Graph

```bash
python "diagram2graph/extract_rdf_ttl/gemini_diagram2graph_zeroshot.py" --dry-run --sample-mode ids --ids 37
python "diagram2graph/extract_rdf_ttl/gemini_diagram2graph_zeroshot.py" --sample-mode all --parallel-workers 4
python "diagram2graph/extract_rdf_ttl/gemini_diagram2graph_oneshot.py" --sample-mode all --parallel-workers 4
python "diagram2graph/extract_rdf_ttl/gemini_diagram2graph_fewshot.py" --sample-mode all --parallel-workers 4
```

Output folders:

```text
diagram2graph/extract_rdf_ttl/zeroshot_outputs/
diagram2graph/extract_rdf_ttl/oneshot_outputs/
diagram2graph/extract_rdf_ttl/fewshot_outputs/
```

### Soil Health

```bash
python "soil_health/extract_rdf_ttl/gemini_soil_health_zeroshot.py" --dry-run --sample-mode ids --ids table_1.2
python "soil_health/extract_rdf_ttl/gemini_soil_health_zeroshot.py" --sample-mode all --parallel-workers 4
python "soil_health/extract_rdf_ttl/gemini_soil_health_oneshot_static.py" --sample-mode all --parallel-workers 4
python "soil_health/extract_rdf_ttl/gemini_soil_health_oneshot_dynamic.py" --sample-mode all --parallel-workers 4
python "soil_health/extract_rdf_ttl/gemini_soil_health_fewshot.py" --sample-mode all --parallel-workers 4
```

Output folders:

```text
soil_health/extract_rdf_ttl/zeroshot/
soil_health/extract_rdf_ttl/oneshot_static/
soil_health/extract_rdf_ttl/oneshot_dynamic/
soil_health/extract_rdf_ttl/fewshot/
```

## Gemini Batch API

Gemini Batch API is available for full-dataset inference. Batch mode is useful for large runs because it avoids keeping a local process open for every request. For one-shot and few-shot strategies, batch mode can also use Gemini context caching for reusable examples with `--batch-context-cache auto` (default). Zero-shot prompts are not cached because they have no reusable examples.

Basic lifecycle:

```bash
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --batch-action submit
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --batch-action status
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --batch-action collect
```

For dynamic one-shot, classification can also be batched before RDF generation:

```bash
python "vistext/extract_rdf_ttl/gemini_vistext_oneshot_dynamic.py" --classification-batch-action submit
python "vistext/extract_rdf_ttl/gemini_vistext_oneshot_dynamic.py" --classification-batch-action collect
python "vistext/extract_rdf_ttl/gemini_vistext_oneshot_dynamic.py" --batch-action submit --require-classification-manifest
python "vistext/extract_rdf_ttl/gemini_vistext_oneshot_dynamic.py" --batch-action collect
```

The same `--batch-action submit/status/collect/wait` pattern is supported by the Diagram2Graph and Soil Health Gemini strategy scripts.

## Traditional Graph Evaluation

Traditional evaluation is used only for labelled datasets with reference RDF graphs.

### VisText Evaluation

```bash
python "vistext/evaluation/evaluate_vistext_llm_outputs.py" --graph-modes full_graph content_only --numeric-tolerance 0.01 --bert-device auto --ged-workers 1
```

Important options:

- `--graph-modes full_graph content_only`: report both complete-schema and content-focused evaluation.
- `--numeric-tolerance 0.01`: apply 1% relative tolerance to numeric datapoint values.
- `--metric-set structural`: compute only triple matching, triple accuracy, and GED.
- `--tolerance-sweep 0 0.005 0.01 0.015 0.02`: evaluate multiple numeric tolerances.
- `--bert-device auto|cpu|cuda`: choose BERTScore device.
- `--ged-workers N`: enable parallel GED where supported.

Main result files:

```text
vistext/evaluation/vistext_prompting_strategy_evaluation_results.json
vistext/evaluation/vistext_zeroshot_tolerance_sweep_results.json
```

### Diagram2Graph Evaluation

```bash
python "diagram2graph/evaluation/evaluate_diagram2graph_llm_outputs.py" --bert-device auto --ged-workers 1
```

Diagram2Graph evaluation uses the full graph. Node and edge ordinals are semantic and are not normalized away.

Main result file:

```text
diagram2graph/evaluation/diagram2graph_llm_evaluation_results.json
```

### Metrics

The retained metric set is:

- triple-match precision, recall, and F1
- mean triple-match accuracy
- normalized graph edit distance (nGED)
- ROUGE-F1
- BLEU-F1
- G-BERTScore-F1

VisText additionally reports chart-type breakdowns and numeric-tolerance sweeps. Diagram2Graph additionally reports a JSON-versus-RDF serialization ablation.

## Diagram2Graph JSON-versus-RDF Ablation

The ablation study tests whether generating JSON instead of RDF/Turtle changes structured extraction performance when both formats encode the same diagram graph.

Files:

```text
diagram2graph/evaluation/ablation_study/eval_img/
diagram2graph/evaluation/ablation_study/eval_gt/
diagram2graph/evaluation/ablation_study/json_prompting/
diagram2graph/evaluation/ablation_study/outputs/
```

Gemini JSON structured-output runners:

```bash
python "diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_zeroshot.py" --sample-mode all --parallel-workers 4
python "diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_oneshot.py" --sample-mode all --parallel-workers 4
python "diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_fewshot.py" --sample-mode all --parallel-workers 4
```

Evaluation:

```bash
python "diagram2graph/evaluation/ablation_study/evaluate_json_outputs.py" --ged-workers 1
```

Main result files:

```text
diagram2graph/evaluation/ablation_study/json_ablation_evaluation_results.json
diagram2graph/evaluation/ablation_study/rdf_ablation_evaluation_results.json
```

## LLM-as-a-Judge Workflow

The LLM-as-a-judge workflow evaluates image-to-RDF outputs with OpenAI structured outputs. It supports:

- direct scoring of one RDF graph against the image
- pairwise comparison between two RDF graphs for the same image
- validation against traditional metrics on labelled datasets
- unlabelled evaluation for Soil Health

The default judge model is `gpt-5-mini`. OpenAI Batch API support is intentionally not included; use `--parallel-workers` for parallel non-batch calls.

Entry points:

```text
llm-as-a-judge/evaluate_vistext_judge.py
llm-as-a-judge/evaluate_diagram2graph_judge.py
llm-as-a-judge/evaluate_soil_health_judge.py
```

Dry-runs:

```bash
python "llm-as-a-judge/evaluate_vistext_judge.py" --dry-run --modes direct pairwise --sample-mode ids --ids 1248
python "llm-as-a-judge/evaluate_diagram2graph_judge.py" --dry-run --modes direct pairwise --sample-mode ids --ids 37
python "llm-as-a-judge/evaluate_soil_health_judge.py" --dry-run --modes direct pairwise --sample-mode ids --ids table_1.2
```

Example live Soil Health direct scoring:

```bash
python "llm-as-a-judge/evaluate_soil_health_judge.py" --modes direct --sample-mode all --judge-model gpt-5-mini --parallel-workers 5
```

Example labelled validation using top-margin strategy sampling:

```bash
python "llm-as-a-judge/evaluate_diagram2graph_judge.py" --modes direct pairwise --validation-design strategy_margin_top_n --top-margin-threshold 0.05 --top-margin-threshold-mode any --parallel-workers 5
```

Judge outputs:

```text
llm-as-a-judge/results/vistext/
llm-as-a-judge/results/diagram2graph/
llm-as-a-judge/results/soil_health/
```

## Included Experiment Results

The repository includes generated RDF/JSON outputs and evaluation reports. The following summary values are provided to orient users before they inspect the JSON reports.

### VisText

Evaluation uses 868 usable test-split charts. Full-graph and content-only results are both reported with 1% numeric tolerance for exact graph-overlap metrics.

| Strategy | Content-only F1 | Content-only nGED | Whole-graph F1 | Whole-graph nGED |
| --- | ---: | ---: | ---: | ---: |
| zero-shot | 0.6674 | 0.0961 | 0.7849 | 0.0749 |
| static one-shot | 0.6664 | 0.0959 | 0.7873 | 0.0738 |
| dynamic one-shot | 0.6769 | 0.0929 | 0.7938 | 0.0718 |
| few-shot | 0.6713 | 0.0945 | 0.7932 | 0.0722 |

The chart-type breakdown shows that bar charts are easier than line and area charts. Best content-only F1 by chart type:

| Chart type | Best F1 | Best strategy |
| --- | ---: | --- |
| bar | 0.7867 | zero-shot |
| line | 0.6501 | dynamic one-shot |
| area | 0.5729 | few-shot |

Numeric tolerance analysis for zero-shot shows that many visual value estimates are close to the reference values:

| Tolerance | F1 | Matched numeric values |
| --- | ---: | ---: |
| 0% | 0.5355 | 27.6% |
| 0.5% | 0.6187 | 53.2% |
| 1% | 0.6674 | 67.6% |
| 1.5% | 0.6920 | 75.0% |
| 2% | 0.7047 | 79.0% |

### Diagram2Graph

Evaluation uses 216 diagrams after excluding the three in-context example images from all strategies.

| Strategy | F1 | nGED |
| --- | ---: | ---: |
| zero-shot | 0.8744 | 0.0624 |
| one-shot | 0.8740 | 0.0626 |
| few-shot | 0.8493 | 0.0799 |

On the 20-image JSON-versus-RDF ablation subset, direct RDF/Turtle generation does not show a performance decline relative to JSON serialization.

| System | Format | F1 | nGED |
| --- | --- | ---: | ---: |
| Gemini zero-shot | JSON | 0.7552 | 0.0931 |
| Gemini one-shot | JSON | 0.8150 | 0.0690 |
| Gemini few-shot | JSON | 0.8235 | 0.0651 |
| Qwen2.5-VL-3B, two-run mean | JSON | 0.7259 | 0.1311 |
| Gemini RDF zero-shot | RDF/Turtle | 0.8599 | 0.0643 |
| Gemini RDF one-shot | RDF/Turtle | 0.8382 | 0.0729 |
| Gemini RDF few-shot | RDF/Turtle | 0.8281 | 0.0866 |

### Soil Health

Soil Health has no reference RDF labels. LLM-as-a-judge direct scores are reported over 69 images per strategy after excluding the figure and table examples used for prompting.

| Strategy | Overall judge score |
| --- | ---: |
| zero-shot | 4.7971 |
| static one-shot | 4.8261 |
| dynamic one-shot | 4.7536 |
| few-shot | 4.7391 |

Pairwise judge comparison between static one-shot and few-shot is nearly balanced: few-shot is preferred for 34 images, static one-shot for 33 images, and 2 images are ties.

### LLM-as-a-Judge Validation

The judge workflow is validated on selected labelled samples before being used for Soil Health. Direct scores are more informative than pairwise preferences when the compared outputs have a visible quality gap.

| Dataset | Compared outputs | Direct score pattern | Pairwise agreement |
| --- | --- | --- | ---: |
| Diagram2Graph | zero-shot vs few-shot | zero-shot scored higher, consistent with graph metrics | 0.4737 |
| VisText | dynamic one-shot vs static one-shot | scores are nearly tied, consistent with small metric gaps | 0.4963 |

## Reproducing the Main Workflow

For a reviewer who wants to validate the repository without spending API calls, use the preserved generated outputs:

1. Install dependencies with `pip install -r requirements.txt`.
2. Run one VisText converter smoke test.
3. Run one Diagram2Graph converter smoke test.
4. Run `evaluate_vistext_llm_outputs.py` on existing VisText outputs.
5. Run `evaluate_diagram2graph_llm_outputs.py` on existing Diagram2Graph outputs.
6. Run the Diagram2Graph ablation evaluator on existing JSON/RDF outputs.
7. Run LLM-as-a-judge dry-runs to confirm input discovery without API calls.

For full model reproduction:

1. Configure `GEMINI_API_KEY`.
2. Run the Gemini strategy scripts for each dataset, either interactively with `--parallel-workers` or through Gemini Batch API.
3. Re-run traditional evaluation for VisText and Diagram2Graph.
4. Configure `OPENAI_API_KEY`.
5. Run LLM-as-a-judge validation on labelled datasets.
6. Run LLM-as-a-judge evaluation on Soil Health.

## Notes for Contributors

- Treat generated TTL, JSON, CSV, and manifest files as preserved research artifacts unless a task explicitly asks to regenerate or prune them.
- Prefer shared helpers in `common/paths.py`, `common/config.py`, and `common/gemini_batch.py` over hardcoded machine paths.
- Do not report "all tests pass"; there is no unified test harness. Report exact commands and what they verified.
- For small changes, prefer real-data smoke commands over synthetic tests.

## License

This repository is licensed under the MIT License. See `LICENSE`.
