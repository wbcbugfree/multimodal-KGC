# Multimodal-KGC

Multimodal-KGC is a research codebase for constructing RDF/Turtle knowledge graphs from multimodal scientific, chart, and diagram images. The repository supports three dataset workflows:

- `vistext`: labelled chart images with JSON labels, converted ground-truth Turtle, Gemini-generated Turtle outputs, and graph-matching evaluation.
- `diagram2graph`: labelled diagram images with JSON labels, converted ground-truth Turtle, Gemini-generated Turtle outputs, graph-matching evaluation, and a JSON-vs-Turtle ablation workflow.
- `soil_health`: soil-health figures and tables used as an unlabelled image-to-KG use case.

The codebase is dataset-first rather than package-first. It intentionally preserves generated TTL/JSON/CSV artifacts used in experiments, and most reproducibility entry points are standalone scripts rather than a single application or test suite.

## Repository Layout

```text
common/              Shared path and config helpers.
vistext/             Chart image-to-KG workflow.
diagram2graph/       Diagram image-to-KG workflow and historical outputs.
soil_health/         Soil-health figure/table image-to-KG workflow.
llm-as-a-judge/      Optional LLM-as-a-judge evaluation workflow.
requirements.txt     Minimal shared Python dependencies.
config               Local gitignored API-key JSON file, if used.
```

## Environment Setup

Use Python 3.10+ if possible. The examples below use `python`, which is the working command in this Windows checkout. On Linux or macOS, `python3` may be the correct executable.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The Gemini and OpenAI SDKs are optional because they are only required for live model calls:

```bash
pip install google-genai openai
```

Some notebooks may require additional packages beyond `requirements.txt`. Those notebooks are retained as research artifacts and are not required for the main script-based reproduction path.

## Secrets

API keys are loaded through `common/config.py`. Environment variables take precedence over a local top-level `config` file.

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

## VisText: Ground Truth Conversion

VisText data is organized by split:

```text
vistext/data/train/images/
vistext/data/train/labels/
vistext/data/train/turtle/
vistext/data/eval/images/
vistext/data/eval/labels/
vistext/data/eval/turtle/
vistext/data/test/images/
vistext/data/test/labels/
vistext/data/test/turtle/
```

The reusable converter is:

```text
vistext/data/json2ttl_converter.py
```

Single-file smoke conversion:

```bash
python "vistext/data/json2ttl_converter.py" "vistext/data/test/labels/1046.json" -o ".tmp/1046.ttl"
```

Folder conversion is supported by the same converter. Each split has an `exceptions_report.json` next to the split folder, marking records that should be included or excluded from experiments.

## VisText: Gemini RDF/Turtle Generation

VisText has four prompting strategies. Each strategy wrapper delegates shared sampling, parallel API calling, retry handling, manifest writing, and Turtle syntax validation to `vistext/extract_rdf_ttl/gemini_vistext_runner_core.py`.

```text
vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py
vistext/extract_rdf_ttl/gemini_vistext_oneshot_static.py
vistext/extract_rdf_ttl/gemini_vistext_oneshot_dynamic.py
vistext/extract_rdf_ttl/gemini_vistext_fewshot.py
```

Dry-run one image:

```bash
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --dry-run --sample-mode ids --ids 1046
```

Run a small sample with parallel Gemini calls:

```bash
python "vistext/extract_rdf_ttl/gemini_vistext_zeroshot.py" --sample-mode random --sample-count 5 --parallel-workers 4
```

Generated outputs are written to strategy-specific folders under `vistext/extract_rdf_ttl/`, for example `vistext_zeroshot_outputs/`.

## VisText: Traditional Evaluation

The main evaluation handler compares generated TTL files against ground-truth TTL files:

```text
vistext/evaluation/evaluate_vistext_llm_outputs.py
```

Evaluate all available strategy outputs in `content_only` mode:

```bash
python "vistext/evaluation/evaluate_vistext_llm_outputs.py" --graph-modes content_only --ged-workers 1
```

Parallel GED can be enabled locally:

```bash
python "vistext/evaluation/evaluate_vistext_llm_outputs.py" --graph-modes content_only --ged-workers 8
```

The output report is:

```text
vistext/evaluation/vistext_llm_evaluation_results.json
```

Metrics include triple matching, ROUGE, BLEU, BERTScore, and normalized graph edit distance. The handler supports numeric tolerance for quantity-like datapoint values.

## Diagram2Graph: Ground Truth Conversion

Diagram2Graph is the second labelled dataset used for the image-to-KG task. It contains diagram images, JSON labels, and converted RDF/Turtle ground truth:

```text
diagram2graph/data/images/
diagram2graph/data/labels/
diagram2graph/data/turtle/
```

The reusable converter is:

```text
diagram2graph/data/json2ttl_converter.py
```

Single-file smoke conversion:

```bash
python "diagram2graph/data/json2ttl_converter.py" "diagram2graph/data/labels/9.json" ".tmp/d2g-9.ttl"
```

Folder conversion is supported by the same converter:

```bash
python "diagram2graph/data/json2ttl_converter.py" "diagram2graph/data/labels" "diagram2graph/data/turtle"
```

The current Turtle schema uses one lightweight namespace:

```turtle
@prefix : <http://example.org/diagram2graph#> .
```

Nodes are represented as `:NodeN` resources, edges as `:EdgeN` resources, and edge semantics are encoded with `:source`, `:target`, `:relationshipType`, and optional `:relationshipValue`.

## Diagram2Graph: Gemini RDF/Turtle Generation

Diagram2Graph has three prompting strategies because all inputs are diagrams and there is no chart-type/category routing:

```text
diagram2graph/extract_rdf_ttl/gemini_diagram2graph_zeroshot.py
diagram2graph/extract_rdf_ttl/gemini_diagram2graph_oneshot.py
diagram2graph/extract_rdf_ttl/gemini_diagram2graph_fewshot.py
```

The shared runner core is:

```text
diagram2graph/extract_rdf_ttl/gemini_diagram2graph_runner_core.py
```

The one-shot and few-shot prompts reuse curated examples under:

```text
diagram2graph/prompt_engineering/ground_truth/
```

Dry-run one image:

```bash
python "diagram2graph/extract_rdf_ttl/gemini_diagram2graph_zeroshot.py" --dry-run --sample-mode ids --ids 37
```

Run all selected images with parallel Gemini calls:

```bash
python "diagram2graph/extract_rdf_ttl/gemini_diagram2graph_zeroshot.py" --sample-mode all --parallel-workers 4
python "diagram2graph/extract_rdf_ttl/gemini_diagram2graph_oneshot.py" --sample-mode all --parallel-workers 4
python "diagram2graph/extract_rdf_ttl/gemini_diagram2graph_fewshot.py" --sample-mode all --parallel-workers 4
```

Generated outputs are written to:

```text
diagram2graph/extract_rdf_ttl/zeroshot_outputs/
diagram2graph/extract_rdf_ttl/oneshot_outputs/
diagram2graph/extract_rdf_ttl/fewshot_outputs/
```

## Diagram2Graph: Traditional Evaluation

The Diagram2Graph evaluator compares generated TTL files against ground-truth TTL files:

```text
diagram2graph/evaluation/evaluate_diagram2graph_llm_outputs.py
```

Evaluate available strategy outputs:

```bash
python "diagram2graph/evaluation/evaluate_diagram2graph_llm_outputs.py" --ged-workers 1
```

The output report is:

```text
diagram2graph/evaluation/diagram2graph_llm_evaluation_results.json
```

Metrics match the VisText retained metric set: triple matching, ROUGE, BLEU, BERTScore, and normalized graph edit distance. Unlike VisText, Diagram2Graph evaluation keeps the full graph only: node and edge ordinals are semantic, and every triple is treated as meaningful.

## Diagram2Graph: JSON-Format Ablation

`diagram2graph/evaluation/ablation_study/` compares structured-output format effects for the same diagram-understanding task. The ablation tests whether asking an LLM for JSON instead of RDF/Turtle changes performance when both formats represent the same diagram graph.

The 20-image ablation set is:

```text
diagram2graph/evaluation/ablation_study/eval_img/
diagram2graph/evaluation/ablation_study/eval_gt/
```

Existing fine-tuned Qwen2.5-VL-3B JSON outputs are preserved in:

```text
diagram2graph/evaluation/ablation_study/outputs/it1_json/
diagram2graph/evaluation/ablation_study/outputs/it2_json/
```

Gemini JSON structured-output runners are:

```text
diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_zeroshot.py
diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_oneshot.py
diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_fewshot.py
```

These runners use Gemini structured output with `responseMimeType="application/json"` and a predefined response schema.

Dry-run one ablation image:

```bash
python "diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_zeroshot.py" --dry-run --sample-mode ids --ids 37
```

Run all three Gemini JSON strategies:

```bash
python "diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_zeroshot.py" --sample-mode all --parallel-workers 4
python "diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_oneshot.py" --sample-mode all --parallel-workers 4
python "diagram2graph/evaluation/ablation_study/json_prompting/gemini_json_fewshot.py" --sample-mode all --parallel-workers 4
```

Evaluate Qwen and Gemini JSON outputs:

```bash
python "diagram2graph/evaluation/ablation_study/evaluate_json_outputs.py" --ged-workers 1
```

The output report is:

```text
diagram2graph/evaluation/ablation_study/json_ablation_evaluation_results.json
```

## LLM-as-a-Judge Evaluation

`llm-as-a-judge/` contains an optional no-gold judge workflow for evaluating image-to-KG outputs when ground-truth labels are unavailable. It is designed for `soil_health`, and should be validated on the two labelled datasets before being used as the main soil-health evaluator:

- `vistext`: compare judge scores against traditional `content_only` metrics.
- `diagram2graph`: compare judge scores against traditional full-graph metrics.

The judge prompts are adapted from the KGEval prompt pattern:

- XML-like sections such as `<role>`, `<task>`, `<rating_scale>`, `<evaluation_steps>`, and `<output_format>`.
- KGEval-style criteria adapted to image-to-KG: `relevance`, `factuality`, `informativeness`, `coherence`, and `specificity`.
- Strict JSON structured outputs through the OpenAI Responses API using Pydantic response models, not prompt-only JSON formatting.

Entry points:

```text
llm-as-a-judge/evaluate_vistext_judge.py
llm-as-a-judge/evaluate_diagram2graph_judge.py
llm-as-a-judge/evaluate_soil_health_judge.py
```

Dry-run VisText judge validation:

```bash
python "llm-as-a-judge/evaluate_vistext_judge.py" --dry-run --modes direct pairwise --sample-mode ids --ids 1248
```

Dry-run Diagram2Graph judge validation:

```bash
python "llm-as-a-judge/evaluate_diagram2graph_judge.py" --dry-run --modes direct pairwise --sample-mode ids --ids 3
```

Dry-run soil-health judge evaluation:

```bash
python "llm-as-a-judge/evaluate_soil_health_judge.py" --dry-run --modes direct pairwise --sample-mode ids --ids table_1.2
```

Live OpenAI judge run:

```bash
python "llm-as-a-judge/evaluate_soil_health_judge.py" --modes direct pairwise --sample-mode random --sample-count 5 --judge-model gpt-5-mini
```

Judge results are written only under:

```text
llm-as-a-judge/results/
```

## Soil-Health Use Case

Soil-health has figure and table images but no full ground-truth label set. It is used as the unlabelled use case for image-to-KG generation and LLM-as-a-judge evaluation.

Raw image folders:

```text
soil_health/data/figures/
soil_health/data/tables/
```

Gemini runner entry points:

```text
soil_health/extract_rdf_ttl/gemini_soil_health_zeroshot.py
soil_health/extract_rdf_ttl/gemini_soil_health_oneshot_static.py
soil_health/extract_rdf_ttl/gemini_soil_health_oneshot_dynamic.py
soil_health/extract_rdf_ttl/gemini_soil_health_fewshot.py
```

Dry-run one image:

```bash
python "soil_health/extract_rdf_ttl/gemini_soil_health_zeroshot.py" --dry-run --sample-mode ids --ids table_1.2
```

## Reproducibility Checklist

For reviewers, the smallest script-based reproduction path is:

1. Install the minimal dependencies with `pip install -r requirements.txt`.
2. Run the VisText converter smoke command to verify JSON-to-Turtle conversion.
3. Run a VisText Gemini dry-run to verify model-call configuration without spending API calls.
4. Run `evaluate_vistext_llm_outputs.py` on available generated outputs.
5. Run the Diagram2Graph converter smoke command to verify diagram JSON-to-Turtle conversion.
6. Run a Diagram2Graph Gemini dry-run to verify labelled diagram generation without spending API calls.
7. Run `evaluate_diagram2graph_llm_outputs.py` on available generated outputs.
8. Optionally run the JSON-format ablation evaluator on the existing Qwen JSON outputs.
9. Optionally run `llm-as-a-judge/evaluate_vistext_judge.py --dry-run` and `llm-as-a-judge/evaluate_diagram2graph_judge.py --dry-run` to verify labelled judge-validation input discovery.
10. Run soil-health Gemini or judge dry-runs to verify the unlabelled use-case workflow.

There is no repo-wide `pytest` or CI workflow. Report exact commands and outputs when validating changes.

## Generated Artifacts

This repository intentionally keeps many generated TTL, JSON, CSV, and notebook artifacts because they document the experiments. Do not delete generated outputs unless the task explicitly asks for pruning or regeneration.

## License

This repository is licensed under the MIT License. See `LICENSE`.
