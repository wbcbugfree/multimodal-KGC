Canonical VisText assets live in `data/images/`, `data/labels/`, `data/json2ttl/`, and `prompt_text/`.

- Use `vistext/data/images/` and `vistext/data/labels/` as the canonical chart inputs.
- Use `vistext/data/json2ttl/json_to_ttl_converter_v2.py` as the maintained local JSON-to-Turtle converter.
- Use the Gemini runners under `vistext/extract_rdf_ttl/` for zeroshot, oneshot, and fewshot extraction experiments.
- Treat `vistext/extract_rdf_ttl/vistext_*_outputs/` as preserved historical outputs rather than source data.
- Treat prompt examples under `vistext/prompt_text/ground_truth_val/` and `vistext/prompt_text/dynamic_oneshot/` as the canonical prompt assets.
