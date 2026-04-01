Canonical diagram2graph assets live in `data/diagram2graph/`, `data/labels/`, and `json2ttl/`.

- Use `diagram2graph_dataset/data/diagram2graph/` and `diagram2graph_dataset/data/labels/` as the canonical dataset inputs.
- Use `diagram2graph_dataset/json2ttl/script.py` and `diagram2graph_dataset/evaluation/report/plot_summaries.py` as the reusable script entry points.
- Treat `diagram2graph_dataset/extract_rdf_ttl/`, `diagram2graph_dataset/build_kg/`, and `diagram2graph_dataset/prompt_engineering/*_outputs/` as preserved historical experiment outputs.
- Treat `diagram2graph_dataset/extract_rdf_json/`, `diagram2graph_dataset/prompt_engineering/evaluation/`, and `diagram2graph_dataset/evaluation/report/plot_summaries_claude.py` as legacy or staging areas pending later cleanup.
