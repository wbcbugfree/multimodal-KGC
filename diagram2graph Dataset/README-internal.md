Canonical diagram2graph assets live in `Data/diagram2graph/`, `Data/labels/`, and `JSON2ttl/`.

- Use `diagram2graph Dataset/Data/diagram2graph/` and `diagram2graph Dataset/Data/labels/` as the canonical dataset inputs.
- Use `diagram2graph Dataset/JSON2ttl/Script.py` and `diagram2graph Dataset/Evaluating/report/plot_summaries.py` as the reusable script entry points.
- Treat `diagram2graph Dataset/Extract RDF ttl/`, `diagram2graph Dataset/Build KG/`, and `diagram2graph Dataset/prompt engineering/*_outputs/` as preserved historical experiment outputs.
- Treat `diagram2graph Dataset/Extrcat RDF json/`, `diagram2graph Dataset/prompt engineering/Evaluation/`, and `diagram2graph Dataset/Evaluating/report/plot_summaries Clude.py` as legacy or staging areas pending later cleanup.
