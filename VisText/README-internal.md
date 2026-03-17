Canonical VisText assets live in `Data/images/`, `Data/labels/`, and `Data/Json2ttl/`.

- Use `VisText/Data/images/` and `VisText/Data/labels/` as the current source of truth for local inputs.
- Use `VisText/Data/Json2ttl/` and `VisText/Evaluation/json_to_ttl_converter_v2.py` as the reusable conversion entry points.
- Treat `VisText/Extract RDF ttl/vistext_*_outputs/` and `VisText/Evaluation/gold_ttl_1000/` as preserved historical outputs rather than reusable source inputs.
- Treat `VisText/Evaluation/complete_data.py` as a manual staging utility, not a reusable pipeline component.
