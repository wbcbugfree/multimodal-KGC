Canonical Soil assets live in `data/figures/`, `data/tables/`, and `extract_all_rdfs/`.

- Use `soil_dataset/data/figures/` and `soil_dataset/data/tables/` as the raw image source of truth.
- Use `soil_dataset/extract_all_rdfs/` as the canonical extraction output location.
- Treat `soil_dataset/prompt_engineering/prompt_text/` as the reusable prompt asset area.
- Treat `soil_dataset/build_kg/` as preserved output material rather than a reusable core data source.
- Treat `soil_dataset/executability/` and `soil_dataset/evaluation/` as legacy or staging areas for checks and evaluation packets.
- Treat `soil_dataset/extract_rdf_method/` as a legacy experiment area retained for provenance.
- Exact mirrors may be trimmed from staging folders when they duplicate canonical files, but non-canonical or evaluation-specific artifacts should stay labeled rather than being over-cleaned.
