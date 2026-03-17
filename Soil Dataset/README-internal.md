Canonical Soil assets live in `Data/figures/`, `Data/tables/`, and `Extract All RDFs/`.

- Use `Soil Dataset/Data/figures/` and `Soil Dataset/Data/tables/` as the raw image source of truth.
- Use `Soil Dataset/Extract All RDFs/` as the canonical extraction output location.
- Treat `Soil Dataset/prompt engineering/Prompt Text/` as the reusable prompt asset area.
- Treat `Soil Dataset/Build KG/` as preserved output material rather than a reusable core data source.
- Treat `Soil Dataset/Executability/` and `Soil Dataset/Evaluating/` as legacy or staging areas for checks and evaluation packets.
- Treat `Soil Dataset/Extract RDF Method/` as a legacy experiment area retained for provenance.
- Exact mirrors may be trimmed from staging folders when they duplicate canonical files, but non-canonical or evaluation-specific artifacts should stay labeled rather than being over-cleaned.
