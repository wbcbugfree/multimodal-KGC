This folder is a staging area for LLM-judge inputs, not the canonical extraction store.

- Canonical fewshot extractions: `soil_dataset/extract_all_rdfs/fewshot/rdf_extractions.json`
- Canonical oneshot extractions: `soil_dataset/extract_all_rdfs/oneshot/rdf_extractions_oneshot.json`
- `rdf_extractions_fewshot.json` is retained here as a compatibility mirror because the `bridge/bridge.ipynb` and `pairwise_comparison/pairwise.ipynb` staging workflows still expect that exact file name.
- `rdf_extractions_oneshot.json` is currently the shared oneshot staging copy used by the legacy LLM-judge packet and `soil_dataset/executability/check_rdf.ipynb`.
- The canonical oneshot extraction file under `soil_dataset/extract_all_rdfs/oneshot/` currently differs from this staging copy, so it is not used as a drop-in replacement yet.
- Prefer the canonical files above for new evaluation or cleanup work.
