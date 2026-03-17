This folder is a staging area for LLM-judge inputs, not the canonical extraction store.

- Canonical FewShot extractions: `Soil Dataset/Extract All RDFs/FewShot/rdf_extractions.json`
- Canonical OneShot extractions: `Soil Dataset/Extract All RDFs/OneShot/rdf_extractions_oneShot.json`
- `rdf_extractions_FewShot.json` is retained here as a compatibility mirror because the `Bridge/bridge.ipynb` and `Pairwise_Comparison/pairwise.ipynb` staging workflows still expect that exact file name.
- `rdf_extractions_oneShot.json` is currently the shared OneShot staging copy used by the legacy LLM-judge packet and `Soil Dataset/Executability/check_rdf.ipynb`.
- The canonical OneShot extraction file under `Soil Dataset/Extract All RDFs/OneShot/` currently differs from this staging copy, so it is not used as a drop-in replacement yet.
- Prefer the canonical files above for new evaluation or cleanup work.
