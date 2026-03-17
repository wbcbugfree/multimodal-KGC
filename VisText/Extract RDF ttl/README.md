# VisText Output Provenance

The preserved output folders in this directory contain valuable historical LLM-generated TTL results and are intentionally kept unchanged during the conservative refactor.

Important provenance note:

- The `manifest.json` files inside `vistext_Zeroshot_outputs/`, `vistext_Oneshot_outputs/`, and `vistext_Fewshot_outputs/` still record the original Kaggle-era source paths such as `sub_image_1000` and `labels_1000`.
- Those manifest paths are historical provenance metadata for the original run, not the new canonical in-repo input locations.
- The canonical local VisText inputs after refactor are:
  - `VisText/Data/images/`
  - `VisText/Data/labels/`

Do not rewrite the manifests unless you are explicitly regenerating the corresponding outputs.
