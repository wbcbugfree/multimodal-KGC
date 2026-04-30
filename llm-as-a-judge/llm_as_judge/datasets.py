from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
VIS_TEXT_STRATEGIES = {
    "zeroshot": Path("vistext/extract_rdf_ttl/vistext_zeroshot_outputs"),
    "oneshot_static": Path("vistext/extract_rdf_ttl/vistext_oneshot_static_outputs"),
    "oneshot_dynamic": Path("vistext/extract_rdf_ttl/vistext_oneshot_dynamic_outputs"),
    "fewshot": Path("vistext/extract_rdf_ttl/vistext_fewshot_outputs"),
}
SOIL_HEALTH_STRATEGIES = {
    "zeroshot": Path("soil_health/extract_rdf_ttl/zeroshot"),
    "oneshot_static": Path("soil_health/extract_rdf_ttl/oneshot_static"),
    "oneshot_dynamic": Path("soil_health/extract_rdf_ttl/oneshot_dynamic"),
    "fewshot": Path("soil_health/extract_rdf_ttl/fewshot"),
}
DIAGRAM2GRAPH_STRATEGIES = {
    "zeroshot": Path("diagram2graph/extract_rdf_ttl/zeroshot_outputs"),
    "oneshot": Path("diagram2graph/extract_rdf_ttl/oneshot_outputs"),
    "fewshot": Path("diagram2graph/extract_rdf_ttl/fewshot_outputs"),
}
GOLD_STRATEGY = "ground_truth"
GOLD_TTL_DIRS = {
    "vistext": Path("vistext/data/test/turtle"),
    "diagram2graph": Path("diagram2graph/data/turtle"),
}


@dataclass(frozen=True)
class CandidateRecord:
    dataset: str
    strategy: str
    item_id: str
    ttl_path: Path
    image_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "strategy": self.strategy,
            "item_id": self.item_id,
            "ttl_path": str(self.ttl_path),
            "image_path": str(self.image_path),
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def judge_root() -> Path:
    return repo_root() / "llm-as-a-judge"


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def result_path(*parts: str | Path) -> Path:
    path = judge_root() / "results"
    for part in parts:
        path /= part
    resolved = path.resolve()
    if not is_relative_to(resolved, judge_root()):
        raise ValueError(f"Result path escapes llm-as-a-judge: {resolved}")
    return resolved


def strategy_dirs(dataset: str) -> dict[str, Path]:
    if dataset == "vistext":
        return dict(VIS_TEXT_STRATEGIES)
    if dataset == "soil_health":
        return dict(SOIL_HEALTH_STRATEGIES)
    if dataset == "diagram2graph":
        return dict(DIAGRAM2GRAPH_STRATEGIES)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _absolute_strategy_dirs(dataset: str, root: Path | None = None) -> dict[str, Path]:
    base = root or repo_root()
    return {name: base / relative for name, relative in strategy_dirs(dataset).items()}


def resolve_image(dataset: str, item_id: str, root: Path | None = None) -> Path:
    base = root or repo_root()
    if dataset == "vistext":
        search_roots = [base / "vistext" / "data" / "test" / "images"]
    elif dataset == "soil_health":
        search_roots = [base / "soil_health" / "data" / "figures", base / "soil_health" / "data" / "tables"]
    elif dataset == "diagram2graph":
        search_roots = [base / "diagram2graph" / "data" / "images"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for search_root in search_roots:
        for extension in IMAGE_EXTENSIONS:
            candidate = search_root / f"{item_id}{extension}"
            if candidate.exists():
                return candidate.resolve()
    for search_root in search_roots:
        if search_root.exists():
            matches = sorted(path for path in search_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == item_id)
            if matches:
                return matches[0].resolve()
    raise FileNotFoundError(f"No image found for {dataset}:{item_id}")


def collect_ttl_records(
    dataset: str,
    *,
    strategies: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    root: Path | None = None,
) -> list[CandidateRecord]:
    selected = list(strategies or strategy_dirs(dataset).keys())
    id_filter = set(ids or [])
    records: list[CandidateRecord] = []
    for strategy in selected:
        strategy_path = _absolute_strategy_dirs(dataset, root).get(strategy)
        if strategy_path is None:
            raise ValueError(f"Unknown {dataset} strategy: {strategy}")
        if not strategy_path.exists():
            continue
        for ttl_path in sorted(strategy_path.glob("*.ttl"), key=lambda path: path.stem):
            item_id = ttl_path.stem
            if id_filter and item_id not in id_filter:
                continue
            try:
                image_path = resolve_image(dataset, item_id, root=root)
            except FileNotFoundError:
                continue
            records.append(
                CandidateRecord(
                    dataset=dataset,
                    strategy=strategy,
                    item_id=item_id,
                    ttl_path=ttl_path.resolve(),
                    image_path=image_path,
                )
            )
    return records


def collect_gold_records(
    dataset: str,
    *,
    ids: Sequence[str] | None = None,
    root: Path | None = None,
    strategy_name: str = GOLD_STRATEGY,
) -> list[CandidateRecord]:
    base = root or repo_root()
    relative_gold_dir = GOLD_TTL_DIRS.get(dataset)
    if relative_gold_dir is None:
        raise ValueError(f"No labelled RDF/Turtle ground truth is configured for dataset: {dataset}")
    gold_dir = base / relative_gold_dir
    if not gold_dir.exists():
        raise FileNotFoundError(f"Ground-truth TTL directory not found: {gold_dir}")
    id_filter = set(ids or [])
    records: list[CandidateRecord] = []
    for ttl_path in sorted(gold_dir.glob("*.ttl"), key=lambda path: path.stem):
        item_id = ttl_path.stem
        if id_filter and item_id not in id_filter:
            continue
        try:
            image_path = resolve_image(dataset, item_id, root=root)
        except FileNotFoundError:
            continue
        records.append(
            CandidateRecord(
                dataset=dataset,
                strategy=strategy_name,
                item_id=item_id,
                ttl_path=ttl_path.resolve(),
                image_path=image_path,
            )
        )
    return records


def sample_records(
    records: Iterable[CandidateRecord],
    *,
    sample_mode: str = "all",
    sample_count: int | None = None,
    ids: Sequence[str] | None = None,
    seed: int = 42,
) -> list[CandidateRecord]:
    record_list = list(records)
    if sample_mode == "all":
        return record_list
    if sample_mode == "ids":
        id_order = list(ids or [])
        id_rank = {item_id: index for index, item_id in enumerate(id_order)}
        filtered = [record for record in record_list if record.item_id in id_rank]
        return sorted(filtered, key=lambda record: (id_rank[record.item_id], record.strategy))
    if sample_mode == "random":
        rng = random.Random(seed)
        grouped = group_by_item(record_list)
        item_ids = sorted(grouped)
        count = min(sample_count or len(item_ids), len(item_ids))
        sampled_ids = set(rng.sample(item_ids, count))
        sampled = [record for item_id in item_ids if item_id in sampled_ids for record in grouped[item_id]]
        return sorted(sampled, key=lambda record: (record.item_id, record.strategy))
    raise ValueError(f"Unsupported sample mode: {sample_mode}")


def group_by_item(records: Iterable[CandidateRecord]) -> dict[str, list[CandidateRecord]]:
    grouped: dict[str, list[CandidateRecord]] = {}
    for record in records:
        grouped.setdefault(record.item_id, []).append(record)
    return {item_id: sorted(items, key=lambda record: record.strategy) for item_id, items in sorted(grouped.items())}
