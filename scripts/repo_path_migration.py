from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


DATASET_ROOTS = {
    "VisText",
    "Soil Dataset",
    "diagram2graph Dataset",
}
NOTEBOOK_SUFFIX = ".ipynb"
FULL_REWRITE_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".txt",
}
SEGMENT_REWRITE_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
}
EXACT_STEM_RENAMES = {
    "plot_summaries Clude": "plot_summaries_claude",
    "vistext_Oneshot_sta_outputs": "vistext_oneshot_static_outputs",
    "vistext_Oneshot_dyn_outputs": "vistext_oneshot_dynamic_outputs",
    "vistext_oneshot_sta_outputs": "vistext_oneshot_static_outputs",
    "vistext_oneshot_dyn_outputs": "vistext_oneshot_dynamic_outputs",
}
TOKEN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"(?i)extrcat", "extract"),
    (r"(?i)inferance", "inference"),
    (r"(?i)onehot", "oneshot"),
    (r"(?i)fewshto", "fewshot"),
    (r"(?i)sepratly", "separately"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize dataset paths to lowercase underscore-separated names.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to refactor.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, the command prints a dry-run summary.",
    )
    return parser.parse_args(argv)


def canonicalize_token(token: str) -> str:
    working = token.strip()
    if not working:
        return working

    working = EXACT_STEM_RENAMES.get(working, working)
    for pattern, replacement in TOKEN_REPLACEMENTS:
        working = re.sub(pattern, replacement, working)

    working = re.sub(r"[ -]+", "_", working)
    working = re.sub(r"_+", "_", working)
    return working.lower()


def canonicalize_segment(name: str, is_file: bool = False) -> str:
    if not is_file:
        return canonicalize_token(name)

    path = Path(name)
    suffix = path.suffix.lower()
    stem = name[: -len(path.suffix)] if path.suffix else name
    normalized_stem = canonicalize_token(stem)
    return f"{normalized_stem}{suffix}"


def canonicalize_relative_path(path: PurePosixPath) -> PurePosixPath:
    parts = list(path.parts)
    normalized_parts = [
        canonicalize_segment(part, is_file=index == len(parts) - 1)
        for index, part in enumerate(parts)
    ]
    return PurePosixPath(*normalized_parts)


def list_tracked_files(repo_root: Path) -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        cwd=repo_root,
        capture_output=True,
    )
    entries = [entry.decode("utf-8") for entry in result.stdout.split(b"\x00") if entry]
    return [PurePosixPath(entry) for entry in entries]


def is_dataset_path(path: PurePosixPath) -> bool:
    return bool(path.parts) and path.parts[0] in DATASET_ROOTS


def build_file_move_map(tracked_files: Iterable[PurePosixPath]) -> dict[PurePosixPath, PurePosixPath]:
    move_map: dict[PurePosixPath, PurePosixPath] = {}
    target_map: dict[PurePosixPath, PurePosixPath] = {}

    for old_path in tracked_files:
        if not is_dataset_path(old_path):
            continue
        new_path = canonicalize_relative_path(old_path)
        if new_path == old_path:
            continue
        existing = target_map.get(new_path)
        if existing is not None and existing != old_path:
            raise ValueError(f"Collision: {existing} and {old_path} both map to {new_path}")
        target_map[new_path] = old_path
        move_map[old_path] = new_path
    return move_map


def build_directory_move_map(paths: Iterable[PurePosixPath]) -> dict[PurePosixPath, PurePosixPath]:
    move_map: dict[PurePosixPath, PurePosixPath] = {}
    for path in paths:
        current = path.parent
        while current != PurePosixPath("."):
            if is_dataset_path(current):
                new_path = canonicalize_relative_path(current)
                if current != new_path:
                    move_map[current] = new_path
            current = current.parent
    return move_map


def sort_paths_by_depth(paths: Iterable[PurePosixPath]) -> list[PurePosixPath]:
    return sorted(paths, key=lambda item: (len(item.parts), str(item)), reverse=True)


def safe_move_file(repo_root: Path, old_path: PurePosixPath, new_path: PurePosixPath) -> None:
    old_abs = repo_root / Path(old_path)
    new_abs = repo_root / Path(new_path)
    if old_abs == new_abs:
        return
    if not old_abs.exists() and new_abs.exists():
        return

    new_abs.parent.mkdir(parents=True, exist_ok=True)
    if old_abs.parent == new_abs.parent and old_abs.name.lower() == new_abs.name.lower():
        temp_name = f".__rename_tmp__{uuid.uuid4().hex}{old_abs.suffix.lower()}"
        temp_abs = old_abs.with_name(temp_name)
        old_abs.rename(temp_abs)
        temp_abs.rename(new_abs)
        return

    if new_abs.exists():
        raise FileExistsError(f"Target already exists: {new_abs}")
    old_abs.rename(new_abs)


def apply_file_moves(repo_root: Path, move_map: dict[PurePosixPath, PurePosixPath]) -> None:
    for old_path in sort_paths_by_depth(move_map):
        safe_move_file(repo_root, old_path, move_map[old_path])


def remove_empty_directories(repo_root: Path, directory_map: dict[PurePosixPath, PurePosixPath]) -> None:
    for old_dir in sort_paths_by_depth(directory_map):
        old_abs = repo_root / Path(old_dir)
        if old_abs.exists():
            try:
                old_abs.rmdir()
            except OSError:
                continue


def build_full_string_replacements(
    repo_root: Path,
    file_map: dict[PurePosixPath, PurePosixPath],
    directory_map: dict[PurePosixPath, PurePosixPath],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    combined = {**directory_map, **file_map}
    for old_rel, new_rel in combined.items():
        old_posix = old_rel.as_posix()
        new_posix = new_rel.as_posix()
        pairs.append((old_posix, new_posix))
        pairs.append((str(Path(old_rel)), str(Path(new_rel))))

        old_abs = repo_root / Path(old_rel)
        new_abs = repo_root / Path(new_rel)
        pairs.append((old_abs.as_posix(), new_abs.as_posix()))
        pairs.append((str(old_abs), str(new_abs)))

    return deduplicate_replacements(pairs)


def build_segment_replacements(
    file_map: dict[PurePosixPath, PurePosixPath],
    directory_map: dict[PurePosixPath, PurePosixPath],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    combined = {**directory_map, **file_map}
    for old_path, new_path in combined.items():
        for old_part, new_part in zip(old_path.parts, new_path.parts):
            if old_part != new_part:
                pairs.append((old_part, new_part))
    return deduplicate_replacements(pairs)


def deduplicate_replacements(pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    unique: dict[str, str] = {}
    for old, new in pairs:
        if old and old != new:
            unique[old] = new
    return sorted(unique.items(), key=lambda item: len(item[0]), reverse=True)


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)
    return updated


def rewrite_notebook_payload(
    payload: dict[str, Any],
    full_replacements: list[tuple[str, str]],
    segment_replacements: list[tuple[str, str]],
) -> dict[str, Any]:
    def rewrite_value(value: Any) -> Any:
        if isinstance(value, str):
            updated = apply_replacements(value, full_replacements)
            updated = apply_replacements(updated, segment_replacements)
            return updated
        if isinstance(value, list):
            return [rewrite_value(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite_value(item) for key, item in value.items()}
        return value

    rewritten = rewrite_value(payload)
    for cell in rewritten.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return rewritten


def rewrite_notebook_file(
    path: Path,
    full_replacements: list[tuple[str, str]],
    segment_replacements: list[tuple[str, str]],
) -> bool:
    original = json.loads(path.read_text(encoding="utf-8"))
    rewritten = rewrite_notebook_payload(original, full_replacements, segment_replacements)
    if rewritten == original:
        return False
    path.write_text(json.dumps(rewritten, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def rewrite_text_file(
    path: Path,
    full_replacements: list[tuple[str, str]],
    segment_replacements: list[tuple[str, str]],
) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = apply_replacements(original, full_replacements)
    if path.suffix.lower() in SEGMENT_REWRITE_SUFFIXES:
        updated = apply_replacements(updated, segment_replacements)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def iter_rewrite_targets(
    tracked_files: Iterable[PurePosixPath],
    file_map: dict[PurePosixPath, PurePosixPath],
) -> list[Path]:
    seen: set[Path] = set()
    targets: list[Path] = []
    for old_path in tracked_files:
        effective = Path(file_map.get(old_path, old_path))
        if effective in seen:
            continue
        seen.add(effective)
        targets.append(effective)
    return sorted(targets)


def rewrite_repo_contents(
    repo_root: Path,
    tracked_files: Iterable[PurePosixPath],
    file_map: dict[PurePosixPath, PurePosixPath],
    full_replacements: list[tuple[str, str]],
    segment_replacements: list[tuple[str, str]],
) -> tuple[int, int]:
    text_updates = 0
    notebook_updates = 0
    for relative_path in iter_rewrite_targets(tracked_files, file_map):
        absolute_path = repo_root / relative_path
        suffix = absolute_path.suffix.lower()
        if suffix == NOTEBOOK_SUFFIX:
            if rewrite_notebook_file(absolute_path, full_replacements, segment_replacements):
                notebook_updates += 1
        elif suffix in FULL_REWRITE_SUFFIXES:
            if rewrite_text_file(absolute_path, full_replacements, segment_replacements):
                text_updates += 1
    return text_updates, notebook_updates


def write_summary_doc(
    repo_root: Path,
    file_map: dict[PurePosixPath, PurePosixPath],
    directory_map: dict[PurePosixPath, PurePosixPath],
) -> None:
    summary_path = repo_root / "docs" / "path_normalization_map.md"
    key_paths = [
        ("VisText", "vistext"),
        ("Soil Dataset", "soil_dataset"),
        ("diagram2graph Dataset", "diagram2graph_dataset"),
        ("VisText/Extract RDF ttl", "vistext/extract_rdf_ttl"),
        ("VisText/Prompt Text", "vistext/prompt_text"),
        ("Soil Dataset/prompt engineering", "soil_dataset/prompt_engineering"),
        ("diagram2graph Dataset/Extrcat RDF json", "diagram2graph_dataset/extract_rdf_json"),
    ]
    changed_files = sorted(file_map.items(), key=lambda item: str(item[0]))
    lines = [
        "# Path Normalization Map",
        "",
        "This repo now uses lowercase underscore-separated dataset paths.",
        "",
        "## Key Directory Changes",
        "",
    ]
    for old_path, new_path in key_paths:
        lines.append(f"- `{old_path}` -> `{new_path}`")

    lines.extend(
        [
            "",
            "## Changed File Count",
            "",
            f"- Renamed files: {len(file_map)}",
            f"- Renamed directories: {len(directory_map)}",
            "",
            "## Representative File Renames",
            "",
        ]
    )
    for old_path, new_path in changed_files[:20]:
        lines.append(f"- `{old_path.as_posix()}` -> `{new_path.as_posix()}`")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_dry_run_summary(
    file_map: dict[PurePosixPath, PurePosixPath],
    directory_map: dict[PurePosixPath, PurePosixPath],
) -> None:
    print(f"Files to rename: {len(file_map)}")
    print(f"Directories to rename: {len(directory_map)}")
    for old_path, new_path in list(sorted(file_map.items(), key=lambda item: str(item[0])))[:20]:
        print(f"  {old_path.as_posix()} -> {new_path.as_posix()}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    tracked_files = list_tracked_files(repo_root)
    file_map = build_file_move_map(tracked_files)
    directory_map = build_directory_move_map(file_map)

    if not args.apply:
        print_dry_run_summary(file_map, directory_map)
        return 0

    apply_file_moves(repo_root, file_map)
    remove_empty_directories(repo_root, directory_map)

    full_replacements = build_full_string_replacements(repo_root, file_map, directory_map)
    segment_replacements = build_segment_replacements(file_map, directory_map)
    text_updates, notebook_updates = rewrite_repo_contents(
        repo_root=repo_root,
        tracked_files=tracked_files,
        file_map=file_map,
        full_replacements=full_replacements,
        segment_replacements=segment_replacements,
    )
    write_summary_doc(repo_root, file_map, directory_map)

    print(f"Renamed files: {len(file_map)}")
    print(f"Renamed directories: {len(directory_map)}")
    print(f"Rewritten text files: {text_updates}")
    print(f"Rewritten notebooks: {notebook_updates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
