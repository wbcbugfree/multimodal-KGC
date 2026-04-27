#!/usr/bin/env python3
"""
Convert diagram JSON (nodes/edges) to RDF/Turtle.

Usage:
  # Single file -> single file
  python convert_json_to_ttl.py input.json out.ttl

  # Folder -> folder (outputs mirror names with .ttl extension)
  python convert_json_to_ttl.py path/to/json_folder path/to/out_folder
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import Iterable

def to_camel_case(s: str) -> str:
    parts = re.split(r"[_\s\-]+", (s or "").strip())
    return "".join(p.capitalize() for p in parts if p)

# Vocabulary mappings to match the lightweight diagram2graph TTL schema.
NODE_TYPE_MAP = {
    "start": "Start",
    "process": "Process",
    "decision": "Decision",
    "delay": "Delay",
    "terminator": "Terminator",
}
EDGE_TYPE_MAP = {
    "solid": "Solid",
    "dashed": "Dashed",
}
REL_TYPE_MAP = {
    "follows": "Follows",
    "branches": "Branches",
    "depends_on": "DependsOn",
}

def escape_literal(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )

def to_local_name(prefix: str, value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip())
    token = token.strip("_-")
    return f":{prefix}{token}" if token else f":{prefix}"

def node_subject(index: int) -> str:
    return f":Node{index}"

def node_id_aliases(value: str) -> list[str]:
    raw = value.strip()
    if not raw:
        return []
    aliases = [raw]
    stripped = raw.strip("_-")
    if stripped and stripped != raw:
        aliases.append(stripped)
    if raw.isdigit():
        aliases.append(f"Node{raw}")
    elif raw.lower().startswith("node") and raw[4:].isdigit():
        aliases.append(raw[4:])
    return aliases

def build_node_maps(nodes: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    node_id_map: dict[str, str] = {}
    node_label_map: dict[str, list[str]] = {}
    for index, node in enumerate(nodes, start=1):
        subject = node_subject(index)

        label = str(node.get("label", "")).strip()
        if label:
            node_label_map.setdefault(label, []).append(subject)

        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        if node_id in node_id_map:
            if node_id_map[node_id] != subject:
                raise ValueError(f"Duplicate node id in input JSON: {node_id}")
            continue
        node_id_map[node_id] = subject
        for alias in node_id_aliases(node_id)[1:]:
            node_id_map.setdefault(alias, subject)
    for index, _node in enumerate(nodes, start=1):
        subject = node_subject(index)
        node_id_map.setdefault(str(index), subject)
        node_id_map.setdefault(f"Node{index}", subject)
    return node_id_map, node_label_map

def edge_endpoint_uri(
    node_id_map: dict[str, str],
    node_label_map: dict[str, list[str]],
    edge: dict,
    raw_id: str,
    edge_index: int,
    field: str,
) -> str:
    node_id = raw_id.strip()
    if node_id in node_id_map:
        return node_id_map[node_id]
    for alias in node_id_aliases(node_id):
        if alias in node_id_map:
            return node_id_map[alias]

    endpoint_label = str(edge.get(f"{field}_label", "")).strip()
    label_matches = node_label_map.get(endpoint_label, [])
    if len(label_matches) == 1:
        return label_matches[0]

    if len(label_matches) > 1:
        raise ValueError(
            f"Edge {edge_index} {field} references missing node id {node_id!r}; "
            f"endpoint label {endpoint_label!r} matches multiple nodes"
        )
    raise ValueError(
        f"Edge {edge_index} {field} references missing node id {node_id!r} "
        f"and endpoint label {endpoint_label!r} could not be resolved"
    )

def json_to_ttl_str(data: dict, base_iri: str | None = None, diagram_id: str | None = None) -> str:
    del base_iri, diagram_id

    nodes = data.get("nodes", [])
    node_id_map, node_label_map = build_node_maps(nodes)

    lines: list[str] = []
    lines.append('@prefix : <http://example.org/diagram2graph#> .')
    lines.append("")

    for node_index, node in enumerate(nodes, start=1):
        label = escape_literal(node.get("label") or "")
        type_key = (node.get("type_of_node") or "").lower().strip()
        shape_key = (node.get("shape") or "").lower().strip()

        specific_type = NODE_TYPE_MAP.get(type_key, to_camel_case(type_key) or "Node")
        shape_cls = to_camel_case(shape_key) or "Task"

        subj = node_subject(node_index)
        block = [
            f"{subj} a :{specific_type}, :Node ;",
            f'    :label "{label}" ;',
            f"    :shape :{shape_cls} .",
            "",
        ]
        lines.extend(block)

    for edge_index, e in enumerate(data.get("edges", []), start=1):
        src = str(e.get("source") or e.get("source_") or "").strip()
        tgt = str(e.get("target", "")).strip()
        etype_key = (e.get("type_of_edge") or "").lower().strip()
        rel_key = (e.get("relationship_type") or "").lower().strip()
        rel_val = escape_literal((e.get("relationship_value") or "").strip())

        edge_type = EDGE_TYPE_MAP.get(etype_key, to_camel_case(etype_key) or "Solid")
        rel_type_cap = REL_TYPE_MAP.get(rel_key, to_camel_case(rel_key) or "Follows")

        subj = f":Edge{edge_index}"
        src_uri = edge_endpoint_uri(node_id_map, node_label_map, e, src, edge_index, "source")
        tgt_uri = edge_endpoint_uri(node_id_map, node_label_map, e, tgt, edge_index, "target")

        block = [
            f"{subj} a :{edge_type}, :Edge ;",
            f"    :source {src_uri} ;",
            f"    :target {tgt_uri} ;",
            f"    :relationshipType :{rel_type_cap}" + (" ;" if rel_val else " ."),
        ]
        if rel_val:
            block.append(f'    :relationshipValue "{rel_val}" .')
        block.append("")
        lines.extend(block)

    return "\n".join(lines).rstrip() + "\n"

def convert_one_file(input_json_path: Path, output_ttl_path: Path, base_iri: str | None = None) -> Path:
    with input_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ttl_str = json_to_ttl_str(data, base_iri=base_iri)
    output_ttl_path.parent.mkdir(parents=True, exist_ok=True)
    output_ttl_path.write_text(ttl_str, encoding="utf-8")
    return output_ttl_path

def iter_json_files(folder: Path) -> Iterable[Path]:
    # Only top-level .json files; change to rglob("*.json") if you want recursive
    yield from folder.glob("*.json")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON diagrams to Turtle.")
    parser.add_argument("input_path", help="Input JSON file or folder containing .json files")
    parser.add_argument("output_path", help="Output TTL file (if input is file) OR output folder (if input is folder)")
    parser.add_argument("--base", default=None, help="Ignored legacy option retained for CLI compatibility")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders when input is a folder")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern for input folder mode (default: *.json)")

    args = parser.parse_args()
    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    base = args.base

    if in_path.is_file():
        # Single file -> file (if output is a dir, write as <name>.ttl inside it)
        if out_path.exists() and out_path.is_dir():
            out_file = out_path / (in_path.stem + ".ttl")
        else:
            out_file = out_path
        written = convert_one_file(in_path, out_file, base_iri=base)
        print(str(written))
        sys.exit(0)

    if not in_path.is_dir():
        print(f"Error: Input path does not exist: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Folder mode
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    elif not out_path.is_dir():
        print("Error: When input is a folder, output_path must be a folder.", file=sys.stderr)
        sys.exit(1)

    if args.recursive:
        files = list(in_path.rglob(args.pattern))
    else:
        files = list(in_path.glob(args.pattern))

    if not files:
        print("No JSON files found to convert.", file=sys.stderr)
        sys.exit(2)

    for jf in files:
        if jf.is_file():
            target = out_path / (jf.stem + ".ttl")
            written = convert_one_file(jf, target, base_iri=base)
            print(str(written))
