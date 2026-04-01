#!/usr/bin/env python3
"""
Convert diagram JSON (nodes/edges) to RDF/Turtle.

Usage:
  # Single file -> single file
  python convert_json_to_ttl.py input.json out.ttl --base http://example.org

  # Folder -> folder (outputs mirror names with .ttl extension)
  python convert_json_to_ttl.py path/to/json_folder path/to/out_folder --base http://example.org
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

# Vocabulary mappings to match the correct TTL format
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
    # value in JSON -> (Class used in d2g:relationshipType, direct property name)
    "follows": ("Follows", "follows"),
    "branches": ("Branches", "branches"),
}

def json_to_ttl_str(data: dict, base_iri: str, diagram_id: str) -> str:
    base = f"{base_iri.rstrip('/')}/diagram/{diagram_id}"

    lines: list[str] = []
    lines.append('@prefix d2g: <http://example.org/diagram2graph#> .')
    lines.append('@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n')

    # === Nodes ===
    lines.append("# === Nodes ===")
    for node in data.get("nodes", []):
        nid = str(node.get("id", "")).strip()
        label = (node.get("label") or "").replace('"', '\\"')
        type_key = (node.get("type_of_node") or "").lower().strip()
        shape_key = (node.get("shape") or "").lower().strip()

        specific_type = NODE_TYPE_MAP.get(type_key, to_camel_case(type_key) or "Node")
        shape_cls = to_camel_case(shape_key) or "Task"

        subj = f"<{base}/node/{nid}>"
        block = [
            f"{subj} a d2g:{specific_type}, d2g:Node ;",
            f'    rdfs:label "{label}" ;',
            f"    d2g:shape d2g:{shape_cls} .",
            "",
        ]
        lines.extend(block)

    # === Edges ===
    lines.append("# === Edges ===")
    for e in data.get("edges", []):
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        etype_key = (e.get("type_of_edge") or "").lower().strip()
        rel_key = (e.get("relationship_type") or "").lower().strip()
        rel_val = (e.get("relationship_value") or "").strip()

        edge_type = EDGE_TYPE_MAP.get(etype_key, to_camel_case(etype_key) or "Solid")
        rel_type_cap, rel_pred = REL_TYPE_MAP.get(
            rel_key, (to_camel_case(rel_key) or "Follows", rel_key or "follows")
        )

        # Edge id = "<source><target>" like 12, 23, etc. (fallback to a hash if missing)
        edge_id = f"{src}{tgt}" if src and tgt else f"e{abs(hash((src,tgt)))%100000}"
        subj = f"<{base}/edge/{edge_id}>"
        src_uri = f"<{base}/node/{src}>"
        tgt_uri = f"<{base}/node/{tgt}>"

        block = [
            f"{subj} a d2g:{edge_type}, d2g:Edge ;",
            f"    d2g:source {src_uri} ;",
            f"    d2g:target {tgt_uri} ;",
            f"    d2g:relationshipType d2g:{rel_type_cap}" + (" ;" if rel_val else " ."),
        ]
        if rel_val:
            block.append(f'    d2g:relationshipValue "{rel_val}" .')
        block.append("")
        lines.extend(block)

        # Direct convenience triple (mirrors the relationship)
        lines.append(f"{src_uri} d2g:{rel_pred} {tgt_uri} .")
        lines.append("")

    return "\n".join(lines)

def convert_one_file(input_json_path: Path, output_ttl_path: Path, base_iri: str) -> Path:
    # Infer diagram ID from filename digits; fallback to "diagram"
    m = re.search(r"(\\d+)", input_json_path.stem)
    diagram_id = m.group(1) if m else "diagram"

    with input_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ttl_str = json_to_ttl_str(data, base_iri=base_iri, diagram_id=diagram_id)
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
    parser.add_argument("--base", default="http://example.org", help="Base IRI before /diagram/{id}")
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
