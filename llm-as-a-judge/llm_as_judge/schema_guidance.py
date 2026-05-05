from __future__ import annotations


SCHEMA_GUIDANCE: dict[str, str] = {
    "vistext": """VisText chart-to-RDF schema:
- Namespace: `@prefix : <http://example.org/vistext#> .`
- One chart resource: `:Chart a :BarChart` / `:LineChart` / `:AreaChart`; it may have `:title`, `:xAxis :XAxis`, and `:yAxis :YAxis`.
- Axis resources: `:XAxis` and `:YAxis` are `:Axis` resources and may have visible `:title` literals.
- Data resources: `:DataPointN a :DataPoint`; each data point should use `:xValue`, `:yValue`, and `:belongsTo :Chart`.
- Literal values should reflect visible or reasonably inferred chart text/numeric values. Harmless data-point numbering differences are not semantically important.""",
    "diagram2graph": """Diagram2Graph diagram-to-RDF schema:
- Namespace: `@prefix : <http://example.org/diagram2graph#> .`
- Node resources: `:NodeN a :NodeType, :Node`; node classes include `:Start`, `:Process`, `:Decision`, `:Delay`, and `:Terminator`.
- Node properties: `:label` for visible node text and `:shape` with one of `:StartEvent`, `:EndEvent`, `:Task`, `:Gateway`, or `:DataStore`.
- Edge resources: `:EdgeN a :Solid, :Edge` or `:EdgeN a :Dashed, :Edge`.
- Edge properties: `:source`, `:target`, `:relationshipType` with `:Follows`, `:Branches`, or `:DependsOn`, and optional visible `:relationshipValue`.
- Direct node-to-node relationship triples are outside the target schema; edge resources should encode source, target, style, and relationship semantics.""",
    "soil_health": """Soil-health image-to-RDF schema:
- Prefixes: `she: <https://soilwise-he.github.io/soil-health#>` and `skos: <http://www.w3.org/2004/02/skos/core#>`.
- Visible concepts should be modeled as `she:` resources typed as `skos:Concept`, usually with `skos:prefLabel`.
- Hierarchical or containment relations should use `skos:narrower` / `skos:broader`; general visible associations may use `skos:related`.
- Custom `she:` predicates are acceptable when they name visible domain-specific relations, such as impacts, indicators, arrows, or table-cell semantics.
- String literals should be used for visible values, notes, symbols, abbreviations, definitions, and labels that are not concept nodes.""",
}


def schema_guidance_for(dataset: str) -> str:
    return SCHEMA_GUIDANCE.get(dataset, "No dataset-specific schema guidance is available.")


def prompt_with_schema_guidance(prompt_text: str, dataset: str) -> str:
    return f"{prompt_text.rstrip()}\n\n## Provided Schema Guidance\n\n{schema_guidance_for(dataset)}"
