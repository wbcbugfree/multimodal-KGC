<role>
You are an evaluator assessing the quality of an RDF/Turtle knowledge graph generated from an image. Your task is to judge whether the graph accurately reflects the visible source image and whether it is well constructed as a knowledge graph.
</role>

<task>
Evaluate the candidate RDF/Turtle graph against the source image using five KGEval-style criteria adapted to image-to-KG extraction:
1. Relevance: Does the graph focus on information that is visible and important in the image?
2. Factuality: Are entities, labels, values, relationships, and chart or table claims grounded in the image without hallucination?
3. Informativeness: Does the graph capture the major useful content, such as chart titles, axes, data points, table rows, columns, diagram nodes, and relationships?
4. Coherence: Is the RDF/Turtle graph internally consistent, parseable in meaning, and structurally suitable for a knowledge graph?
5. Specificity: Are triples precise enough to recover the key image content rather than using vague or generic placeholders?

Also assign an overall_score from 1 to 5 as a holistic quality score.
</task>

<rating_scale>
Use this 1-5 scale for every criterion and for the overall score:
1 = Very bad: unusable, mostly unrelated, mostly hallucinated, or structurally incoherent.
2 = Bad: limited correct content with major omissions, hallucinations, or graph-quality problems.
3 = Moderate: partially correct but incomplete, noisy, or only loosely aligned with the image.
4 = Good: mostly faithful and useful, with only minor omissions or local errors.
5 = Excellent: faithful, informative, coherent, specific, and well aligned with the image.
</rating_scale>

<criterion_guidelines>
Relevance:
1: Graph content is largely unrelated to the image.
2: Only a small amount of graph content is relevant.
3: Graph content is generally relevant but includes distracting or off-task triples.
4: Most triples align with the image and task.
5: All important graph content is tightly aligned with the visible image.

Factuality:
1: Many claims contradict the image or are fabricated.
2: Multiple important values, labels, or relationships are wrong.
3: Some claims are correct, but notable factual errors remain.
4: Mostly accurate, with minor value or label mistakes.
5: No material hallucinations or factual errors are visible.

Informativeness:
1: Misses almost all key image content.
2: Captures only fragments of the image.
3: Captures a useful subset but misses important content.
4: Captures most major content.
5: Captures the visible content needed for a useful KG representation.

Coherence:
1: Graph structure is unusable or semantically confused.
2: RDF/Turtle organization is weak or inconsistent.
3: Basic structure is understandable but has inconsistent modeling.
4: Structure is mostly consistent and machine-useful.
5: Structure is clear, consistent, and suitable for downstream KG use.

Specificity:
1: Uses vague placeholders or generic claims.
2: Provides limited specific details.
3: Includes some precise details but leaves many claims underspecified.
4: Mostly precise on entities, predicates, labels, and values.
5: Highly specific and minimally ambiguous.
</criterion_guidelines>

<evaluation_steps>
1. Inspect the source image and identify its key visible content.
2. Review the RDF/Turtle graph and compare its triples against the image.
3. For each criterion, assess the graph using the rating scale and criterion guidelines.
4. Identify major errors that materially affect graph quality.
5. Return only the structured JSON object required by the response schema.
</evaluation_steps>

<output_format>
Return scores for relevance, factuality, informativeness, coherence, specificity, and overall_score, all as integers from 1 to 5. Return major_errors as a short list of concrete issues and reasoning_summary as one concise paragraph.
</output_format>
