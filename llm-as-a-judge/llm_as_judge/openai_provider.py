from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Mapping

from .schemas import DIRECT_JSON_SCHEMA, PAIRWISE_JSON_SCHEMA


DEFAULT_OPENAI_JUDGE_MODEL = "gpt-4.1-mini"


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content", [])
            else:
                content = getattr(item, "content", [])
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)
    raise ValueError("OpenAI response did not contain output_text.")


def _json_schema_format(name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": name,
        "schema": dict(schema),
        "strict": True,
    }


class OpenAIJudgeProvider:
    def __init__(self, *, model: str | None = None, client: Any | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_JUDGE_MODEL") or DEFAULT_OPENAI_JUDGE_MODEL
        self._client = client
        self._api_key = api_key

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        from common.config import get_api_key
        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key or get_api_key("openai_api_key"))
        return self._client

    def _create(self, *, image_path: Path, text: str, schema_name: str, schema: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self._client_instance().responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {"type": "input_image", "image_url": image_to_data_url(image_path)},
                    ],
                }
            ],
            text={"format": _json_schema_format(schema_name, schema)},
        )
        parsed = json.loads(_response_text(response))
        if not isinstance(parsed, Mapping):
            raise ValueError("OpenAI structured output must be a JSON object.")
        return parsed

    def judge_direct(self, *, image_path: Path, ttl_text: str, prompt_text: str) -> Mapping[str, Any]:
        text = f"{prompt_text}\n\nCandidate RDF/Turtle:\n```turtle\n{ttl_text}\n```"
        return self._create(
            image_path=image_path,
            text=text,
            schema_name="direct_image_to_kg_judge",
            schema=DIRECT_JSON_SCHEMA,
        )

    def judge_pairwise(
        self,
        *,
        image_path: Path,
        ttl_a: str,
        ttl_b: str,
        prompt_text: str,
        strategy_a: str,
        strategy_b: str,
    ) -> Mapping[str, Any]:
        text = (
            f"{prompt_text}\n\n"
            f"Candidate A strategy: {strategy_a}\n```turtle\n{ttl_a}\n```\n\n"
            f"Candidate B strategy: {strategy_b}\n```turtle\n{ttl_b}\n```"
        )
        return self._create(
            image_path=image_path,
            text=text,
            schema_name="pairwise_image_to_kg_judge",
            schema=PAIRWISE_JSON_SCHEMA,
        )
