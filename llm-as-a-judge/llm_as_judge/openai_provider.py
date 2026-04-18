from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel

from .schemas import DirectJudgeResult, PairwiseJudgeResult


DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5-mini"
JudgeResultT = TypeVar("JudgeResultT", bound=BaseModel)


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _part_value(part: Any, key: str) -> Any:
    if isinstance(part, dict):
        return part.get(key)
    return getattr(part, key, None)


def _parsed_response(response: Any, result_model: type[JudgeResultT]) -> JudgeResultT:
    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed is not None:
        return output_parsed if isinstance(output_parsed, result_model) else result_model.model_validate(output_parsed)

    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                content = item.get("content", [])
            else:
                content = getattr(item, "content", [])
            for part in content:
                if _part_value(part, "type") == "refusal":
                    raise ValueError(f"OpenAI judge refused the request: {_part_value(part, 'refusal')}")
                parsed = _part_value(part, "parsed")
                if parsed is not None:
                    return parsed if isinstance(parsed, result_model) else result_model.model_validate(parsed)
                text = _part_value(part, "text")
                if isinstance(text, str) and text.strip():
                    return result_model.model_validate_json(text)

    raise ValueError("OpenAI structured response did not contain parsed output.")


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

    def _create(self, *, image_path: Path, text: str, result_model: type[JudgeResultT]) -> Mapping[str, Any]:
        response = self._client_instance().responses.parse(
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
            text_format=result_model,
        )
        return _parsed_response(response, result_model).model_dump()

    def judge_direct(self, *, image_path: Path, ttl_text: str, prompt_text: str) -> Mapping[str, Any]:
        text = f"{prompt_text}\n\nCandidate RDF/Turtle:\n```turtle\n{ttl_text}\n```"
        return self._create(
            image_path=image_path,
            text=text,
            result_model=DirectJudgeResult,
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
            result_model=PairwiseJudgeResult,
        )
