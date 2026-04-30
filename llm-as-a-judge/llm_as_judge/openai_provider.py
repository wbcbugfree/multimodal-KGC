from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel

from .schemas import DirectJudgeResult, PairwiseJudgeResult


DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5-mini"
JudgeResultT = TypeVar("JudgeResultT", bound=BaseModel)


def _json_schema_name(result_model: type[BaseModel]) -> str:
    name = result_model.__name__
    chars: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def structured_text_format(result_model: type[BaseModel]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": _json_schema_name(result_model),
            "schema": result_model.model_json_schema(),
            "strict": True,
        }
    }


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _part_value(part: Any, key: str) -> Any:
    if isinstance(part, dict):
        return part.get(key)
    return getattr(part, key, None)


def _parsed_response(response: Any, result_model: type[JudgeResultT]) -> JudgeResultT:
    output_parsed = response.get("output_parsed") if isinstance(response, Mapping) else getattr(response, "output_parsed", None)
    if output_parsed is not None:
        return output_parsed if isinstance(output_parsed, result_model) else result_model.model_validate(output_parsed)

    output = response.get("output") if isinstance(response, Mapping) else getattr(response, "output", None)
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
        self._client_lock = Lock()

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            from common.config import get_api_key
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key or get_api_key("openai_api_key"))
        return self._client

    def build_request_body(self, *, image_path: Path, text: str, result_model: type[JudgeResultT]) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {"type": "input_image", "image_url": image_to_data_url(image_path)},
                    ],
                }
            ],
            "text": structured_text_format(result_model),
        }

    def _create(self, *, image_path: Path, text: str, result_model: type[JudgeResultT]) -> Mapping[str, Any]:
        response = self._client_instance().responses.parse(
            model=self.model,
            input=self.build_request_body(image_path=image_path, text=text, result_model=result_model)["input"],
            text_format=result_model,
        )
        return _parsed_response(response, result_model).model_dump()

    def direct_text(self, *, ttl_text: str, prompt_text: str) -> str:
        return f"{prompt_text}\n\nCandidate RDF/Turtle:\n```turtle\n{ttl_text}\n```"

    def pairwise_text(
        self,
        *,
        ttl_a: str,
        ttl_b: str,
        prompt_text: str,
        strategy_a: str,
        strategy_b: str,
    ) -> str:
        del strategy_a, strategy_b
        return (
            f"{prompt_text}\n\n"
            f"Candidate A RDF/Turtle:\n```turtle\n{ttl_a}\n```\n\n"
            f"Candidate B RDF/Turtle:\n```turtle\n{ttl_b}\n```"
        )

    def build_direct_request_body(self, *, image_path: Path, ttl_text: str, prompt_text: str) -> dict[str, Any]:
        return self.build_request_body(
            image_path=image_path,
            text=self.direct_text(ttl_text=ttl_text, prompt_text=prompt_text),
            result_model=DirectJudgeResult,
        )

    def build_pairwise_request_body(
        self,
        *,
        image_path: Path,
        ttl_a: str,
        ttl_b: str,
        prompt_text: str,
        strategy_a: str,
        strategy_b: str,
    ) -> dict[str, Any]:
        return self.build_request_body(
            image_path=image_path,
            text=self.pairwise_text(
                ttl_a=ttl_a,
                ttl_b=ttl_b,
                prompt_text=prompt_text,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
            ),
            result_model=PairwiseJudgeResult,
        )

    @staticmethod
    def parse_direct_response_body(body: Mapping[str, Any]) -> Mapping[str, Any]:
        return _parsed_response(body, DirectJudgeResult).model_dump()

    @staticmethod
    def parse_pairwise_response_body(body: Mapping[str, Any]) -> Mapping[str, Any]:
        return _parsed_response(body, PairwiseJudgeResult).model_dump()

    def judge_direct(self, *, image_path: Path, ttl_text: str, prompt_text: str) -> Mapping[str, Any]:
        return self._create(
            image_path=image_path,
            text=self.direct_text(ttl_text=ttl_text, prompt_text=prompt_text),
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
        return self._create(
            image_path=image_path,
            text=self.pairwise_text(
                ttl_a=ttl_a,
                ttl_b=ttl_b,
                prompt_text=prompt_text,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
            ),
            result_model=PairwiseJudgeResult,
        )
