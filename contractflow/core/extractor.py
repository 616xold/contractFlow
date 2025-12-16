"""Baseline PDF -> JSON extractor using an LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from contractflow.core.pdf_utils import read_pdf_text


DEFAULT_MODEL = "gpt-5.1"


@dataclass
class ExtractionResult:
    raw_text: str
    json_result: Dict[str, Any]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def load_schema(schema_path: str | Path) -> Dict[str, Any]:
    path = Path(schema_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def schema_to_description(schema: Dict[str, Any]) -> str:
    """Condense the JSON schema into a bullet-friendly description."""
    parts = []
    for field, meta in schema.items():
        desc = meta.get("description", "").strip()
        type_info = meta.get("type")
        enum_vals = meta.get("enum")
        detail = f"{field} ({type_info}"
        if enum_vals:
            detail += f", one of {enum_vals}"
        detail += ")"
        if desc:
            detail += f": {desc}"
        parts.append(detail)
    return "\n".join(parts)


def call_llm_for_schema(
    contract_text: str,
    schema: Dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    client: Optional[OpenAI] = None,
) -> ExtractionResult:
    """Call the LLM to fill the schema from contract text."""
    client = client or OpenAI()
    schema_description = schema_to_description(schema)

    system_prompt = (
        "You are an AI assistant that extracts structured fields from legal contracts. "
        "Return only valid JSON matching the provided schema. Do not add commentary or prose."
    )
    user_prompt = (
        "Here is the contract text:\n"
        f"{contract_text}\n\n"
        "Here is a JSON schema you must fill:\n"
        f"{schema_description}\n\n"
        "Return only valid JSON matching this schema."
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_output_tokens=1500,
    )

    raw_output = _extract_response_text(response)
    parsed = _safe_parse_json(raw_output)

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)

    return ExtractionResult(
        raw_text=raw_output,
        json_result=parsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def extract_fields_naive(pdf_path: str | Path, schema_path: str | Path) -> ExtractionResult:
    """Read a PDF, call the LLM once with the schema, and return parsed JSON."""
    contract_text = read_pdf_text(pdf_path)
    schema = load_schema(schema_path)
    return call_llm_for_schema(contract_text, schema)


def _extract_response_text(response: Any) -> str:
    """Normalize the OpenAI Responses output to plain text."""
    if hasattr(response, "output_text"):
        return response.output_text

    # Responses objects are pydantic models; model_dump is usually available.
    if hasattr(response, "model_dump"):
        data = response.model_dump()
        text = _dig_for_text(data)
        if text:
            return text

    return str(response)


def _dig_for_text(data: Any) -> Optional[str]:
    """Recursively hunt for the first text payload inside the response dict."""
    if isinstance(data, dict):
        for key in ("output_text", "text"):
            if key in data and isinstance(data[key], str):
                return data[key]
        for value in data.values():
            found = _dig_for_text(value)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _dig_for_text(item)
            if found:
                return found
    return None


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    """Try to parse JSON; if invalid, attempt to recover the first JSON object."""
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Attempt to extract the first JSON object from the text.
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError(f"LLM response was not valid JSON: {raw!r}")




