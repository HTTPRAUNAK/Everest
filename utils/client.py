"""
LLM access: Ollama (default) -> optional cloud fallback -> rule-based fallback.

Every node passes a `fallback` callable that produces a sane answer WITHOUT a model.
It is used when MOCK_LLM=true, or when the model keeps returning bad JSON / is down.
So the graph never crashes on a flaky 7-8B model, and the rest of the pipeline is testable offline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeVar

import httpx
from pydantic import BaseModel

import config

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


@dataclass
class LLMOut:
    value: Any
    tokens: int = 0
    calls: int = 0
    model: str = ""


def _ollama(system: str, user: str, json_mode: bool) -> tuple[str, int, str]:
    body: dict[str, Any] = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 384},
    }
    if json_mode:
        body["format"] = "json"
    try:
        r = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=config.LLM_TIMEOUT)
        r.raise_for_status()
        d = r.json()
        return d["message"]["content"], d.get("prompt_eval_count", 0) + d.get("eval_count", 0), config.OLLAMA_MODEL
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise LLMError(f"ollama: {type(e).__name__}") from e


def _cloud(system: str, user: str, json_mode: bool) -> tuple[str, int, str]:
    body: dict[str, Any] = {
        "model": config.CLOUD_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        r = httpx.post(f"{config.CLOUD_URL}/chat/completions", json=body, timeout=config.LLM_TIMEOUT,
                       headers={"Authorization": f"Bearer {config.CLOUD_KEY}"})
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"]["content"], d.get("usage", {}).get("total_tokens", 0), config.CLOUD_MODEL
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise LLMError(f"cloud: {type(e).__name__}") from e


def _complete(system: str, user: str, json_mode: bool) -> tuple[str, int, str]:
    try:
        return _ollama(system, user, json_mode)
    except LLMError:
        if config.CLOUD_FALLBACK and config.CLOUD_URL:
            return _cloud(system, user, json_mode)
        raise


def extract_json(text: str) -> dict:
    """Tolerate ```json fences and chatter around the object."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no json object")
    return json.loads(text[start:end + 1])


def generate_json(system: str, user: str, model_cls: Type[T], fallback: Callable[[], dict]) -> LLMOut:
    """Structured output, validated against a Pydantic model; retry, then fall back to rules."""
    if config.MOCK_LLM:
        return LLMOut(model_cls.model_validate(fallback()), 0, 0, "mock")
    tokens = calls = 0
    for _ in range(1 + config.LLM_RETRIES):
        try:
            text, used, model = _complete(system, user, True)
            calls += 1
            tokens += used
            return LLMOut(model_cls.model_validate(extract_json(text)), tokens, calls, model)
        except (LLMError, ValueError):  # ValidationError and JSONDecodeError are ValueErrors
            continue
    return LLMOut(model_cls.model_validate(fallback()), tokens, calls, "rule-fallback")


def generate_text(system: str, user: str, fallback: Callable[[], str]) -> LLMOut:
    if config.MOCK_LLM:
        return LLMOut(fallback(), 0, 0, "mock")
    tokens = calls = 0
    for _ in range(1 + config.LLM_RETRIES):
        try:
            text, used, model = _complete(system, user, False)
            calls += 1
            tokens += used
            if text.strip():
                return LLMOut(text.strip(), tokens, calls, model)
        except LLMError:
            continue
    return LLMOut(fallback(), tokens, calls, "rule-fallback")
