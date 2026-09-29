"""Thin, robust wrapper around the Google Gemini API.

Features
* Structured JSON output constrained by a Pydantic schema, then validated.
* Exponential backoff with jitter on rate limits (429), server errors (5xx)
  and malformed/invalid JSON.
* On-disk response cache keyed by (model, settings, prompts, schema), so
  re-running the same analysis costs nothing.
* Thread-safe token-usage accounting for cost/efficiency reporting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .config import LLMConfig

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the LLM call fails permanently."""


class LLMClient(Protocol):
    """Interface the analyzer depends on (lets tests inject a fake)."""

    def generate_json(self, system: str, prompt: str, schema: type[T]) -> T: ...

    def generate_text(self, system: str, prompt: str) -> str: ...


@dataclass
class UsageStats:
    requests: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ResponseCache:
    """Tiny file-based cache: one JSON file per request hash."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        except (OSError, ValueError, KeyError):
            return None

    def set(self, key: str, text: str) -> None:
        tmp = self.directory / f"{key}.tmp"
        tmp.write_text(json.dumps({"text": text}), encoding="utf-8")
        tmp.replace(self.directory / f"{key}.json")  # atomic on the same volume


class GeminiClient:
    """LLMClient implementation backed by google-genai."""

    def __init__(self, api_key: str, config: LLMConfig, cache: ResponseCache | None = None):
        if not api_key:
            raise LLMError("Missing Gemini API key. Set GEMINI_API_KEY in your .env file.")
        self._client = genai.Client(api_key=api_key)
        self.config = config
        self.cache = cache
        self.usage = UsageStats()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def generate_json(self, system: str, prompt: str, schema: type[T]) -> T:
        gen_config = self._build_config(
            system,
            self.config.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )

        def parse(text: str) -> T:
            return schema.model_validate_json(text)

        return self._call(system, prompt, gen_config, parse, cache_tag=schema.__name__)

    def generate_text(self, system: str, prompt: str) -> str:
        gen_config = self._build_config(system, self.config.qa_temperature)
        return self._call(system, prompt, gen_config, lambda text: text, cache_tag="text")

    # ----------------------------------------------------------------- helpers

    def _build_config(self, system: str, temperature: float, **extra) -> types.GenerateContentConfig:
        kwargs = dict(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=self.config.max_output_tokens,
            **extra,
        )
        if self.config.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.config.thinking_budget)
        return types.GenerateContentConfig(**kwargs)

    def _call(self, system, prompt, gen_config, parse, cache_tag):
        cache_key = None
        if self.cache:
            cache_key = ResponseCache.key(
                self.config.model, str(gen_config.temperature), cache_tag, system, prompt
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                try:
                    result = parse(cached)
                    with self._lock:
                        self.usage.cache_hits += 1
                    return result
                except (ValidationError, ValueError):
                    log.warning("Ignoring corrupt cache entry %s", cache_key[:8])

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                delay = self.config.retry_base_delay * 2 ** (attempt - 1) + random.uniform(0, 1)
                log.info("Retry %d/%d in %.1fs (%s)", attempt, self.config.max_retries, delay, last_error)
                with self._lock:
                    self.usage.retries += 1
                time.sleep(delay)
            try:
                response = self._client.models.generate_content(
                    model=self.config.model, contents=prompt, config=gen_config
                )
                self._record_usage(response)
                text = response.text
                if not text:
                    reason = response.candidates[0].finish_reason if response.candidates else "unknown"
                    raise ValueError(f"Empty response (finish_reason={reason})")
                result = parse(text)
            except errors.APIError as exc:
                if exc.code not in RETRYABLE_STATUS:
                    raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
                last_error = exc
            except (ValidationError, ValueError) as exc:
                # Truncated / malformed JSON or schema mismatch: retrying usually fixes it.
                last_error = exc
            else:
                if cache_key:
                    self.cache.set(cache_key, text)
                return result

        raise LLMError(f"LLM call failed after {self.config.max_retries + 1} attempts: {last_error}")

    def _record_usage(self, response) -> None:
        meta = getattr(response, "usage_metadata", None)
        with self._lock:
            self.usage.requests += 1
            if meta:
                self.usage.input_tokens += meta.prompt_token_count or 0
                self.usage.output_tokens += meta.candidates_token_count or 0
