"""OpenRouter adapter.

OpenRouter exposes an OpenAI-compatible Chat Completions API, so the message /
tool translation and response parsing are identical to the OpenAI adapter — we
just point the `openai` SDK at `https://openrouter.ai/api/v1` and read the
`OPENROUTER_API_KEY` env var. OpenRouter also accepts two optional ranking
headers (`HTTP-Referer`, `X-Title`) that surface your app on openrouter.ai;
they're passed through `default_headers` when provided.
"""

from __future__ import annotations

import os
from typing import Any

from .openai import OpenAIAdapter

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAdapter(OpenAIAdapter):
    """Adapter for OpenRouter's Chat Completions API (reuses the `openai` SDK)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        http_referer: str | None = None,
        x_title: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        import openai  # lazy

        headers: dict[str, str] = dict(client_kwargs.pop("default_headers", {}) or {})
        if http_referer is not None:
            headers["HTTP-Referer"] = http_referer
        if x_title is not None:
            headers["X-Title"] = x_title

        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
            default_headers=headers or None,
            **client_kwargs,
        )
