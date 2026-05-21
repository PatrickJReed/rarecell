"""Anthropic SDK wrapper for rarecell.

Thin layer over the anthropic Python client that:
  - Loads the rarecell system prompt from system_prompt.md
  - Exposes a `messages_create()` helper returning the raw response dict
  - Defers the actual SDK import to runtime so the package imports
    without the [agent] extra
"""

from __future__ import annotations

from importlib import resources
from typing import Any


def _load_system_prompt() -> str:
    return (resources.files("rarecell.agent") / "system_prompt.md").read_text()


class AnthropicClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-opus-4-7",
        max_tokens: int = 4096,
    ):
        import anthropic

        self._sdk = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = _load_system_prompt()

    def messages_create(self, *, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Thin pass-through to anthropic.messages.create. Returns the raw JSON dict."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools
        response = self._sdk.messages.create(**kwargs)
        return response.model_dump()
