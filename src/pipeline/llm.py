"""Centralised LLM client — all LLM interactions go through this module."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from pipeline.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Accumulates token counts across calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


@dataclass
class LLMClient:
    """Thin wrapper for OpenAI-compatible chat completion API."""

    settings: Settings
    _client: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self.settings.openai_api_key}
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Send a chat completion and return parsed JSON + token counts.

        Parameters
        ----------
        system_prompt:
            The system message.
        user_prompt:
            The user message.
        schema:
            Optional Pydantic model for structured output.

        Returns
        -------
        tuple[dict, int, int]
            (parsed_response, input_tokens, output_tokens)
        """
        effective_user = user_prompt
        if self.settings.llm_think_mode:
            effective_user = "/think\n" + effective_user

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": effective_user},
        ]

        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }

        if schema is not None:
            json_schema = schema.model_json_schema()
            if self.settings.llm_supports_structured_output:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema,
                        "strict": True,
                    },
                }
            else:
                json_instruction = (
                    "\n\nRespond with ONLY a valid JSON object matching this schema "
                    "(no markdown, no explanation):\n"
                    f"```json\n{json.dumps(json_schema, indent=2)}\n```"
                )
                messages[-1]["content"] += json_instruction

        response = self._client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            content = "\n".join(lines).strip()

        parsed: dict[str, Any] = json.loads(content)

        if schema is not None and not self.settings.llm_supports_structured_output:
            schema.model_validate(parsed)

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        return parsed, tokens_in, tokens_out
