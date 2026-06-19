"""Qwen Cloud memory-agent wrapper."""

from __future__ import annotations

import json
import os
from typing import Any

from src.config import QWEN_BASE_URL, QWEN_MAX_TOKENS, QWEN_MODEL, QWEN_TEMPERATURE
from src.memory import MemoryStore
from src.prompts import SYSTEM_PROMPT, build_user_prompt


class AgentConfigError(RuntimeError):
    """Raised when the agent cannot be configured from the environment."""


class AlphaSignalMemoryAgent:
    """Memory-augmented Qwen Cloud agent for alpha-signal analysis."""

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        model: str = QWEN_MODEL,
        base_url: str = QWEN_BASE_URL,
    ):
        self.memory_store = memory_store or MemoryStore()
        self.model = model
        self.base_url = base_url

    def analyze(
        self,
        text: str,
        source: str = "news",
        persist: bool = True,
        memory_limit: int = 5,
    ) -> dict[str, Any]:
        """Retrieve related memory, call Qwen Cloud, and optionally persist the result."""
        memories = self.memory_store.search(text, limit=memory_limit)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, source, memories)},
        ]

        raw = self._chat(messages)
        try:
            parsed = parse_model_json(raw)
        except json.JSONDecodeError as exc:
            parsed = {
                "error": f"Qwen response was not valid JSON: {exc}",
                "raw_output": raw,
                "memory_updates": [],
            }
        parsed["retrieved_memories"] = memories

        if persist:
            self.memory_store.add(
                text=text,
                source=source,
                title=parsed.get("event_type", ""),
                metadata={"analysis": parsed},
            )

            for update in parsed.get("memory_updates", []):
                if isinstance(update, str) and update.strip():
                    self.memory_store.add(
                        text=update,
                        source="agent_memory_update",
                        title=parsed.get("event_type", ""),
                        metadata={"derived_from_source": source},
                    )

        return parsed

    def _chat(self, messages: list[dict[str, str]]) -> str:
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        if not api_key:
            raise AgentConfigError("Missing DASHSCOPE_API_KEY or QWEN_API_KEY.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentConfigError("Install dependencies with `pip install -r requirements.txt`.") from exc

        client = OpenAI(api_key=api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=QWEN_TEMPERATURE,
            max_tokens=QWEN_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""


def parse_model_json(raw: str) -> dict[str, Any]:
    """Parse JSON from a model response with light cleanup."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
        raise
