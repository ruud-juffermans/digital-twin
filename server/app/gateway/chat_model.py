"""Chat completions over the gateway client: one model turn in, simple
events out ({type: text|citation|turn|meta}).

Gemini, OpenAI, DeepSeek and Kimi all speak the same wire protocol; the
differences (token cap name, cached-token field) are absorbed here so nothing
above the gateway knows which provider is active.
"""

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from ..config import Settings
from .client import OpenAICompatClient
from .pricing import cost_usd
from .providers import PROVIDERS


def _cached_tokens(usage: dict[str, Any]) -> int:
    # Gemini, OpenAI, and Kimi report cache hits in prompt_tokens_details;
    # DeepSeek uses its own top-level field.
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0


def _parse_arguments(raw: str) -> dict[str, Any]:
    # Always json-parse tool input — never string-match. Malformed JSON
    # becomes an empty input, which the tool's schema rejects downstream.
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class OpenAICompatProvider:
    """One chat model behind the configured provider.

    `system_prompt` is the stable prefix every request starts with (the
    context layer builds it); `titles` are the knowledge document titles to
    watch for in the stream so a citation event fires the moment one appears.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        system_prompt: str,
        titles: Sequence[str] = (),
        client: OpenAICompatClient | None = None,
        provider: str | None = None,
    ):
        self._provider = provider or settings.llm_provider
        spec = PROVIDERS[self._provider]
        self._model = getattr(settings, spec["model_field"])
        # OpenAI's current models reject max_tokens in favor of the newer name.
        self._cap = ("max_completion_tokens" if self._provider == "openai"
                     else "max_tokens")
        self._max_output_tokens = settings.max_output_tokens
        self._titles = list(titles)
        self._system = system_prompt
        self.client = client or OpenAICompatClient(
            api_key=getattr(settings, spec["key_field"]),
            base_url=spec["base_url"],
        )

    @property
    def provider(self) -> str:
        return self._provider

    async def stream(
        self,
        turns: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        extension: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream one model turn as events: text, citation, then turn and meta.

        The meta event is one model call's telemetry: tokens, cost, latency,
        time to first token, finish reason and how many retries it took.
        `extension` carries the tool loop's assistant/tool messages after the
        visitor's turns; `tool_choice="none"` forces a text answer.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            self._cap: self._max_output_tokens,
            "messages": [
                {"role": "system", "content": self._system},
                *turns,
                *(extension or []),
            ],
            # ask for the final usage-bearing chunk
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        started = time.monotonic()
        model = self._model
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        first_token_ms: int | None = None
        stats: dict[str, Any] = {}
        answer = ""
        cited: set[str] = set()
        calls: dict[int, dict[str, Any]] = {}
        async for chunk in self.client.stream(payload, stats):
            model = chunk.get("model") or model
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            finish_reason = choices[0].get("finish_reason") or finish_reason
            delta = choices[0].get("delta") or {}
            if first_token_ms is None and (delta.get("content") or delta.get("tool_calls")):
                # Time to first token: the first delta that carries output,
                # text or a tool call, measured from the request (retries included).
                first_token_ms = round((time.monotonic() - started) * 1000)
            if delta.get("content"):
                answer += delta["content"]
                yield {"type": "text", "text": delta["content"]}
                for title in self._titles:
                    if title not in cited and title in answer:
                        cited.add(title)
                        yield {"type": "citation", "title": title}
            for part in delta.get("tool_calls") or []:
                # Arguments stream as JSON fragments; assemble per call index.
                call = calls.setdefault(part.get("index", 0), {"id": "", "name": "", "args": ""})
                call["id"] = part.get("id") or call["id"]
                function = part.get("function") or {}
                call["name"] = function.get("name") or call["name"]
                call["args"] += function.get("arguments") or ""
        yield {
            "type": "turn",
            "text": answer,
            "toolCalls": [
                {
                    "id": call["id"] or f"call_{index}",
                    "name": call["name"],
                    "input": _parse_arguments(call["args"]),
                }
                for index, call in sorted(calls.items())
            ],
            "finishReason": finish_reason,
        }
        input_tokens = usage.get("prompt_tokens") or 0
        output_tokens = usage.get("completion_tokens") or 0
        yield {
            "type": "meta",
            "provider": self._provider,
            "model": model,
            "latencyMs": round((time.monotonic() - started) * 1000),
            "ttftMs": first_token_ms,
            "finishReason": finish_reason,
            "retries": stats.get("retries") or 0,
            "toolCallCount": len(calls),
            "outputChars": len(answer),
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cachedTokens": _cached_tokens(usage),
            "costUsd": round(cost_usd(self._provider, input_tokens, output_tokens), 5),
        }

    async def close(self) -> None:
        await self.client.close()
