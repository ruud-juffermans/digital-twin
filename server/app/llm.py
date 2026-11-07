"""Provider layer: build the grounded prompt, translate raw API chunks
into simple client events: {type: text|meta|error}.

Pick a provider with LLM_PROVIDER in .env — gemini (default), openai,
deepseek, or kimi. All speak the same chat-completions protocol.
"""

import hashlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from .config import Settings
from .knowledge import KnowledgeDoc
from .openai_client import OpenAICompatClient
from .pricing import cost_usd

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_field": "gemini_api_key", "model_field": "gemini_model",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_field": "openai_api_key", "model_field": "openai_model",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "key_field": "deepseek_api_key", "model_field": "deepseek_model",
    },
    "kimi": {
        "base_url": "https://api.moonshot.ai/v1",
        "key_field": "moonshot_api_key", "model_field": "kimi_model",
    },
}

SYSTEM_PROMPT = """You are "Digital Twin", an AI assistant on Ruud Juffermans's
website, answering visitors' questions about Ruud and his work.

- Ground every factual claim about Ruud in the provided documents. When you
  state a fact from a document, name the document title inline, exactly as
  written (e.g. "Curriculum Vitae", never "his CV").
- If the documents don't contain the answer, say so plainly and suggest
  contacting Ruud directly. Never invent facts.
- Stay on topic: Ruud and his work. Politely decline anything else.
- Use the tools where they answer better than the documents — e.g.
  get_availability for any hiring or availability question, and
  draft_contact_message whenever a visitor asks you to pass a message on to
  Ruud or leaves contact details for him.
- Be concise and friendly."""


def prompt_version() -> str:
    """Short hash of the system prompt: the admin stats group turns by it so
    a prompt edit shows up as a new version."""
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]


def knowledge_version(docs: list[KnowledgeDoc]) -> str:
    """Short hash of the whole knowledge base (titles + text, in load order)."""
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(f"{doc.slug}\n{doc.title}\n{doc.text}\n".encode())
    return digest.hexdigest()[:8]


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

def knowledge_system_prompt(docs: list[KnowledgeDoc]) -> str:
    # The chat-completions protocol has no document blocks; the knowledge
    # base rides in the system prompt as titled markdown sections.
    sections = "\n\n---\n\n".join(f"# {doc.title}\n\n{doc.text}" for doc in docs)
    return f"{SYSTEM_PROMPT}\n\nKnowledge base documents:\n\n{sections}"


class OpenAICompatProvider:
    def __init__(self, settings: Settings, docs: list[KnowledgeDoc],
                 client: OpenAICompatClient | None = None):
        spec = PROVIDERS[settings.llm_provider]
        self._model = getattr(settings, spec["model_field"])
        self._provider = settings.llm_provider
        # OpenAI's current models reject max_tokens in favor of the newer name.
        self._cap = ("max_completion_tokens" if settings.llm_provider == "openai"
                     else "max_tokens")
        self._max_output_tokens = settings.max_output_tokens
        self._titles = [doc.title for doc in docs]
        self._system = knowledge_system_prompt(docs)
        self.client = client or OpenAICompatClient(
            api_key=getattr(settings, spec["key_field"]),
            base_url=spec["base_url"],
        )

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
