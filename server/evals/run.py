"""Eval runner: drive the real agent graph over the labeled datasets.

Runs against whichever provider .env configures — pick a cheap one
(LLM_PROVIDER=gemini or deepseek) for routine runs.

    uv run python -m evals.run                       # both datasets
    uv run python -m evals.run --dataset completion  # one dataset
    uv run python -m evals.run --failure-injection   # drop every other request
    uv run python -m evals.run --no-strict           # report, don't gate

Thresholds: tool selection >= 95%, task completion >= 90%.
Exit code 1 when a threshold is missed (unless --no-strict). With failure
injection every dropped request must be absorbed by the client's retries —
an unrecovered failure fails its case, and so the threshold. With
DATABASE_URL set, every run is also recorded in eval_runs / eval_cases.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph import build_agent
from app.config import Settings, get_settings
from app.gateway.client import OpenAICompatClient
from app.gateway.providers import PROVIDERS
from app.knowledge import load_knowledge
from app.observability.evals import record_eval_run
from app.pipeline.provider import grounded_provider
from app.storage.engine import create_engine_and_sessions, run_migrations

from .scoring import score_completion, score_tool_selection

DATASETS_DIR = Path(__file__).parent / "datasets"
TOOL_SELECTION_THRESHOLD = 0.95
COMPLETION_THRESHOLD = 0.90


class FlakyTransport(httpx.AsyncBaseTransport):
    """Failure injection: drop every other request at the socket.

    Wraps a real transport (or a fixture one in tests). The client's
    retry/backoff must absorb every injected failure for the run to pass —
    recovery proven across the real datasets, not a mock.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()
        self.calls = 0
        self.injected = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls % 2 == 1:
            self.injected += 1
            raise httpx.ConnectError("injected failure", request=request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def make_client(
    settings: Settings, transport: httpx.AsyncBaseTransport | None
) -> OpenAICompatClient:
    spec = PROVIDERS[settings.llm_provider]
    return OpenAICompatClient(
        api_key=getattr(settings, spec["key_field"]), base_url=spec["base_url"], transport=transport
    )

async def run_case(agent: Any, question: str, thread_id: str) -> dict[str, Any]:
    """One question through the graph; auto-rejects approval interrupts."""
    config = {"configurable": {"thread_id": thread_id}}
    events: list[dict[str, Any]] = []

    async for event in agent.astream(
        {"turns": [{"role": "user", "content": question}]}, config, stream_mode="custom"
    ):
        events.append(event)

    snapshot = await agent.aget_state(config)
    tools = list(snapshot.values.get("tools_used") or [])
    if snapshot.interrupts:
        # The selection itself is what's being scored; resolve the pause by
        # rejecting so no high-risk action executes during evals.
        payload = snapshot.interrupts[0].value or {}
        tools += [call.get("name") for call in payload.get("tool_calls") or []]
        async for event in agent.astream(
            Command(resume={"approved": False, "note": "eval run"}), config, stream_mode="custom"
        ):
            events.append(event)

    metas = [event for event in events if event.get("type") == "meta"]
    return {
        "text": "".join(event.get("text", "") for event in events if event.get("type") == "text"),
        "citations": sum(1 for event in events if event.get("type") == "citation"),
        "tools": tools,
        "errors": [event["message"] for event in events if event.get("type") == "error"],
        "costUsd": round(sum(meta.get("costUsd") or 0.0 for meta in metas), 5),
        "latencyMs": sum(meta.get("latencyMs") or 0 for meta in metas),
    }


def load_dataset(name: str) -> dict[str, Any]:
    return json.loads((DATASETS_DIR / f"{name}.json").read_text(encoding="utf-8"))


async def eval_tool_selection(agent: Any, run_id: str) -> dict[str, Any]:
    dataset = load_dataset("tool_selection")
    allowed_extra = dataset.get("allowed_extra") or []
    results = []
    for case in dataset["cases"]:
        outcome = await run_case(agent, case["question"], f"eval-{run_id}-tools-{case['id']}")
        passed, detail = score_tool_selection(
            case["expected_tools"], outcome["tools"], allowed_extra
        )
        results.append({**_base_row(case, outcome, passed), "detail": detail})
        _print_row(results[-1])
    return _summarize("tool_selection", results, TOOL_SELECTION_THRESHOLD)


async def eval_task_completion(agent: Any, run_id: str) -> dict[str, Any]:
    dataset = load_dataset("task_completion")
    results = []
    for case in dataset["cases"]:
        outcome = await run_case(agent, case["question"], f"eval-{run_id}-task-{case['id']}")
        passed, failures = score_completion(case, outcome["text"], outcome["citations"])
        results.append({**_base_row(case, outcome, passed), "detail": failures})
        _print_row(results[-1])
    return _summarize("task_completion", results, COMPLETION_THRESHOLD)


def _base_row(case: dict[str, Any], outcome: dict[str, Any], passed: bool) -> dict[str, Any]:
    return {
        "id": case["id"],
        "passed": passed and not outcome["errors"],
        "tools": outcome["tools"],
        "errors": outcome["errors"],
        "costUsd": outcome["costUsd"],
        "latencyMs": outcome["latencyMs"],
    }


def _print_row(row: dict[str, Any]) -> None:
    status = "PASS" if row["passed"] else "FAIL"
    detail = "" if row["passed"] else f"  {row['detail']}{row['errors']}"
    print(f"  [{status}] {row['id']}  (${row['costUsd']:.4f}, {row['latencyMs']}ms){detail}")


def _summarize(name: str, results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    passed = sum(1 for row in results if row["passed"])
    accuracy = passed / len(results) if results else 0.0
    summary = {
        "dataset": name,
        "passed": passed,
        "total": len(results),
        "accuracy": round(accuracy, 3),
        "threshold": threshold,
        "met": accuracy >= threshold,
        "costUsd": round(sum(row["costUsd"] for row in results), 5),
        "results": results,
    }
    print(
        f"{name}: {passed}/{len(results)} = {accuracy:.1%} "
        f"(threshold {threshold:.0%}, {'MET' if summary['met'] else 'MISSED'}, "
        f"${summary['costUsd']:.4f} total)\n"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agent eval suite.")
    parser.add_argument("--dataset", choices=["tools", "completion", "all"], default="all")
    parser.add_argument("--failure-injection", action="store_true")
    parser.add_argument("--no-strict", action="store_true", help="never exit non-zero")
    parser.add_argument("--report", default=str(Path(__file__).parent / "report.json"))
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    docs = load_knowledge(settings.knowledge_dir)
    flaky = FlakyTransport() if args.failure_injection else None
    provider = grounded_provider(settings, docs, client=make_client(settings, flaky))
    print(
        f"provider={settings.llm_provider} model={settings.active_model} "
        f"failure_injection={flaky is not None}\n"
    )

    run_id = "run"
    summaries: list[dict[str, Any]] = []
    agent = build_agent(lambda: provider, docs, checkpointer=InMemorySaver())
    if args.dataset in ("tools", "all"):
        summaries.append(await eval_tool_selection(agent, run_id))
    if args.dataset in ("completion", "all"):
        summaries.append(await eval_task_completion(agent, run_id))
    await provider.close()

    report: dict[str, Any] = {
        "provider": settings.llm_provider,
        "model": settings.active_model,
        "failure_injection": None,
        "datasets": summaries,
    }
    if flaky is not None:
        errored = any(row["errors"] for summary in summaries for row in summary["results"])
        recovered = 0 if errored else flaky.injected
        report["failure_injection"] = {"injected": flaky.injected, "recovered": recovered}
        print(
            f"failure injection: {flaky.injected} failures injected, "
            f"{'all recovered' if recovered else 'NOT all recovered'} by retry\n"
        )
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report written to {args.report}")

    if settings.database_url:
        # report.json is only the latest snapshot; the database keeps one row
        # per run and per case, so regressions show up as history.
        await asyncio.to_thread(run_migrations, settings.database_url)
        engine, sessions = create_engine_and_sessions(settings.database_url)
        try:
            run_pk = await record_eval_run(sessions, report)
            print(f"eval history: recorded run #{run_pk}")
        finally:
            await engine.dispose()

    if args.no_strict:
        return 0
    return 0 if all(summary["met"] for summary in summaries) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
