import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_agent
from app.config import Settings
from app.context.knowledge import load_knowledge
from app.gateway.client import OpenAICompatClient
from app.pipeline.provider import grounded_provider
from evals.run import FlakyTransport, eval_tool_selection, load_dataset, run_case
from evals.scoring import looks_like_refusal, score_completion, score_tool_selection
from tests.conftest import fake_llm, fake_transport, text_stream, tool_call_stream


def test_score_tool_selection():
    ok, detail = score_tool_selection(["get_availability"], ["get_availability"], [])
    assert ok and detail == {"missing": [], "extra": []}

    ok, detail = score_tool_selection(["get_availability"], [], [])
    assert not ok and detail["missing"] == ["get_availability"]

    # search_knowledge is an allowed extra; a high-risk extra is not.
    ok, _ = score_tool_selection([], ["search_knowledge"], ["search_knowledge"])
    assert ok
    ok, detail = score_tool_selection([], ["draft_contact_message"], ["search_knowledge"])
    assert not ok and detail["extra"] == ["draft_contact_message"]


def test_score_completion_checks():
    case = {"must_contain_any": ["Leiden"], "must_cite": True}
    assert score_completion(case, "He studied in Leiden.", 1) == (True, [])

    ok, failures = score_completion(case, "He studied somewhere.", 1)
    assert not ok and "missing all of" in failures[0]

    ok, failures = score_completion(case, "Leiden.", 0)
    assert not ok and failures == ["no citations"]

    refusal_case = {"expect_refusal": True, "must_not_contain": ["Google engineer"]}
    ok, _ = score_completion(refusal_case, "I don't have that in the knowledge base.", 0)
    assert ok
    ok, failures = score_completion(refusal_case, "Yes, he was a Google engineer.", 0)
    assert not ok and len(failures) == 2


def test_looks_like_refusal():
    assert looks_like_refusal("I don't have information on that — contact Ruud directly.")
    assert not looks_like_refusal("Ruud studied in Leiden.")


def test_datasets_are_well_formed():
    tools = load_dataset("tool_selection")
    completion = load_dataset("task_completion")
    assert len(tools["cases"]) >= 10
    assert len(completion["cases"]) >= 8
    assert len({case["id"] for case in tools["cases"]}) == len(tools["cases"])
    assert len({case["id"] for case in completion["cases"]}) == len(completion["cases"])


def eval_agent(llm):
    """A real graph over the real knowledge base, backed by a fixture transport."""
    docs = load_knowledge(Settings().knowledge_dir)
    provider = grounded_provider(Settings(llm_provider="gemini", gemini_api_key="k"), docs, llm)

    async def _run(fn):
        agent = build_agent(lambda: provider, docs, checkpointer=InMemorySaver())
        return await fn(agent)

    return _run


def test_run_case_collects_tools_text_and_cost():
    llm, _ = fake_llm([
        tool_call_stream("get_availability", {}),
        text_stream("Available from now."),
    ])
    run = eval_agent(llm)

    outcome = asyncio.run(run(lambda agent: run_case(agent, "Available?", "t-eval-1")))

    assert outcome["tools"] == ["get_availability"]
    assert outcome["text"] == "Available from now."
    assert outcome["errors"] == []
    assert outcome["costUsd"] > 0


def test_run_case_records_high_risk_selection_and_rejects():
    # The interrupt pauses the run; run_case must record the selected tool and
    # resolve the pause by rejecting, never executing the high-risk action.
    llm, recorder = fake_llm([
        tool_call_stream(
            "draft_contact_message",
            {"subject": "s", "message": "m", "sender_contact": "a@b.c"},
        ),
        text_stream("I couldn't send that."),
    ])
    run = eval_agent(llm)

    outcome = asyncio.run(run(lambda agent: run_case(agent, "Contact Ruud", "t-eval-2")))

    assert outcome["tools"] == ["draft_contact_message"]
    declined_result = recorder.payloads[1]["messages"][-1]
    assert declined_result["role"] == "tool"
    assert '"declined"' in declined_result["content"]


def test_eval_tool_selection_end_to_end_scoring():
    # Every case answered with plain text -> only the no-tool cases pass;
    # proves the dataset wiring and accuracy math without a live model.
    llm, _ = fake_llm([text_stream("An answer.")])
    run = eval_agent(llm)

    summary = asyncio.run(run(lambda agent: eval_tool_selection(agent, "test")))

    dataset = load_dataset("tool_selection")
    expected_passes = sum(1 for case in dataset["cases"] if case["expected_tools"] == [])
    assert summary["passed"] == expected_passes
    assert summary["total"] == len(dataset["cases"])
    assert summary["met"] is False  # tool-needing cases failed, as they should

def test_failure_injection_is_absorbed_by_retries():
    # Every other request dies at the transport; the client's retries must
    # hide that completely from the graph — no error events, a full answer.
    flaky = FlakyTransport(inner=fake_transport(text_stream("Still here.")))
    llm = OpenAICompatClient(
        api_key="k", base_url="https://example.test/v1", transport=flaky, retry_base_delay=0,
    )
    run = eval_agent(llm)

    outcome = asyncio.run(run(lambda agent: run_case(agent, "Who is Ruud?", "t-eval-3")))

    assert outcome["errors"] == []
    assert outcome["text"] == "Still here."
    assert flaky.injected == 1 and flaky.calls == 2
