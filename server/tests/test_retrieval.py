"""Semantic search: chunking, incremental sync, the embeddings client, and
the search tool with its term-overlap fallback. Runs on SQLite (the embedding
column degrades to JSON); cosine ordering itself is Postgres-only and covered
by the live check."""

import asyncio
import json

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.graph import build_agent
from app.agent.tools import build_tools
from app.config import Settings
from app.context.chunking import split_into_chunks
from app.context.knowledge import KnowledgeDoc
from app.context.retrieval import sync_knowledge_chunks
from app.gateway.embeddings import EMBED_BATCH_SIZE, OpenAICompatEmbeddings
from app.pipeline.provider import grounded_provider
from app.storage.models import Base, KnowledgeChunkRow
from tests.conftest import fake_llm, text_stream, tool_call_stream


class FakeEmbedder:
    def __init__(self, model_id: str = "fake:v1"):
        self.model_id = model_id
        self.embedded: list[str] = []

    async def embed(self, texts):
        self.embedded += texts
        return [[float(len(text)), 1.0] for text in texts]


@pytest.fixture
def sessions():
    engine = create_async_engine("sqlite+aiosqlite://")

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())
    yield async_sessionmaker(engine, expire_on_commit=False)
    asyncio.run(engine.dispose())


DOC = KnowledgeDoc(
    title="About me",
    slug="about",
    text=(
        "Ruud is an independent data & AI consultant based in Amsterdam.\n\n"
        "## Contact\n\n"
        "- Email: me@ruudjuffermans.nl\n- A free 30-minute intro call.\n\n"
        "Tiny.\n"
    ),
)


def test_split_into_chunks_tracks_headings_and_drops_noise():
    chunks = split_into_chunks(DOC)

    assert [chunk.heading for chunk in chunks] == ["", "Contact"]
    assert chunks[0].content.startswith("Ruud is an independent")
    assert chunks[1].content.startswith("- Email:")
    assert chunks[0].embed_text.startswith("About me\n\n")
    assert chunks[1].embed_text.startswith("About me — Contact\n\n")
    assert chunks[0].content_hash != chunks[1].content_hash


def test_sync_embeds_only_changed_chunks(sessions):
    embedder = FakeEmbedder()

    counts = asyncio.run(sync_knowledge_chunks(sessions, embedder, [DOC]))
    assert counts == {"embedded": 2, "kept": 0, "deleted": 0}

    # Unchanged content: nothing re-embedded.
    counts = asyncio.run(sync_knowledge_chunks(sessions, embedder, [DOC]))
    assert counts == {"embedded": 0, "kept": 2, "deleted": 0}
    assert len(embedder.embedded) == 2

    # One paragraph edited: exactly one chunk re-embedded, its old row deleted.
    edited = KnowledgeDoc(
        title=DOC.title, slug=DOC.slug, text=DOC.text.replace("Amsterdam", "Amsterdam (NL)")
    )
    counts = asyncio.run(sync_knowledge_chunks(sessions, embedder, [edited]))
    assert counts == {"embedded": 1, "kept": 1, "deleted": 1}

    async def rows():
        async with sessions() as session:
            return (await session.execute(select(KnowledgeChunkRow))).scalars().all()

    stored = asyncio.run(rows())
    assert len(stored) == 2
    assert any("Amsterdam (NL)" in row.content for row in stored)


def test_sync_re_embeds_everything_on_model_switch(sessions):
    asyncio.run(sync_knowledge_chunks(sessions, FakeEmbedder("fake:v1"), [DOC]))
    counts = asyncio.run(sync_knowledge_chunks(sessions, FakeEmbedder("fake:v2"), [DOC]))

    assert counts == {"embedded": 2, "kept": 0, "deleted": 2}

    async def models():
        async with sessions() as session:
            rows = (await session.execute(select(KnowledgeChunkRow))).scalars().all()
            return {row.embedding_model for row in rows}

    assert asyncio.run(models()) == {"fake:v2"}


def test_embeddings_client_batches_and_orders_by_index():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            # OpenAI-style: indexed, and here deliberately in reverse order.
            data = [
                {"index": i, "embedding": [float(i)]}
                for i in reversed(range(len(payload["input"])))
            ]
        else:
            # Gemini-style: no index field, input order.
            data = [{"embedding": [9.0]} for _ in payload["input"]]
        return httpx.Response(200, json={"data": data, "model": payload["model"]})

    embedder = OpenAICompatEmbeddings(
        provider="gemini", model="gemini-embedding-001", api_key="k",
        base_url="https://example.test/v1", transport=httpx.MockTransport(handler),
    )
    texts = [f"chunk {i}" for i in range(EMBED_BATCH_SIZE + 1)]
    vectors = asyncio.run(embedder.embed(texts))

    assert embedder.model_id == "gemini:gemini-embedding-001"
    assert [len(r["input"]) for r in requests] == [EMBED_BATCH_SIZE, 1]
    assert vectors[:3] == [[0.0], [1.0], [2.0]]  # first batch, restored to input order
    assert vectors[-1] == [9.0]  # second batch's single, unindexed item


class FakeRetriever:
    def __init__(self, results=None, error=False):
        self.results = results or []
        self.error = error
        self.queries: list[str] = []

    async def search(self, query):
        self.queries.append(query)
        if self.error:
            raise RuntimeError("connection lost")
        return self.results


SNIPPET = {"document": "About me", "heading": "Contact", "snippet": "Email: me@..."}


def test_search_tool_uses_retriever_when_wired():
    retriever = FakeRetriever(results=[SNIPPET])
    tool = build_tools([DOC], retriever)["search_knowledge"]

    assert asyncio.run(tool.run(query="contact email")) == [SNIPPET]
    assert retriever.queries == ["contact email"]


@pytest.mark.parametrize("retriever", [FakeRetriever(), FakeRetriever(error=True)])
def test_search_tool_falls_back_to_term_overlap(retriever):
    # Empty results and a failing vector store both fall back to term overlap.
    tool = build_tools([DOC], retriever)["search_knowledge"]

    results = asyncio.run(tool.run(query="independent consultant Amsterdam"))
    assert results and "independent data & AI consultant" in results[0]["snippet"]


def test_search_tool_without_retriever_stays_sync():
    tool = build_tools([DOC])["search_knowledge"]
    results = tool.run(query="independent consultant Amsterdam")
    assert results and results[0]["document"] == "About me"


def test_graph_executes_async_search_tool():
    llm, recorder = fake_llm([
        tool_call_stream("search_knowledge", {"query": "contact email"}),
        text_stream("His email is on the contact page."),
    ])
    provider = grounded_provider(Settings(llm_provider="gemini", gemini_api_key="k"), [DOC], llm)
    retriever = FakeRetriever(results=[SNIPPET])
    agent = build_agent(
        lambda: provider, [DOC], checkpointer=InMemorySaver(), retriever=retriever
    )

    async def run():
        config = {"configurable": {"thread_id": "t-async-tool"}}
        turns = [{"role": "user", "content": "What is Ruud's email?"}]
        async for _ in agent.astream({"turns": turns}, config, stream_mode="custom"):
            pass

    asyncio.run(run())
    assert retriever.queries == ["contact email"]
    tool_result = recorder.payloads[1]["messages"][-1]
    assert tool_result["role"] == "tool"
    assert "me@..." in tool_result["content"]
