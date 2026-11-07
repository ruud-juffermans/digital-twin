# Digital Twin vs. "Building A Generative AI Platform"

Chip Huyen's post ([huyenchip.com, July 2024](https://huyenchip.com/2024/07/25/genai-platform.html))
describes the architecture most teams converge on when they put a generative
AI application in production. It starts from the simplest shape (query → model
→ response) and adds components one at a time, in the order teams usually
need them:

1. Enhance context
2. Put in guardrails
3. Add model router and gateway
4. Reduce latency and cost with cache
5. Add complex logic and write actions

plus two cross-cutting concerns, **observability** and **orchestration**, and
the standing rule that **evaluation is necessary at every step**.

This document maps each of those ideas onto this repository: what is
implemented, where it lives, where the project deliberately deviates, and
what the post suggests we are still missing. File paths are relative to the
repo root. `server/app/` is laid out one folder per component of the post
(`context`, `guardrails`, `gateway`, `actions`, `pipeline`, `observability`,
`storage`, plus `api` for transport); see `docs/PRD_RESTRUCTURE.md`.

---

## 1. At a glance

| Post component | Project implementation | Where | Status |
|---|---|---|---|
| Simplest form: query → model API → response | Streaming `/api/chat` over SSE, one OpenAI-compatible call per model round | `server/app/api/chat.py`, `server/app/gateway/chat_model.py` | Done |
| Context: inline documents | Four knowledge documents ride in the system prompt as titled markdown sections | `server/app/context/prompt.py` (`knowledge_system_prompt`), `server/knowledge/*.py` | Done |
| Context: term-based retrieval | Paragraph ranking by query/paragraph word overlap | `server/app/context/retrieval.py` (`term_search`) | Done (baseline + fallback) |
| Context: embedding-based retrieval | Heading-scoped chunks embedded into pgvector, cosine search | `server/app/context/retrieval.py`, `server/app/gateway/embeddings.py`, migration `0005` | Done (opt-in via `EMBEDDING_PROVIDER`) |
| Context: hybrid search / reranking | Not combined; vector search falls back to term overlap only when it fails or returns nothing | `server/app/context/retrieval.py` (`make_search`) | Gap (low priority at current corpus size) |
| Context: RAG with tabular / structured data | `get_availability` returns a structured dict instead of prose | `server/app/actions/tools/get_availability.py` | Done (minimal form) |
| Context: agentic RAG (retrieval as a tool) | `search_knowledge` is a model-callable tool executed in a graph node | `server/app/actions/tools/search_knowledge.py`, `server/app/pipeline/nodes/execute_tools.py` | Done |
| Context: query rewriting | None; the full history is sent each turn and the model writes the search query itself | `server/app/api/schemas.py` (`ChatRequest`) | Deliberately skipped |
| Input guardrails: PII leakage to external API | Card/account-length digit runs are blocked before any API call | `server/app/guardrails/input.py` (`SENSITIVE_NUMBER`) | Partial (block, no reversible masking) |
| Input guardrails: jailbreak / injection | Phrase filter for injection markers, off-topic regex, system-prompt scope | `server/app/guardrails/input.py` (`screen_input`) | Done (phrase-filter level) |
| Input guardrails: human approval for risky actions | `interrupt()` pauses the graph; Ruud approves via admin endpoint | `server/app/pipeline/nodes/request_approval.py`, `server/app/api/admin.py` | Done |
| Output guardrails: hallucination | Unknown citations dropped; long uncited answers flagged | `server/app/pipeline/nodes/generate.py`, `server/app/guardrails/output.py` | Done (flag, not block) |
| Output guardrails: format / toxicity / AI judge | None | | Gap (free-text answers, low traffic) |
| Failure management: retries | Exponential backoff on 408/409/429/5xx and transport errors, only before the first chunk is delivered | `server/app/gateway/client.py` | Done |
| Failure management: turn limit | `MAX_TOOL_ROUNDS = 3`, then `tool_choice="none"` forces a text answer | `server/app/pipeline/nodes/generate.py` | Done |
| Failure management: human fallback | Approval queue plus "contact Ruud directly" in refusals | `server/app/api/chat.py` (`APPROVAL_NOTICE`), prompts | Done |
| Router: intent classifier | None; routing is deterministic Python, not a model | `server/app/pipeline/graph.py` (`route_after_*`) | Deliberately skipped |
| Router: out-of-scope detection | Stock refusal without an API call for clearly off-topic asks | `server/app/guardrails/input.py` (`OFFTOPIC_PATTERNS`) | Done |
| Router: next-action predictor | The generation model itself decides which tool to call | `server/app/pipeline/nodes/generate.py` | Done (model-native) |
| Gateway: unified layer over many models | One raw Chat Completions client; Gemini, OpenAI, DeepSeek, Kimi via a provider table and `LLM_PROVIDER` | `server/app/gateway/providers.py`, `server/app/gateway/client.py` | Done |
| Gateway: access control and cost management | API keys never leave the server; per-IP rate limit; fail-closed daily budget | `server/app/api/rate_limit.py`, `server/app/actions/budget.py` | Done |
| Gateway: fallback policy to another model | None; after retries the error propagates | `server/app/gateway/client.py` (lands in `gateway/fallback.py`) | Gap |
| Cache: prompt cache | Stable system prompt + documents form the cached prefix; provider cache hits surfaced as `cachedTokens` | `server/app/gateway/chat_model.py` (`_cached_tokens`) | Done (provider-side) |
| Cache: exact cache | Embeddings are content-hashed and only re-computed when text changes | `server/app/context/retrieval.py` (`sync_knowledge_chunks`) | Done for embeddings only |
| Cache: semantic cache | None | | Deliberately skipped (post calls its value "dubious") |
| Complex logic: loops and branching | LangGraph state machine: guard → generate ⇄ tools → verify | `server/app/pipeline/graph.py` | Done |
| Write actions | `draft_contact_message`, `high_risk=True`, never runs without approval, archived on approval | `server/app/actions/tools/draft_contact_message.py`, `server/app/actions/approval.py`, `server/app/api/admin.py` | Done (draft-only; nothing leaves the system) |
| Observability: metrics | Per-turn `meta` event and `turn_log` row: tokens, cached tokens, cost, latency, tools, guard flags | `server/app/gateway/chat_model.py`, `server/app/observability/telemetry.py` (`TurnRecorder`) | Done |
| Observability: logs | Python logging; guard incidents and unanswered questions persisted | `server/app/storage/models.py` (`GuardIncidentRow`, `UnansweredQuestionRow`) | Partial (does not "log everything") |
| Observability: traces | Per-node and per-tool spans streamed as a `trace` event and stored on `turn_log.trace`, with a `model_calls` row per model round | `server/app/observability/tracing.py`, `server/app/observability/telemetry.py` | Done |
| Evaluation at every step | Tool-selection and task-completion datasets, thresholds, failure injection, eval history tables | `server/evals/`, migration `0004` | Done |
| User feedback (post leaves this out) | 👍/👎 per answer with optional comment | `server/app/api/chat.py` (`/api/feedback`), `server/app/observability/feedback.py` | Done |
| Orchestration | LangGraph `StateGraph` with typed `AgentState` and a Postgres or in-memory checkpointer | `server/app/pipeline/graph.py`, `server/app/pipeline/state.py`, `server/app/main.py` | Done |

---

## 2. The architecture progression, mapped to the build phases

The post builds the platform up in steps. `BUILD_GUIDE.md` (in `.claude/docs/`) builds this project up in commits. They line up closely:

| Post stage | Build guide phase / commits |
|---|---|
| Simplest form | Phase 3: raw client (Commit 5), streaming endpoint (Commit 7), React client (Commit 8) |
| Step 1: Enhance context | Phase 2: knowledge modules in the prompt (Commit 4); Commit 25: chunked, embedded, semantically searched knowledge |
| Step 2: Guardrails | Commit 9: validation and rate limiting; Commit 16: deterministic input/output guards; Commit 17: budget guard; Commit 18: approval interrupt; Commit 29: security headers and error-leak audit |
| Step 3: Router and gateway | Commit 5: one client for every OpenAI-compatible provider; `LLM_PROVIDER` switch |
| Step 4: Cache | Commit 10: prompt caching and telemetry; Commit 25: content-hashed embedding sync |
| Step 5: Complex logic and write actions | Phase 7: LangGraph graph (Commit 14), narrow tools (Commit 15), approval for the write tool (Commit 18) |
| Observability | Commit 10: per-message telemetry; Commit 23: turn log, budget ledger, guard incidents; Commit 26: feedback and unanswered questions |
| Orchestration | Commit 14: the graph itself |
| Evaluation | Phase 8: datasets, runner, failure injection (Commits 19–20); Commit 24: eval history |

One difference in ordering: the post puts the gateway in step 3, but this
project built it first (Commit 5), because supporting several providers from
day one is what made cheap local development and cheap eval runs possible.

---

## 3. Step by step

### Step 1: Enhance context

**What the post says.** Context construction is the feature engineering of
foundation models. Chunk documents, retrieve with term-based search (fast,
cheap, a strong baseline) or embeddings (more expensive, improves over time),
combine them when it pays off, and treat retrieval as one of several actions
the model can call. Follow-up questions need rewriting into standalone
queries.

**What the project does.**

- **Inlined core documents.** The four knowledge modules (About me,
  Curriculum Vitae, Projects, Vaardigheden) are joined into the system prompt
  as titled markdown sections in `knowledge_system_prompt`. For a corpus this
  small, stuffing beats retrieving: nothing can be missed, and the prefix is
  stable enough to be cached. The post's framing applies directly: context
  reduces reliance on the model's internal knowledge and keeps answers
  current without retraining.
- **Term-based retrieval as the baseline.** `term_search` in
  `server/app/context/retrieval.py` scores paragraphs by the number of query terms
  they contain. This is the post's "term-based retrieval works well out of the
  box" in its simplest form, and it is what runs when no database is
  configured.
- **Embedding-based retrieval on top.** `server/app/context/retrieval.py` splits each
  document into heading-scoped paragraph chunks, prefixes each with
  `title — heading` before embedding (so a paragraph keeps the context it
  loses when pulled out of its document), hashes the text, and stores vectors
  in pgvector. `SemanticRetriever.search` embeds the query and ranks by cosine
  distance in Postgres. The post's ANN libraries (FAISS, hnswlib) are not
  needed at this scale; the migration leaves a note about adding an HNSW
  index later.
- **Incremental sync.** Only chunks whose text changed are re-embedded, stale
  rows are deleted, and a provider/model switch re-embeds everything because
  `embedding_model` scopes every row. This avoids mixing vector spaces, a
  failure the post does not call out but that follows from its advice to
  evaluate embedding quality.
- **Fallback, not hybrid.** If semantic search throws or returns nothing, the
  tool falls back to term overlap. The post's hybrid, sequential-rerank and
  ensemble patterns are not implemented. At three documents this does not
  matter; it will start to matter when `PORTFOLIO_GUIDE.md` adds a
  retrieval-only corpus of blog posts (Phase 11).
- **Agentic RAG.** `search_knowledge` and `get_availability` are tools the
  model calls; execution happens in the `execute_tools` node, never in the
  model layer. The post's distinction between read-only and write actions is
  encoded directly as the `high_risk` flag on `Tool`.
- **Structured data.** `get_availability` is the post's "tabular RAG" idea
  reduced to its essence: a structured source the model quotes exactly
  instead of paraphrasing prose. There is no text-to-SQL because there is no
  table worth querying.
- **Query rewriting: skipped.** The client sends the complete history every
  turn (`ChatRequest.messages`, up to 40 turns), and the model writes the
  search query itself when it calls `search_knowledge`. For a one-topic
  assistant this is enough; a separate rewriting model would add a round trip
  for little gain.

**Design note worth keeping.** The portfolio guide's rule "core documents are
inlined, posts are retrieved" is the post's inline-versus-retrieve trade-off
stated for this project: the prefix stays cacheable and the citation scan
stays cheap, while a growing corpus reaches the model through retrieval.

### Step 2: Put in guardrails

**What the post says.** Guard inputs (private data leaking to third-party
APIs, jailbreaks) and outputs (empty, malformed, toxic, hallucinated,
sensitive, brand-damaging). Manage failures with retries, parallel calls, and
human fallback. Note the trade-offs: guardrails add latency, output
guardrails are hard in streaming mode, and third-party APIs need the input
side more than self-hosting does.

**Input guards in the project.** All deterministic Python in
`server/app/guardrails/input.py`, run in the `guard_input` node before a single
token is spent:

- **Budget guard first.** When today's spend has reached `DAILY_BUDGET_USD`,
  the agent refuses politely and emits `input:budget`. The post puts cost
  management under the gateway; here it is also the first guard, because an
  exhausted budget is the one failure that should never reach the model.
- **Injection screen.** A phrase list (`INJECTION_MARKERS`) catches the
  obvious "ignore previous instructions" family. This is the post's "simple
  phrase filters work" tier; it does not reach the "AI-based classifier"
  tier.
- **PII screen.** `SENSITIVE_NUMBER` blocks card- and account-length digit
  runs. Phone numbers and emails are allowed on purpose: visitors leave them
  so Ruud can reply. That means contact details do go to a third-party model
  API, which is exactly the leakage the post warns about. The project accepts
  this because the contact tool cannot work otherwise; reversible masking
  (the post's `[PHONE NUMBER]` placeholder trick) would be the way to close
  it if it ever matters.
- **Off-topic screen.** `OFFTOPIC_PATTERNS` short-circuits "write me a Python
  script" style requests with a stock redirect and no API call. Subtler cases
  are left to the system prompt's "stay on topic" instruction.
- **Transport-level guards.** Pydantic caps message count and lengths
  (`server/app/api/schemas.py`), the security middleware refuses bodies over
  256 KB with a 413 and sets a strict CSP (`server/app/api/security.py`), and
  slowapi enforces 20 requests per 10 minutes per IP.

**Output guards in the project.**

- **Unknown citations are dropped.** The provider emits a `citation` event
  when a document title appears in the streamed answer; `generate` only
  forwards titles that exist in the knowledge base and flags the rest as
  `output:unknown-citation:<title>`.
- **Uncited long answers are flagged.** `verify` marks any answer of 200+
  characters that cites nothing, used no tools, and does not read as a
  refusal with `output:uncited-answer`. This is the project's hallucination
  detector, a heuristic stand-in for the post's SelfCheckGPT/SAFE class of
  methods.
- **Streaming trade-off, taken as the post describes it.** Tokens are
  forwarded to the browser as they arrive, so the output guards can only
  flag after the fact, never block. The post states this problem directly;
  the project chooses low latency and logs every flag to `guard_incidents`
  for review instead.
- **Not implemented.** No toxicity scorer, no format validator (answers are
  free text), no AI judge on live traffic, no explicit brand-risk keyword
  monitor. For a one-person site with a daily budget of a couple of dollars
  these are the components the post says you add "when needed", and the need
  has not appeared.

**Failure management.**

- **Retries** live in `OpenAICompatClient.stream`: two retries with
  exponential backoff on transport errors and transient statuses, but only
  while nothing has been delivered. Once a chunk has gone out, a retry would
  duplicate text, so the error propagates. The eval runner's
  `--failure-injection` mode drops every other request and proves these
  retries absorb all of them.
- **Turn limit.** `MAX_TOOL_ROUNDS = 3`; past the cap the model is forced
  to answer in text. This is the post's "turn limit to avoid infinite loops".
- **Tool failures are contained.** Each tool has a timeout; timeouts and
  exceptions become `"Tool timed out."` / `"Tool failed."` tool messages so
  the model can still answer.
- **Human fallback.** High-risk tool calls pause for Ruud's approval, and
  every refusal points the visitor to the contact page. Parallel calls (send
  the query twice, pick the better answer) are not used; they would double
  cost for a marginal gain here.

### Step 3: Add model router and gateway

**What the post says.** A router sends each query to the right solution
(intent classification, out-of-scope detection, next-action prediction). A
gateway is one secure layer in front of all models: access control, cost
management, fallback policies, logging.

**Gateway in the project.** This is one of the project's strongest matches.

- `OpenAICompatClient` is a raw Chat Completions client over httpx; Gemini,
  OpenAI, DeepSeek and Kimi all speak that protocol, so the `PROVIDERS` table
  in `server/app/gateway/providers.py` plus `LLM_PROVIDER` is the whole switch. The same
  pattern covers embeddings (`server/app/gateway/embeddings.py`), with
  `EMBEDDING_PROVIDER` independent of `LLM_PROVIDER` because DeepSeek and
  Kimi have no embeddings endpoint.
- The post's "if a model API changes, only the gateway needs updating" shows
  up concretely: the `max_tokens` versus `max_completion_tokens` rename and
  the differing cached-token fields are absorbed in one place.
- **Access control:** visitors never touch provider keys; the only
  credential exposed is the optional admin bearer token, and the admin
  endpoints are off entirely when it is unset.
- **Cost management:** per-turn cost from flat rates in
  `server/app/gateway/pricing.py` (deliberately over-counting cached tokens), a
  fail-closed daily budget, and a Postgres ledger that makes the cap survive
  restarts.
- **Logging and analytics:** the `meta` event and `turn_log` row carry
  model, tokens, cached tokens, cost and latency.
- **Fallback policy: missing.** When retries are exhausted the request
  fails; it does not fall over to a second provider. The post calls API
  failures "unfortunately common", and this project already has four
  interchangeable providers configured, so a secondary provider on 5xx/429
  would be a small change with a real reliability gain.

**Router in the project.** There is no classifier model. Routing is
deterministic Python in the graph:

- `route_after_guard` sends tripped guards to `refuse` and everything else
  to `generate`.
- `route_after_generate` sends pending tool calls to `request_approval`
  (high-risk, undecided), `execute_tools`, or `verify`.
- Out-of-scope detection is the regex screen plus the system prompt, which
  matches the post's point that stock responses save API calls.
- Next-action prediction is done by the generation model through native
  tool calling. There is no cheap-model/expensive-model split; one model per
  deployment keeps evals comparable.

### Step 4: Reduce latency and cost with cache

**What the post says.** Prompt caches avoid re-processing shared prefixes
(the author expects them to become as common as KV caches). Exact caches
store processed items for repeat requests. Semantic caches reuse answers for
similar queries, but their value is "more dubious" because they depend on a
good embedding, a trustworthy similarity metric and a tuned threshold.

**Project.**

- **Prompt cache: used, provider-side.** The system prompt and the four
  documents form a long stable prefix; the visitor's turns come after it.
  `_cached_tokens` reads the cache-hit count from whichever field the
  provider uses and surfaces it as `cachedTokens`, so cache behaviour is
  observable per turn. The portfolio guide's refusal to inline blog posts is
  a prompt-cache decision: a prefix that changes with every post added, or
  grows past what is "modest", stops paying for itself.
- **Exact cache: embeddings only.** `sync_knowledge_chunks` keys every
  vector by content hash and model id, so unchanged text is never
  re-embedded. This is the post's "cache the result of time-consuming
  actions" applied to the one expensive offline step. There is no exact
  cache of chat answers: with free-form questions and a tiny audience the hit
  rate would be near zero.
- **Semantic cache: skipped, in agreement with the post.** Low traffic, a
  high chance of returning a mismatched cached answer, and the pgvector round
  trip would cost as much as it saved.

### Step 5: Add complex logic and write actions

**What the post says.** Real workflows loop and branch; outputs feed back
into context until the task is done. Write actions unlock automation but
carry real risk, especially under prompt injection. Do not give an unreliable
system authority you would not give an intern.

**Project.**

- **Loop with a hard cap.** `generate` → `execute_tools` → `generate` runs
  up to three rounds; tool results are appended to `loop_messages` and token
  and cost totals are summed across rounds by `_accumulate_meta`.
- **One write action, gated.** `draft_contact_message` is the only tool that
  changes anything. It is marked `high_risk`, and the graph `interrupt()`s
  before it runs. The thread is checkpointed, the visitor is told the request
  is queued, and the action executes only when Ruud posts an approval to
  `/api/admin/approvals/{thread_id}`. Without approval the tool returns a
  declined status and the model finishes its answer anyway. Approved drafts
  are archived in `contact_messages`.
- **Still only a draft.** Even approved, the tool records a message; it does
  not send email or call anything external. The post's "intern" rule is
  applied literally: nothing in the system can act on the outside world.
- **Injection defence in depth.** The input phrase filter catches crude
  attempts; the approval gate means a successful injection still cannot
  cause a write; the `tool_selection` eval dataset checks that no case
  triggers a disallowed tool, with high-risk tools called out explicitly.

### Observability

**What the post says.** Build it in from the start. Track system metrics,
model metrics, failures, lengths, latency (TTFT, TBT, TPS, total), and cost.
Log everything with ids so origin is traceable. Trace each request's full
path so you can pinpoint where it went wrong. Inspect production data daily.

**Project.**

- **Metrics.** Every turn produces a `meta` event (model, `latencyMs`,
  input/output/cached tokens, `costUsd`) and, with a database, a `turn_log`
  row that adds outcome (answered / refused / pending_approval / error),
  tools used and guard flags. `budget_ledger` holds daily spend. Eval runs
  write `eval_runs` and `eval_cases` so regressions show up as history, and
  the thresholds (95% tool selection, 90% task completion) gate the run.
  Metrics can be sliced by provider, model, thread and time with plain SQL.
- **Failure tracking.** Timeouts and tool failures are logged; guard trips
  land in `guard_incidents` with the visitor's message; refusals and honest
  "not in my knowledge base" answers land in `unanswered_questions` tagged
  `guard` or `model`, which is the post's content-gap analysis in table form.
- **Traces.** `traced` wraps every node with a timer, `execute_tools` adds a
  span per tool call, and `verify` streams the list with the guard flags as a
  `trace` event. The same list is stored on `turn_log.trace`, with one
  `model_calls` row per model round (tokens, cost, latency, time to first
  token, finish reason, retries). This is a per-request execution path in
  the post's sense, without LangSmith.
- **Feedback.** `/api/feedback` stores 👍/👎 with the question, answer and an
  optional comment. The post explicitly leaves user feedback out of scope;
  the project has it.
- **Length metrics.** `turn_log` keeps question, context (tool results fed
  back) and answer lengths after the text itself is purged, the post's
  cheapest drift detector.
- **Prompt logging with retention.** Visitor messages, answers, citations and
  tool calls are stored on the turn for `TEXT_RETENTION_DAYS` (default 90),
  then nulled by the retention job while the numbers stay. `0` never stores
  text at all.
- **Where it still falls short of "log everything".**
  - No tokens-per-second; time to first token and total latency only.
  - The admin read API serves the tables; a dashboard over it is a separate
    client concern.

### AI pipeline orchestration

**What the post says.** An orchestrator defines components (models,
retrievers, actions) and chains them. Do not reach for one on day one;
orchestrators add complexity and can hide API calls. When you do pick one,
weigh integration, support for branching and error handling, and
performance.

**Project.** LangGraph's `StateGraph` is the orchestrator, used in the
narrow way the post favours:

- **Components** are plain objects: a provider factory, the `Tool`
  dataclasses from `build_tools`, an optional `SemanticRetriever`, a
  `BudgetTracker` and a checkpointer.
- **Chaining** is explicit edges and three small routing functions. Every
  node is ordinary async Python; only `generate` calls a model, and it does
  so through the project's own httpx client rather than a framework model
  wrapper, so there are no hidden API calls to debug.
- **Branching and error handling** are first class: conditional edges,
  `interrupt()` for the approval pause, and per-thread checkpoints in
  Postgres or memory so a paused run can resume after a restart.
- **Parallelism**, which the post suggests for latency-critical steps, is
  not used. The guards are regex and run in microseconds, so there is
  nothing worth parallelising yet.

The project also followed the post's "build without an orchestrator first"
advice in its history: Phase 3 shipped a working streaming chat with no
graph, and LangGraph arrived in Phase 7 once there was state and branching
to manage.

---

## 4. Where the project knowingly deviates

- **Context stuffing instead of RAG for the core.** Justified by corpus
  size and prompt caching; the portfolio guide switches to retrieval exactly
  where the post would (a corpus too large to inline).
- **Guards are regex, not models.** The post ranks AI classifiers above
  phrase filters. Here the cheaper tier is chosen on purpose: zero latency,
  zero cost, fully testable, and the system prompt handles the long tail.
- **Output guards flag rather than block.** A direct consequence of
  streaming, which the post calls out as the central trade-off.
- **No semantic cache, no parallel calls, no router model.** All three are
  cost/benefit calls that go the post's way for a low-traffic, single-topic
  assistant.
- **Less logging than "log everything".** A privacy choice for a public
  widget on a personal site.

## 5. Gaps the post would ask about, in order of value

1. **Provider fallback in the gateway.** Four compatible providers are
   already configured; falling over to a second one on exhausted retries
   is the post's most direct reliability recommendation and a small change:
   a `server/app/gateway/fallback.py` wired into `gateway/chat_model.py`.
2. **Persist the trace and add first-token latency.** Done (migration
   `0009`): `turn_log.trace`, `turn_log.ttft_ms` and the `model_calls` table.
3. **Length metrics.** Done (migration `0009`): `question_chars`,
   `context_chars` and `answer_chars` on `turn_log`.
4. **Hybrid retrieval before the posts corpus grows.** Combine term overlap
   (or Postgres full-text search) with vector search instead of using one as
   fallback for the other. Only worth it once Phase 11 adds posts.
5. **Sampled offline quality checks.** An AI judge over a daily sample of
   `turn_log` / `unanswered_questions` would cover the post's "spot checks"
   and brand-risk monitoring without touching live latency.
6. **Reversible PII masking.** Only if visitor contact details sent to the
   model API ever become a concern; the post's placeholder-and-dictionary
   approach would fit between `guard_input` and `generate`.
7. **Semantic cache.** Agree with the post: skip unless traffic and
   repeat-rate make it obviously worthwhile.
