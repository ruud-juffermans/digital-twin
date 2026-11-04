# Digital Twin observability vs. OpenTelemetry GenAI observability

The OpenTelemetry blog post
[GenAI Observability with OpenTelemetry](https://opentelemetry.io/blog/2026/genai-observability/)
is a walkthrough: it wires VS Code Copilot (with OpenAI Codex and Claude
Code as further examples) to an OTLP backend (Aspire Dashboard) and shows
what the **OpenTelemetry Semantic Conventions for Generative AI** give you
out of the box. Its central argument is that an agent request is a chain of
model calls, tool invocations and token exchanges, and that without a
standard shape for recording that chain you cannot tell whether a slow
answer was the model, a slow tool, or a retry loop.

The post leans on the GenAI semantic conventions (now maintained at
`github.com/open-telemetry/semantic-conventions-genai`). Where this document
cites exact span, attribute, metric or event names, they come from those
conventions, which the post demonstrates rather than defines.

This document compares what the post and the conventions expect with what
this project records today. File paths are relative to the repo root.

---

## 1. Summary

The project captures most of the **data** the post cares about (model,
tokens, cached tokens, duration, tool names, finish reason, failures) but
in a **bespoke shape**: a `meta` SSE event, a `trace` SSE event, and
Postgres rows written by `TurnRecorder`. Nothing is emitted as
OpenTelemetry spans, metrics or events, so none of it can be sent to an
OTLP backend or compared across tools the way the post shows.

| Aspect | Post / conventions | Project | Verdict |
|---|---|---|---|
| Transport | OTLP to any compatible backend | SSE to the browser; SQL rows in Postgres | Different; not interoperable |
| Request as a tree of spans (`invoke_agent` → `chat` → `execute_tool`) | Yes | Flat per-node list `[{node, ms}]`, per-run, not nested | Partial |
| Model and provider identity | `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.provider.name` | `meta.model` (response model), `turn_log.provider`, `turn_log.model` | Equivalent data |
| Token usage | `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens` | `inputTokens`, `outputTokens`, `cachedTokens` | Equivalent data |
| Finish reason | `gen_ai.response.finish_reasons` | Captured in the internal `turn` event, not recorded anywhere | Gap (trivial) |
| Operation duration | `gen_ai.client.operation.duration` histogram, span duration per call | `latencyMs` summed across model rounds; per-node ms in `trace` | Partial (no per-call breakdown persisted) |
| Time to first token | `gen_ai.response.time_to_first_chunk` on streaming spans | Not measured | Gap |
| Tool execution | `execute_tool {name}` span with `gen_ai.tool.name`, `gen_ai.tool.call.id`, duration, `error.type` | `tool` event with the name; `tools_used` list; timeout/failure logged, not timed | Partial |
| Retrieval | `retrieval` span, `gen_ai.retrieval.top_k`, `gen_ai.retrieval.duration` | Not instrumented; happens inside the tool call | Gap |
| Embeddings | `embeddings` operation span and duration | Startup log line with chunk counts only | Gap |
| Conversation id | `gen_ai.conversation.id` when readily available | `thread_id` on every row and every SSE stream | Equivalent data |
| Error classification | `error.type` on span and metrics | `turn_log.outcome = "error"`; exception text in logs | Partial (no error type) |
| Content capture | Opt-in, off by default (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`) | Prompts never logged; answers stored only in `unanswered_questions` and `feedback` | Same stance, different mechanism |
| Agent-level metrics | `gen_ai.invoke_agent.duration`, `.inference_calls`, `.tool_calls` | `tool_rounds` and `tools_used` in state; not aggregated as metrics | Partial |
| Cost | Post: estimate from tokens × model | `costUsd` computed per turn from flat rates, summed in `budget_ledger` | Project goes further |
| Evaluation results | `gen_ai.evaluation.result` event parented to the evaluated span | `eval_runs` / `eval_cases` tables, offline only; runtime guard flags are evaluations without a standard shape | Different shape |
| Dashboard | Aspire Dashboard (free, Docker, OTLP) | None; tables only | Gap |

---

## 2. What the post expects, mapped to the code

### 2.1 Span structure

The post's example trace is three levels deep:

- `invoke_agent` for the whole agent run
- `chat` for each model call (the conventions name it `{gen_ai.operation.name} {gen_ai.request.model}`, kind `CLIENT`)
- `execute_tool {gen_ai.tool.name}` for each tool invocation (kind `INTERNAL`)

The project has the same decomposition in its code, but not in its telemetry:

- The agent run is one `astream` call in `server/app/chat.py` (`stream_sse`). There is no span for it. The closest equivalent is the `turn_log` row, which carries totals for the run.
- Each model call is one `OpenAICompatProvider.stream` call in `server/app/llm.py`. It times itself with `time.monotonic()` and emits one `meta` event per call, but `generate` in `server/app/agent/graph.py` sums those into one total via `_accumulate_meta`. After that, the per-call breakdown is gone. A run with three model rounds and two tool calls is recorded as one latency number.
- Each tool call runs in `execute_tools`. It emits `{"type": "tool", "name": ...}` before running and appends the name to `tools_used`. Duration, arguments, result size and failure are not recorded, only logged on timeout or exception.
- The `_traced` wrapper gives per-node timing (`guard_input`, `generate`, `execute_tools`, `verify`), which is a flat list in the order nodes ran, not a tree, and the whole `generate` node is one entry even when it contains a slow model call. For the post's headline question ("was it the model, the tool, or a retry loop?") the list answers "which node" but not "which call inside the node", and the retry loop in `OpenAICompatClient.stream` is invisible: the conventions say retries are included in the span duration, but the project records neither the retry count nor the per-attempt time.

### 2.2 Attributes

| Convention attribute | Project field | Where | Notes |
|---|---|---|---|
| `gen_ai.provider.name` (`openai`, `deepseek`, `gcp.gemini`, `moonshot_ai`) | `turn_log.provider` (`openai`, `deepseek`, `gemini`, `kimi`) | `server/app/db.py` (`TurnRecorder`) | Same information, project-specific values |
| `gen_ai.request.model` | `Settings.active_model`, stored as `turn_log.model` | `server/app/config.py` | Request model |
| `gen_ai.response.model` | `meta.model` (taken from the response chunk) | `server/app/llm.py` | The project records the response model on the event and the request model in the row; the two can differ (aliases) and are never stored together |
| `gen_ai.usage.input_tokens` | `inputTokens` | `server/app/llm.py` | Includes cached tokens, as the convention requires |
| `gen_ai.usage.output_tokens` | `outputTokens` | same | |
| `gen_ai.usage.cache_read.input_tokens` | `cachedTokens` | same, `_cached_tokens` | Normalised across providers' differing field names, which is exactly the normalisation the conventions do |
| `gen_ai.usage.cache_write.input_tokens` | none | | Not exposed by the chat-completions usage object the project reads |
| `gen_ai.response.finish_reasons` | `finishReason` on the internal `turn` event | `server/app/llm.py` | Used for control flow only; never written to `turn_log` or `meta`. Losing this means you cannot tell a `length`-truncated answer from a `stop` one after the fact |
| `gen_ai.request.max_tokens` | `max_output_tokens` setting | `server/app/config.py` | Config only, not recorded per turn |
| `gen_ai.request.stream` | always true | | Implicit |
| `gen_ai.conversation.id` | `thread_id` | everywhere | Present on every row and every SSE stream; the conventions warn against synthesising one, and the project does generate a UUID when the client sends none, which is acceptable for the thread but would not be a legitimate conversation id in the convention's sense for a one-off request |
| `gen_ai.tool.name` | `tool` event name, `tools_used` | `server/app/agent/graph.py` | |
| `gen_ai.tool.call.id` | `call["id"]` | same | Present in state, not recorded |
| `gen_ai.tool.call.arguments` / `.result` (opt-in) | not recorded | | Approved contact drafts are archived in `contact_messages`, which is the one place tool arguments persist |
| `gen_ai.retrieval.top_k` | `MAX_SEARCH_RESULTS = 3` | `server/app/retrieval.py` | Constant, not recorded |
| `gen_ai.retrieval.query.text` / `.documents` (opt-in) | not recorded | | The retrieved snippets go back to the model as a tool message and vanish |
| `error.type` | none | | `outcome = "error"` plus a stack trace in the log; no low-cardinality error class (timeout vs 429 vs 5xx vs tool failure) |
| `gen_ai.agent.name` | none | | A single agent, so the name would be constant |

### 2.3 Metrics

The post shows two histograms in the Aspire Dashboard: operation duration
filtered by model, and token usage filtered by token type. The conventions
have since split these into a larger set. The table lists what the project
could populate from data it already has.

| Convention metric | Unit | Project source | Status |
|---|---|---|---|
| `gen_ai.client.inference.usage.input_tokens` (counter) | `{token}` | `turn_log.input_tokens` | Data exists per run, not per call |
| `gen_ai.client.inference.usage.output_tokens` (counter) | `{token}` | `turn_log.output_tokens` | Same |
| `gen_ai.client.inference.usage.cache_read.input_tokens` (counter) | `{token}` | `turn_log.cached_tokens` | Same |
| `gen_ai.client.inference.operation.input_tokens` (histogram, per call) | `{token}` | per-call `meta` before aggregation | Lost at `_accumulate_meta` |
| `gen_ai.client.operation.duration` (histogram) | `s` | `turn_log.latency_ms` (per run, ms) | Different granularity and unit |
| `gen_ai.response.time_to_first_chunk` (span attribute) | `s` | none | Not measured; the provider stream knows when the first `delta.content` arrives |
| `gen_ai.server.time_per_output_token` | `s` | derivable as `latencyMs / outputTokens` per call | Not computed |
| `gen_ai.invoke_agent.duration` | `s` | `turn_log.latency_ms` approximates it, but excludes guard and verify time | Partial |
| `gen_ai.invoke_agent.inference_calls` | `{inference_call}` | `tool_rounds + 1` in state | Not recorded |
| `gen_ai.invoke_agent.tool_calls` | `{tool_call}` | `len(tools_used)` | Recorded as a list, so derivable |
| `gen_ai.execute_tool.duration` | `s` | none | Tool runs are not timed |
| `gen_ai.retrieval.duration` | `s` | none | |
| `gen_ai.client.embeddings.duration` | `s` | none | Startup sync logs counts, not time |

The post's "catch token-hungry prompts before production" use case needs
per-call input token histograms by model. The project's `evals/report.json`
and `eval_cases` rows record cost and latency per case, which covers the
pre-production half of that use case through evals rather than metrics.

### 2.4 Content capture and privacy

The post is explicit that content capture is opt-in and that by default
"no prompt content or tool arguments are captured". The conventions put
`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`,
`gen_ai.tool.definitions`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`,
`gen_ai.retrieval.query.text` and `gen_ai.retrieval.documents` behind a
single switch, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`, with
`NO_CONTENT` as the default.

The project's stance is the same, reached without a switch:

- Prompts, the system prompt and tool definitions are never logged or stored.
- Answers are stored in two narrow cases: `unanswered_questions` (the
  visitor's question and the answer, when a guard refused or the model
  declined) and `feedback` (question and answer when the visitor clicks
  👍/👎). `guard_incidents` stores the visitor's message when a guard trips.
- Tool arguments persist only for approved `draft_contact_message` calls, in
  `contact_messages`, because that is the product feature.

Two differences worth noting:

1. The project has **no way to turn content capture on** for debugging. When
   an answer is wrong, there is no record of the prompt that produced it or
   the retrieved snippets the model saw. The post's answer is a flag you flip
   in a dev environment; here it would be a code change.
2. The project's stored content is **selected by outcome**, not by a global
   setting. That is arguably better for a public widget: the interesting
   failures are kept, the routine traffic is not.

### 2.5 Sessions and conversations

The post mentions sessions only through Codex. The conventions attach
`gen_ai.conversation.id` to inference, agent and tool spans when an id is
readily available. The project is stronger here than most: `thread_id` is
the LangGraph checkpoint key, so the same id links the SSE stream, every
`turn_log` row, guard incidents, feedback, pending approvals and archived
contact messages. A per-thread history is one SQL query. What is missing is
the per-call level underneath it (no span or row per model call or tool
call), so within a thread you can see turns but not what each turn did
internally.

### 2.6 Evaluation

The conventions define a `gen_ai.evaluation.result` event with
`gen_ai.evaluation.name`, `gen_ai.evaluation.score.value`,
`gen_ai.evaluation.score.label` and `gen_ai.evaluation.explanation`,
parented to the span being evaluated.

The project evaluates in two places, neither in that shape:

- **Offline:** `server/evals/run.py` scores tool selection and task
  completion and writes `eval_runs` / `eval_cases`. Each case row has a
  pass/fail and a detail object, which maps cleanly onto
  `score.label = pass|fail` and `explanation = detail`. There is no link
  from an eval case to a production turn, by design: evals run against the
  graph directly.
- **Runtime:** the guard flags (`output:uncited-answer`,
  `output:unknown-citation:<title>`, `input:injection`, ...) are evaluations
  of individual turns and are stored on `turn_log.guard_flags` and
  `guard_incidents`. In convention terms each flag is an evaluation result
  with a name and a `fail` label, attached to the turn it judged. The
  project already has this correlation through `thread_id` and the row.

### 2.7 Backend and dashboards

The post's practical payoff is a dashboard you did not build: Aspire
Dashboard renders traces, filters metrics by `gen_ai.request.model`, and a
GenAI visualizer turns captured messages into a chat view. The project has
seven telemetry tables and no view over any of them. Every question the post
answers with a click (latency by model, tokens by type, slow tool calls)
is a SQL query here, and some (per-call latency, TTFT, tool duration) cannot
be answered because the data is not stored.

---

## 3. Where the project is ahead of the post

- **Cost is computed, not estimated.** The post stops at "estimate per-request
  cost from token counts and model". The project prices every turn in
  `server/app/pricing.py`, sums it into `budget_ledger`, and enforces a daily
  cap that fails closed. The conventions have no cost attribute at all.
- **Cache visibility across providers.** `_cached_tokens` normalises four
  providers' cache-hit fields into one number, which is the same job the
  conventions do with `gen_ai.usage.cache_read.input_tokens`, done before
  the conventions reached this project.
- **Outcome taxonomy.** `turn_log.outcome` (`answered`, `refused`,
  `pending_approval`, `error`) and the `guard` / `model` reason on
  `unanswered_questions` are product-level signals the generic conventions
  do not have. Keep them as custom attributes if the project ever adopts OTel.
- **User feedback.** `/api/feedback` ties a visitor's 👍/👎 to a thread. The
  post does not touch feedback.
- **Gated evals with history.** Thresholds, failure injection and
  `eval_runs` history give regression detection the post does not cover.
- **Telemetry never breaks the request.** `TurnRecorder.record` is
  best-effort and swallows its own errors. That is the right default and
  matches how OTel exporters behave.

---

## 4. Where the project falls short

In rough order of how much each would help answer "why was this answer slow
or wrong":

1. **No per-call breakdown.** One `latencyMs` and one token total per run.
   A three-round tool loop and a single slow call look identical in
   `turn_log`.
2. **No time to first token.** The visitor experiences TTFT, not total
   latency, and the stream already knows when the first content delta
   arrives.
3. **Tool and retrieval calls are not timed.** `execute_tools` and
   `SemanticRetriever.search` have no duration, so "slow tool call" cannot
   be distinguished from "slow model".
4. **Finish reason is discarded.** Truncated answers (`length`) are
   indistinguishable from complete ones after the fact.
5. **Retries are invisible.** The backoff loop in `OpenAICompatClient`
   logs a warning but records no count; a request that succeeded on its
   third attempt looks like a slow first attempt.
6. **No error type.** `outcome = "error"` with a stack trace in stdout. The
   conventions' `error.type` (a low-cardinality class) is what makes error
   rates sliceable.
7. **The `trace` event is not persisted.** It streams to the browser, where
   the client does not render it (`client/src/App.tsx` ignores `trace`), and
   then it is gone.
8. **No OTLP export and no dashboard.** Everything above would still need a
   place to be looked at.

---

## 5. Recommendations

Two paths, not mutually exclusive.

### 5.1 Minimal: enrich what exists (no new dependencies)

Keep the SSE and Postgres shape, add the missing fields. Each item is a
small change to code already read in this comparison.

- In `OpenAICompatProvider.stream`, record the time of the first
  `delta.content` and emit `ttftMs` on the `meta` event; record
  `finishReason` on `meta` too.
- In `OpenAICompatClient.stream`, count attempts and surface `retries` so
  the provider can put it on `meta`.
- In `execute_tools`, time each tool call and append
  `{"node": "tool:<name>", "ms": ..., "ok": bool}` to `trace`.
- In `verify`, pass `trace` through to `chat.py` and store it on
  `turn_log` as JSON (needs one alembic migration adding a `trace` column,
  plus `ttft_ms`, `finish_reason`, `retries`, `model_calls`).
- Add `error_type` to `turn_log` with a small fixed vocabulary:
  `timeout`, `rate_limited`, `provider_5xx`, `tool_failed`, `tool_timeout`,
  `other`.
- Pick the per-call aggregation deliberately: either keep summing in
  `_accumulate_meta` and store the per-call list alongside, or store one
  row per model call in a new `model_calls` table keyed by `turn_log.id`.

This closes items 1 through 7 in section 4 and keeps the project's
privacy posture unchanged.

### 5.2 Standard: emit OpenTelemetry alongside

Add `opentelemetry-sdk` and the OTLP exporter, and emit spans named per
the conventions from the places the code already has boundaries:

- `invoke_agent` (kind `INTERNAL`) around `agent.astream` in
  `stream_sse`, with `gen_ai.conversation.id = thread_id`.
- `chat {model}` (kind `CLIENT`) around `OpenAICompatProvider.stream`, with
  `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `gen_ai.usage.cache_read.input_tokens`, `gen_ai.response.finish_reasons`,
  `gen_ai.response.time_to_first_chunk`, `gen_ai.request.stream = true`.
- `execute_tool {name}` (kind `INTERNAL`) around `_call_tool`, with
  `gen_ai.tool.name`, `gen_ai.tool.call.id`, `error.type`.
- `retrieval` (kind `INTERNAL`) around `SemanticRetriever.search`, with
  `gen_ai.retrieval.top_k = 3`.
- `embeddings` around `OpenAICompatEmbeddings.embed`.
- Guard flags as `gen_ai.evaluation.result` events on the `invoke_agent`
  span (`gen_ai.evaluation.name = "uncited-answer"`, `score.label = "fail"`).
- Project-specific attributes as custom keys: `twin.outcome`,
  `twin.cost_usd`, `twin.guard_flags`, `twin.retries`.
- Leave `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` at its default
  `NO_CONTENT` in production; turn it on locally when debugging a wrong
  answer, which is the use case the project cannot serve today.

Point the exporter at Aspire Dashboard in `docker-compose.yml` for dev
(one extra service) and at whatever the host runs in production. The
Postgres tables stay as the product-level record (budget, approvals,
feedback, unanswered questions); OTel becomes the engineering-level record.

The second path is more work but is the one the post is arguing for: the
value is not the data, which the project largely has, but the standard shape
that lets any backend render it and lets this project's traces sit next to
Claude Code's or Copilot's in the same dashboard.
