# PRD: Restructure `server/app/` along the GenAI platform components

| | |
|---|---|
| Status | Implemented on branch `refactor/genai-layout` (2026-11-07) |
| Author | Ruud Juffermans |
| Date | 2026-11-07 |
| Inputs | `docs/FOLDERSTRUCTURE_GENAI.md` (target shape), `docs/GENAI_PLATFORM_MAPPING.md` (what exists, what is missing) |
| Outcome | See §12 |
| Scope | `server/app/`, `server/evals/run.py`, `server/tests/`, `server/migrations/env.py`, the two guides under `.claude/docs/` |

---

## 1. Problem

`server/app/` has grown to 23 modules in two directories. The separation Chip
Huyen's post describes (context, guardrails, gateway, cache, actions,
orchestration, observability, storage) is all present, but it is encoded in
file names rather than folders, and several files hold more than one
component:

| File | Lines | Components mixed |
|---|---|---|
| `app/admin_read.py` | 762 | admin API routes + JSON shaping + stats queries |
| `app/db.py` | 560 | storage models, engine/migrations, approval store, telemetry (`TurnRecorder`), retention, feedback, eval history |
| `app/agent/graph.py` | 328 | orchestration, all six node bodies, tracing, output guard heuristic |
| `app/providers.py` | 247 | provider balance clients + admin routes |
| `app/llm.py` | 191 | provider table, system prompt, prompt versioning, prompt-cache accounting, streaming model call |
| `app/agent/tools.py` | 163 | tool base type, three tools, term-based search, semantic search adapter |
| `app/agent/guards.py` | 110 | input detectors, output detectors, refusal copy |

Two concrete costs today:

- **A dependency points the wrong way.** `app/db.py` imports
  `app.agent.guards.looks_like_refusal` so the telemetry recorder can classify
  unanswered questions. Storage depends on a guardrail detector.
- **The gaps the mapping document ranks highest have no home.** Provider
  fallback, persisted traces, time-to-first-token, length metrics and hybrid
  retrieval (mapping doc §5, items 1 to 4) would each land in an
  already-crowded file. The restructure is what makes those follow-on changes
  small.

## 2. Goals

1. One folder per platform component, named as in the post, so a reader who
   knows the post can navigate the code without a map.
2. The dependency rule from `FOLDERSTRUCTURE_GENAI.md` holds and is enforced
   by a test: component folders never import each other; only `pipeline/`
   composes them; `observability/` and `storage/` are leaves.
3. Zero behaviour change. Same endpoints, same events, same database schema,
   same eval scores.
4. Every module under `app/` has one responsibility and stays under roughly
   300 lines.
5. Each of the mapping document's top gaps has a named file where it will go.

## 3. Non-goals

- No new features. Fallback, trace persistence, TTFT, length metrics and
  hybrid search are listed in §8 as follow-ons, not delivered here.
- No change to the public HTTP API, the SSE event shapes, the admin API, or
  the LangGraph node names (the `trace` event exposes them).
- No change to `server/migrations/`, `server/knowledge/`, `server/evals/datasets/`,
  or `client/`.
- No change to the `app` package name: the Dockerfile runs `app.main:app` and
  every test imports `app.*`.
- No framework changes. LangGraph, FastAPI, SQLAlchemy and httpx stay.

## 4. Target structure

Adapted from `FOLDERSTRUCTURE_GENAI.md`. Deviations from that document are
marked **(dev)** and explained in §5.

```
server/
├── app/
│   ├── main.py                     # wiring only: build components, compile graph, mount routers
│   ├── config.py                   # Settings (unchanged)
│   │
│   ├── api/                        # transport
│   │   ├── chat.py                 # POST /api/chat (SSE), POST /api/feedback
│   │   ├── admin.py                # approvals write endpoint, require_admin
│   │   ├── admin_read.py           # read API: stats, turns, threads, unanswered, incidents, feedback, contact, evals, knowledge, prompt
│   │   ├── providers.py            # GET /api/admin/providers, /spend
│   │   ├── schemas.py              # ChatRequest, FeedbackRequest
│   │   ├── rate_limit.py           # slowapi limiter
│   │   └── security.py             # headers, CSP, body cap, error-leak handling
│   │
│   ├── context/                    # Step 1: enhance context
│   │   ├── knowledge.py            # KnowledgeDoc, load_knowledge
│   │   ├── chunking.py             # split_into_chunks
│   │   ├── embeddings.py           # OpenAICompatEmbeddings, create_embedder
│   │   ├── retrieval.py            # term_search (from tools._search), SemanticRetriever, sync_knowledge_chunks
│   │   └── prompt.py      (dev)    # SYSTEM_PROMPT, knowledge_system_prompt, prompt_version, knowledge_version
│   │
│   ├── guardrails/                 # Step 2: guardrails
│   │   ├── input.py                # INJECTION_MARKERS, SENSITIVE_NUMBER, OFFTOPIC_PATTERNS, screen_input
│   │   ├── output.py               # valid_citation, looks_like_refusal, uncited_answer (from graph.verify)
│   │   └── policies.py             # refusal copy: INPUT_REFUSAL, PII_REFUSAL, OFFTOPIC_REDIRECT, BUDGET_REFUSAL
│   │
│   ├── gateway/                    # Step 3: model gateway
│   │   ├── client.py               # OpenAICompatClient, OpenAICompatError (from openai_client.py)
│   │   ├── providers.py            # PROVIDERS table (from llm.PROVIDERS)
│   │   ├── chat_model.py           # OpenAICompatProvider.stream, _parse_arguments, _cached_tokens
│   │   ├── pricing.py              # cost_usd
│   │   ├── balances.py             # ProviderBalances, per-provider balance clients (from providers.py)
│   │   └── fallback.py             # (follow-on, §8) alternate provider on exhausted retries
│   │
│   ├── actions/                    # Step 5: tools and write actions
│   │   ├── tools/
│   │   │   ├── base.py             # Tool dataclass, tool_definitions, MAX_* constants
│   │   │   ├── search_knowledge.py
│   │   │   ├── get_availability.py # AVAILABILITY
│   │   │   ├── draft_contact_message.py   # high_risk=True
│   │   │   └── registry.py         # build_tools
│   │   ├── approval.py             # ApprovalStore protocol, InMemory/Postgres stores (from db.py)
│   │   └── budget.py               # BudgetTracker
│   │
│   ├── pipeline/                   # orchestration: the only place components meet
│   │   ├── state.py                # AgentState
│   │   ├── graph.py                # build_agent: nodes, edges, routing functions, checkpointer
│   │   └── nodes/
│   │       ├── guard_input.py
│   │       ├── refuse.py
│   │       ├── generate.py         # model round, citation filtering, _accumulate_meta
│   │       ├── request_approval.py # interrupt()
│   │       ├── execute_tools.py    # _call_tool with timeout
│   │       └── verify.py           # output flags, trace event
│   │
│   ├── observability/              # leaf
│   │   ├── tracing.py              # _traced node wrapper
│   │   ├── telemetry.py            # TurnRecorder (from db.py)
│   │   ├── feedback.py             # record_feedback
│   │   ├── evals.py                # record_eval_run
│   │   └── retention.py            # purge_expired_text, retention loop (from main.py)
│   │
│   └── storage/                    # leaf
│       ├── engine.py               # sqlalchemy_url, create_engine_and_sessions, run_migrations
│       └── models.py               # Base and every *Row class
│
├── migrations/                     # unchanged; env.py imports app.storage.models.Base
├── knowledge/                      # unchanged
├── evals/                          # unchanged layout; run.py imports updated
└── tests/                          # mirror app/ (see §7, phase 8)
```

## 5. Deviations from `FOLDERSTRUCTURE_GENAI.md`

- **No `cache/` folder.** The project's cache behaviour is provider-side
  prompt caching (observed through `_cached_tokens`, 6 lines) and the
  content-hash check inside `sync_knowledge_chunks`. Neither is a component;
  both stay where they are used. The folder document's own rule applies: an
  unused component is absent. Create `cache/` the day an exact or semantic
  cache is actually built.
- **`context/prompt.py` instead of a top-level prompts folder.** The system
  prompt is context construction in the post's sense, and `prompt_version` /
  `knowledge_version` belong next to the documents they hash.
- **`gateway/balances.py` has no analogue in the post.** Provider fund checks
  are a gateway concern (cost management), so they live there; the routes
  that expose them live in `api/providers.py`.
- **`observability/evals.py`** holds `record_eval_run`, because eval history
  is telemetry written by `server/evals/run.py`; the datasets and scorers stay
  in `server/evals/` as the folder document prescribes.
- **No `gateway/router.py`.** The mapping document records that routing is
  deterministic Python inside the graph, not a model. The routing functions
  stay in `pipeline/graph.py`.

## 6. Dependency rules

Enforced by a new test `tests/test_architecture.py` that walks the AST of
every module under `app/` and asserts the import graph below. The test is
written in phase 0 so it fails until the move is complete, then stays green.

| From | May import |
|---|---|
| `api/` | everything except `main` (transport is the outermost ring) |
| `pipeline/` | everything except `api/` and `main` |
| `context/`, `guardrails/`, `gateway/`, `actions/` | `config`, `storage/`, `observability/`, and their own folder only |
| `observability/` | `storage/`, `config` |
| `storage/` | `config` |
| `main.py` | everything |

The rule as enforced is simpler than the first draft of this table: the
four component folders never import each other, the two leaves import only
downward, and `pipeline/` and `api/` are the composition layers.

Two existing imports violate this and are fixed as part of the move:

1. `db.py → agent.guards.looks_like_refusal`. Fix: the `verify` node already
   knows whether the answer is a refusal. It writes `refusal: bool` into the
   state; `TurnRecorder` reads the flag and stops importing a detector.
2. `retrieval.py → db.KnowledgeChunkRow` is allowed under the table above
   (`context/` may import `storage/`), so it stays.

`actions/tools/search_knowledge.py` importing `context/retrieval.py` would
break the rule. Instead the tool receives a search callable from `main.py`,
exactly as it receives `retriever` today. The tool folder knows the shape of
the function, not where it comes from.

## 7. Migration plan

Eight commits. Each one moves a single component, updates every importer
(tests, `evals/run.py`, `migrations/env.py`), and leaves `uv run pytest` and
`uv run ruff check` green. No compatibility shims: with 21 test files and one
eval runner, updating imports is cheaper than maintaining re-exports.

| Phase | Commit | Moves | Importers to update |
|---|---|---|---|
| 0 | `test: assert component import rules` | adds `tests/test_architecture.py`, marked `xfail(strict=True)` until phase 8 | none |
| 1 | `refactor: split db.py into storage and observability` | `storage/engine.py`, `storage/models.py`, `observability/telemetry.py`, `observability/feedback.py`, `observability/evals.py`, `observability/retention.py`, `actions/approval.py`; `refusal` flag replaces the guards import | `main`, `chat`, `admin`, `admin_read`, `providers`, `retrieval`, `evals/run`, `migrations/env`, `test_db`, `test_retrieval`, `test_evals`, `test_admin_read` |
| 2 | `refactor: gateway folder` | `gateway/client.py`, `gateway/providers.py`, `gateway/chat_model.py`, `gateway/pricing.py`, `gateway/balances.py` (balance clients out of `providers.py`) | `main`, `embeddings`, `admin_read`, `evals/run`, `test_openai_compat`, `test_chat`, `test_providers`, `test_agent` |
| 3 | `refactor: context folder` | `context/knowledge.py`, `context/chunking.py`, `context/embeddings.py`, `context/retrieval.py` (gains `term_search` from `tools._search`), `context/prompt.py` | `main`, `gateway/chat_model`, `admin_read`, `agent/tools`, `agent/graph`, `evals/run`, `test_knowledge`, `test_retrieval`, `test_citations` |
| 4 | `refactor: guardrails folder` | `guardrails/input.py`, `guardrails/output.py` (incl. the uncited-answer heuristic lifted out of `graph.verify`), `guardrails/policies.py` (incl. `BUDGET_REFUSAL` from `graph.py`) | `agent/graph`, `observability/telemetry` (removed), `test_guards`, `test_agent` |
| 5 | `refactor: actions folder, one module per tool` | `actions/tools/{base,search_knowledge,get_availability,draft_contact_message,registry}.py`, `actions/budget.py`; `search_knowledge` takes a search callable | `agent/graph`, `main`, `test_tools`, `test_budget`, `test_approvals` |
| 6 | `refactor: pipeline folder with one module per node` | `pipeline/state.py`, `pipeline/graph.py`, `pipeline/nodes/*.py`, `observability/tracing.py` | `main`, `evals/run`, `test_agent`, `test_approvals`, `test_chat` |
| 7 | `refactor: api folder` | `api/{chat,admin,admin_read,providers,schemas,rate_limit,security}.py`; `main.py` reduced to lifespan + `create_app` | `main`, `test_chat`, `test_security`, `test_hardening`, `test_admin_read`, `test_providers` |
| 8 | `refactor: mirror app/ in tests/, enable architecture test` | `tests/` reorganised into `tests/{api,context,guardrails,gateway,actions,pipeline,observability,storage}/`; `xfail` removed from phase 0 test; `.claude/docs/BUILD_GUIDE.md`, `PORTFOLIO_GUIDE.md` and `docs/GENAI_PLATFORM_MAPPING.md` paths updated | docs only |

Ordering rationale: phase 1 first because `db.py` is the hub everything else
imports and because the mapping document names it as the highest-value
split. Phases 2 to 5 are leaves and can be done in any order. Phase 6 depends
on 3, 4 and 5 because the nodes import from all three. Phase 7 is last among
code moves because the routers import everything else.

Per-phase checklist:

- `uv run pytest` green, `uv run ruff check .` clean.
- `uv run alembic check` (or `alembic revision --autogenerate` producing an
  empty migration) to prove `storage/models.py` is byte-for-byte the same
  schema.
- `docker compose up` serves `/health` and a chat turn with
  `LLM_PROVIDER=deepseek` (cheapest configured provider).
- Commit message in the imperative form above, ticked in `BUILD_GUIDE.md`
  if that guide gains a "Phase 12: restructure" section (decision in §10).

## 8. Follow-ons the structure makes room for

Not part of this PRD. Listed so the restructure can be judged by whether
these become small changes. Order follows `GENAI_PLATFORM_MAPPING.md` §5.

| # | Gap | Lands in | Size after restructure |
|---|---|---|---|
| 1 | Provider fallback on exhausted retries | `gateway/fallback.py`, composed in `pipeline/provider.py`; `FALLBACK_PROVIDER` in `config.py` | done: one module, one setting (branch `feat/provider-fallback`) |
| 2 | Persist per-node trace, record time-to-first-token | `observability/tracing.py` collects, `observability/telemetry.py` writes, `storage/models.py` + migration `0009` adds columns | two modules, one migration |
| 3 | Query, prompt and answer length metrics | `pipeline/nodes/verify.py` computes, `observability/telemetry.py` records | two edits |
| 4 | Hybrid retrieval (term + vector) before the posts corpus arrives | `context/retrieval.py` only; the tool already receives a callable | one module |
| 5 | Sampled offline AI judge | new `evals/judge.py` reading `turn_log`, writing to `observability/evals.py` | outside `app/` |
| 6 | Reversible PII masking | `guardrails/input.py` masks, `pipeline/nodes/verify.py` unmasks | two modules |
| 7 | Semantic cache | create `cache/` then | skip, per the post |

## 9. Acceptance criteria

- [ ] `tests/test_architecture.py` passes without `xfail`.
- [ ] No module under `app/` exceeds 300 lines, except `api/admin_read.py`,
      which is allowed up to 400 after its JSON shaping helpers move to a
      sibling `api/admin_json.py` (optional; decide in phase 7).
- [ ] `grep -rn "from app.db\|from .db\|app.agent\|app.llm\|app.openai_client" server/` returns nothing.
- [ ] `alembic check` reports no schema drift.
- [ ] Full test suite green; `ruff` clean.
- [ ] `uv run python -m evals.run --dataset tool_selection` scores identical
      to the last run recorded in `eval_runs` (same provider and model).
- [ ] Both guides under `.claude/docs/` and `docs/GENAI_PLATFORM_MAPPING.md`
      reference the new paths. `docs/FOLDERSTRUCTURE_GENAI.md` is left as the
      generic reference it is.
- [ ] `docker compose up` on `main` serves a chat turn end to end.

## 10. Open decisions

1. **Does the restructure become a `BUILD_GUIDE.md` phase?** The guide is a
   commit-by-commit tutorial and `/validate-commit` reads it. Recommendation:
   yes, as Phase 12 with the eight commits from §7, because the guide's
   readers otherwise meet a layout the guide never explains.
2. **Branch strategy.** One PR with eight commits (reviewable as a sequence,
   merges once) or eight PRs (each independently revertable). Recommendation:
   one branch `refactor/genai-layout`, one PR, squash disabled so the
   commits survive.
3. **`api/admin_read.py` split.** Leave as one 762-line router, or split per
   resource (`admin_stats.py`, `admin_turns.py`, `admin_content.py`).
   Recommendation: defer; it is already isolated once it sits under `api/`,
   and the split carries no architectural value.
4. **Shim for `app.db`?** `server/migrations/env.py` is the only importer
   outside `app/` and tests that touches `Base`. Recommendation: no shim,
   update `env.py` in phase 1.

## 11. Risks

| Risk | Mitigation |
|---|---|
| A node move changes LangGraph node names and breaks the `trace` event the client renders | Node names are strings in `graph.add_node`; phase 6 keeps them and `test_agent` asserts the trace list |
| Alembic autogenerate sees a renamed model module as a new table | Table names are explicit in each `Row` class; `alembic check` in every phase |
| Checkpointed threads in Postgres reference the old `AgentState` import path | `AgentState` is a `TypedDict`; checkpoints store plain dicts, not class paths. Verify by resuming a pending approval across the phase 6 deploy |
| The `refusal` flag replacing the `looks_like_refusal` import changes which questions land in `unanswered_questions` | `test_db` gains a case that feeds a refusal-shaped answer through `verify` and asserts the row is tagged `model` |
| `uv.lock` / Dockerfile paths | Neither references module paths; only `app.main:app`, which does not move |

## 12. Outcome

Implemented in eight commits on `refactor/genai-layout`, each leaving the
suite green (106 tests, `ruff` clean). Decisions taken while implementing
that differ from §4 and §5:

- **`gateway/embeddings.py`, not `context/embeddings.py`.** The embeddings
  client needs the provider table (base URL, key field), so under the import
  rule it belongs to the gateway, which the folder-structure document also
  says should own every outbound model call. `context/retrieval.py` takes
  the embedder as a duck-typed argument.
- **`pipeline/provider.py` (`grounded_provider`).** The chat model no longer
  builds the system prompt itself (that would make gateway import context).
  It takes `system_prompt` and `titles`; the one place that composes prompt
  and model is this small pipeline module, used by `main.py`, the eval
  runner and the tests.
- **`build_tools(search)`.** The tool registry takes the search callable
  instead of `(docs, retriever)`; `context/retrieval.py: make_search` builds
  the semantic-with-fallback callable and `pipeline/graph.py` passes it in.
- **`model_refusal` state flag.** The telemetry recorder no longer imports
  the refusal detector. The `verify` node writes `model_refusal` into the
  run state and `api/chat.py` hands it to `TurnRecorder.record`. Same rows
  land in `unanswered_questions`.
- **One provider table.** `gateway/providers.py` merges the chat table
  (base URLs) and the admin-panel table (labels) in the admin panel's order,
  so the `/api/admin/providers` listing is unchanged.
- **`spend_by_provider` stayed with its route** in `api/providers.py`, like
  the other admin read queries in `api/admin_read.py`, which was moved
  unsplit (762 lines; §10 item 3 deferred as recommended).
- **Node modules are factories.** Each `pipeline/nodes/*.py` exports a
  `make_<node>(deps)` returning the async node, so a node declares exactly
  what it needs (`budget`, `tools_by_name`, `get_provider`) and the graph
  module only wires and routes.
- **Tests mirror `app/`** under `tests/<folder>/`, with `conftest.py` and
  `test_architecture.py` at the root. The architecture test is live (no
  `xfail`).
- **§8 items 2 and 3 landed independently** (performance telemetry and
  trace persistence, migration `0009`) between the PRD and the move, and are
  recorded as done in the mapping document.
- **Not done:** `BUILD_GUIDE.md` is untouched by decision (no Phase 12). Its
  commit descriptions reference the historical paths. `PORTFOLIO_GUIDE.md`
  has updated paths and a note that its snippets predate the layout.
  `alembic check` against Postgres was not run here (no database in the
  session); `storage/models.py` was moved verbatim and the SQLite-backed
  tests that run the migrations and inspect the schema pass.
