looking at this blogpost https://huyenchip.com/2024/07/25/genai-platform.html , what would be a good folder structure for a generelised genai app           

⏺ A good structure is one folder per component the post names, with the orchestrator as the only thing that knows how they connect. That keeps the post's
  point that "a component can be skipped if the system works without it": an unused folder is empty or absent, and nothing else changes.

  genai-app/
  ├── app/
  │   ├── main.py                  # wiring only: build components, compile pipeline, serve
  │   ├── config.py                # settings, feature switches (which components are on)
  │   ├── api/                     # transport: HTTP/SSE routes, request schemas, rate limits
  │   │
  │   ├── context/                 # Step 1: Enhance context
  │   │   ├── knowledge/           # source documents, loaders, chunking
  │   │   ├── retrieval/           # term-based, embedding-based, hybrid, reranking
  │   │   ├── embeddings.py
  │   │   └── query_rewrite.py     # standalone-query rewriting for follow-ups
  │   │
  │   ├── guardrails/              # Step 2: Put in guardrails
  │   │   ├── input/               # PII masking, injection, out-of-scope
  │   │   ├── output/              # format, hallucination, toxicity, brand, citations
  │   │   └── policies.py          # what to do on each failure mode: retry, block, human
  │   │
  │   ├── gateway/                 # Step 3: Model router and gateway
  │   │   ├── client.py            # one wire-protocol client: auth, streaming, retry/backoff
  │   │   ├── providers.py         # provider table: base URLs, key names, quirks
  │   │   ├── router.py            # intent / out-of-scope / next-action routing
  │   │   ├── fallback.py          # alternate model on 429/5xx
  │   │   └── pricing.py           # cost per token per model
  │   │
  │   ├── cache/                   # Step 4: Cache
  │   │   ├── prompt_cache.py      # stable-prefix layout, cached-token accounting
  │   │   ├── exact.py             # keyed cache for expensive deterministic steps
  │   │   └── semantic.py          # only if hit rate justifies it
  │   │
  │   ├── actions/                 # Step 5: Complex logic and write actions
  │   │   ├── tools/               # one module per tool, strict schema, read vs write flag
  │   │   ├── approval.py          # human-in-the-loop gate for write actions
  │   │   └── budget.py            # spend limits, fail closed
  │   │
  │   ├── pipeline/                # Orchestration
  │   │   ├── state.py             # typed state flowing through the run
  │   │   ├── graph.py             # nodes and edges; the only place components meet
  │   │   └── nodes/               # guard_input, generate, execute_tools, verify, ...
  │   │
  │   ├── observability/           # cross-cutting, built in from the start
  │   │   ├── metrics.py           # tokens, cost, latency, TTFT, lengths
  │   │   ├── tracing.py           # per-step spans (OTel GenAI conventions)
  │   │   ├── logging.py           # structured logs with request/thread ids
  │   │   └── feedback.py          # user thumbs, unanswered questions
  │   │
  │   └── storage/                 # durable state behind the above
  │       ├── models.py            # tables: turn log, ledger, approvals, chunks, evals
  │       └── migrations/
  │
  ├── evals/                       # "evaluation is necessary at every step"
  │   ├── datasets/                # one file per capability: retrieval, tools, completion
  │   ├── scorers/                 # deterministic scorers and LLM judges
  │   └── run.py                   # thresholds, failure injection, history
  │
  ├── knowledge/                   # content, separate from code
  ├── client/                      # UI, if any
  ├── tests/                       # mirror app/ one-to-one
  └── docker-compose.yml

  A few rules make the structure hold up as the app grows:

  - Dependencies point inward to the pipeline. context, guardrails, gateway, cache and actions never import each other. Only pipeline/graph.py composes them.
    That is what lets you add or remove a step without touching the others, which is the post's whole progression argument.
  - The gateway owns every outbound model call. Retrieval embeddings, routing classifiers and scorers all go through gateway/client.py. Access control, cost
    and fallback then live in one place, as the post describes.
  - Guardrails are split by where they run, not by what they detect. Input guards run before any tokens are spent, output guards after. Policies for what to
    do on a failure sit apart from detection, because the same detector can block in one deployment and only flag in a streaming one.
  - Tools declare read versus write. The approval gate and the budget guard key off that flag, so adding a tool never means touching the orchestrator.
  - Observability and storage are leaves, not hubs. Every component emits to them; nothing reads from them at request time except the budget guard and the
    cache.
  - Evals live outside app/ and drive the real pipeline, with one dataset per component so a regression names the folder that caused it.

  How this project compares. Your server/app/ already has the right separation, just flatter: agent/ holds pipeline, guards, tools and budget together, llm.py
  plus openai_client.py plus pricing.py are the gateway, retrieval.py plus embeddings.py plus knowledge.py are context, and db.py is storage and
  observability in one file. At the current size that flatness is fine. The moves that would pay off first if the app keeps growing are splitting db.py into
  storage models versus a telemetry module, and giving tools one module each under a folder so the read/write flag and approval logic are visible per tool.