# Ticket Triage Agent

A LangGraph-based agent that triages support tickets using retrieval over a
knowledge base, exposed via a FastAPI service.

Built for a fictional fitness app called **PulseTrack**. The knowledge base
covers account/login, billing, and shipping policies.

## Structure

```
ticket-triage-agent/
  app/
    main.py          # FastAPI app entrypoint
    config.py        # Settings (pydantic-settings + .env)
    llm.py           # LLM factories: cheap (Ollama) + generation (OpenAI)
    models/          # Pydantic schemas
    graph/
      state.py       # TicketState TypedDict
      nodes.py       # One function per graph node
      build_graph.py # Conditional edges + graph compiler
      timing.py      # Per-node latency instrumentation
    rag/
      ingest.py      # KB ingestion + retriever factory (similarity / MMR / hybrid / rerank)
      crag.py        # Corrective RAG: grade docs, web-search fallback (DuckDuckGo)
      crag_graph.py  # Same CRAG logic expressed as a LangGraph subgraph
      experiments.py # HyDE and RAG Fusion vs. direct retrieval demos
      eval_ragas.py  # RAGAS-based RAG evaluation harness
    routers/
      tickets_router.py  # FastAPI routes
  data/
    kb_docs/         # Knowledge base source documents (.txt / .pdf)
    chroma_kb/       # Persisted Chroma vectorstore (created on first ingest)
    checkpoints.db   # SQLite checkpoint store (created on first run)
```

## Prerequisites

- **Python 3.10+**
- **OpenAI API key** — used for embeddings and the customer-facing draft.
- **Ollama running locally** — used for all internal LLM calls (guardrails,
  classifier, document grading). Pull the model once:
  ```bash
  ollama pull llama3.2:3b
  ```
  Ollama defaults to `http://localhost:11434`. Override with `OLLAMA_BASE_URL`
  in `.env` if needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

### Environment variables (`.env`)

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | For embeddings (`text-embedding-ada-002`) and `gpt-4o-mini` generation. |
| `LANGCHAIN_TRACING_V2` | No | `true` to enable LangSmith tracing. |
| `LANGCHAIN_API_KEY` | No | LangSmith API key (needed only if tracing is on). |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (default: `ticket-triage-agent`). |
| `CHEAP_MODEL` | No | Ollama model for internal calls (default: `llama3.2:3b`). |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: `http://localhost:11434`). |
| `GENERATION_MODEL` | No | OpenAI model for draft generation (default: `gpt-4o-mini`). |

## Run

Populate the knowledge base once before triaging tickets:

```bash
python -m app.rag.ingest
```

Then start the server:

```bash
uvicorn app.main:app --reload
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

## The graph

`TicketState` (`app/graph/state.py`) is threaded through every node. Its
`messages` field uses `add_messages` as a reducer, so each node's audit note
appends to a running trail instead of overwriting it.

```
                     +-------------------+
                     |    guard_input    |
                     +---------+---------+
                               |
                 route_after_input_guard (conditional)
                               |
             unsafe        off-topic          in-scope & safe
                |               |                     |
                v               v                     v
      +-------------------+ +----------------+ +-------------------+
      |  flag_for_human   | |  reject_ticket | |  classify_ticket  |
      |  (interrupt_before)| +-------+--------+ +---------+---------+
      +---------+---------+         |                     |
                |                   v                     v
                |                  END          +-------------------+
                |             +----------------->|     search_kb     |
                |             |                  +---------+---------+
                |             |                            |
                |             |            route_after_search (conditional)
                |             |                            |
                |             |     kb_results too thin      kb_results OK
                |             |     & retry_count < 2        or retries used
                |             |            |                      |
                |             |            v                      v
                |             | +----------------------+  +-------------------+
                |             +-| request_clarification|  |   draft_response  |
                |               +----------------------+  +---------+---------+
                |                                                    |
                |                                                    v
                |                                          +-------------------+
                |                                          |    guard_output   |
                |                                          +---------+---------+
                |                                                    |
                |                                route_after_output_guard (conditional)
                |                                                    |
                |                     failed guardrail or confidence < 0.6     confidence >= 0.6 & passed
                |                                    |                                  |
                +------------------------------------+                                  v
                                                                             +-------------------+
                                                                             |   send_response   |
                                                                             +---------+---------+
                                                                                       |
                                                                                       v
                                                                                      END
```

### Key design decisions

- **Dual-model setup:** Internal calls (guardrails, classifier, document
  grader) use a free local Ollama model (`llama3.2:3b`) — no per-call cost.
  Customer-facing draft generation uses `gpt-4o-mini`. Configured in
  `app/llm.py` and `app/config.py`.

- **Cycle:** `search_kb → request_clarification → search_kb` loops when
  retrieved KB content is too thin (< 30 chars). Capped by `retry_count`
  (max 2) so it can't loop forever — after that it proceeds to
  `draft_response` regardless.

- **Guardrails:** `guard_input` (before `classify_ticket`) judges the raw
  ticket text as in-scope/off-topic and safe/unsafe (prompt-injection or
  jailbreak attempts) via structured LLM output. Off-topic-but-safe tickets
  are politely rejected (`reject_ticket → END`); unsafe ones are escalated
  straight to human review instead of processed. `guard_output` (after
  `draft_response`) judges the drafted reply as grounded in `kb_results` and
  free of PII/internal-detail leakage before it's allowed to auto-send —
  failing either check forces human review regardless of `confidence`.

- **Conditional edges:** `route_after_input_guard`, `route_after_search`
  (thin KB vs. enough KB), and `route_after_output_guard` (guardrail result
  + confidence-based auto-send vs. human escalation).

- **Human-in-the-loop gate:** the graph is compiled with
  `interrupt_before=["flag_for_human"]`, so execution pauses *before* the
  escalation node runs — a reviewer can inspect `draft_response`,
  `confidence`, and `kb_results` first. Resuming (`graph.invoke(None,
  config)`) continues straight to escalation, but a reviewer can also call
  `graph.update_state(config, {...})` before resuming — e.g. raising
  `confidence` above the 0.6 threshold (or setting `output_guardrail_passed`
  to `True`) causes `route_after_output_guard` to re-evaluate and route to
  `send_response` instead, effectively overriding the escalation.

- **Persistence:** the graph is compiled with a `SqliteSaver` checkpointer
  backed by `./data/checkpoints.db` (not in-memory), so paused/completed
  tickets survive server restarts. Each ticket's `ticket_id` is used as the
  LangGraph `thread_id`.

- **Per-node timing:** every node is wrapped with `timed_node` from
  `app/graph/timing.py`, which prints latency per step to stdout so slow
  nodes are easy to spot.

- **`send_response` is a stub:** the node logs a message but does not send
  an actual email or call any external API. Wire it up to your delivery
  mechanism of choice.

## API

All ticket endpoints are under `/tickets`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check. |
| POST | `/tickets` | Submit a new ticket (`{"ticket_id": str, "ticket_text": str}`). Runs the graph to completion or until it pauses for human review; returns the resulting state. |
| GET | `/tickets/{ticket_id}/stream` | Streaming variant of `POST /tickets` for a **new** submission only (not resume). Server-Sent Events; `ticket_text` is a query param since browser `EventSource` only issues GET requests. Emits one `event: state` per graph step (`stream_mode="values"`), so a client can show "Classifying..." → "Searching KB..." → "Drafting..." live. |
| GET | `/tickets/{ticket_id}` | Fetch the current persisted state for a ticket (e.g. to check `awaiting_human`). 404 if the ticket doesn't exist. |
| POST | `/tickets/{ticket_id}/resume` | Resume a ticket paused at the human-review gate. 400 if the ticket isn't paused, 404 if it doesn't exist. |

Every response has the same shape:

```json
{
  "category": "billing | technical | general | \"\"",
  "draft_response": "string",
  "confidence": 0.0,
  "awaiting_human": false,
  "input_guardrail": {"in_scope": true, "is_safe": true, "reason": "string"},
  "output_guardrail": {"passed": true, "reason": "string"},
  "messages": [{"type": "ai", "content": "..."}]
}
```

### Example

```bash
# Submit a ticket
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "t-001", "ticket_text": "I was charged twice for my subscription this month."}'

# Check state of a paused ticket
curl http://localhost:8000/tickets/t-001

# Resume a ticket that is awaiting human review
curl -X POST http://localhost:8000/tickets/t-001/resume

# Stream a new ticket's progress (Server-Sent Events)
curl "http://localhost:8000/tickets/t-002/stream?ticket_text=How+do+I+reset+my+password"
```

## RAG extras

Beyond the main retriever used by the graph, the `app/rag/` folder contains
standalone scripts that explore retrieval techniques:

### Retriever options (`app/rag/ingest.py`)

`get_kb_retriever()` supports several `search_type` values and optional
post-retrieval improvements:

| Option | Description |
|---|---|
| `similarity` (default) | Dense vector search via Chroma. |
| `mmr` | Maximal Marginal Relevance — balances relevance with diversity. |
| `similarity_score_threshold` | Only returns chunks above a similarity cutoff. |
| `hybrid` | Ensemble of dense (Chroma) + sparse (BM25) retrieval, configurable via `dense_weight`. |
| `rerank=True` | Adds a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) on top of any base retriever. |
| `multi_query=True` | Rephrases the query into several variants and unions the results before reranking. |

### Corrective RAG — `app/rag/crag.py`

Plain-Python CRAG loop: retrieve → grade each doc relevant/irrelevant → if
zero relevant docs, fall back to a DuckDuckGo web search instead of
generating from empty context. No API key needed for the web fallback.

```bash
python -m app.rag.crag
```

A LangGraph version of the same logic is in `app/rag/crag_graph.py`.

### HyDE & RAG Fusion — `app/rag/experiments.py`

Compares three retrieval strategies side-by-side:

- **Direct:** embed the raw question and search.
- **HyDE (Hypothetical Document Embeddings):** generate a hypothetical answer
  first, embed that, then search — the embedding lands closer to real policy
  excerpts.
- **RAG Fusion:** generate multiple query variants, retrieve for each,
  re-rank with Reciprocal Rank Fusion.

```bash
python -m app.rag.experiments
```

### RAGAS evaluation — `app/rag/eval_ragas.py`

Runs the golden dataset through the retriever and scores it with RAGAS
metrics (faithfulness, answer relevancy, context precision/recall). Requires
`OPENAI_API_KEY`.

```bash
python -m app.rag.eval_ragas
```
