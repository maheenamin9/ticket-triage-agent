# Ticket Triage Agent

A LangGraph-based agent that triages support tickets using retrieval over a
knowledge base, exposed via a FastAPI service.

## Structure

```
ticket-triage-agent/
  app/
    main.py       # FastAPI app entrypoint
    config.py     # Settings (pydantic-settings + .env)
    models/       # Pydantic schemas
    graph/        # LangGraph state, nodes, edges, graph builder
    rag/          # Knowledge base ingestion + retriever
    routers/      # FastAPI routers
  data/
    kb_docs/      # Sample knowledge base documents
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your keys
```

## Run

```bash
uvicorn app.main:app --reload
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

Populate the knowledge base once before triaging tickets (see `data/kb_docs/`):

```bash
python -m app.rag.ingest
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

Notes:
- **Cycle:** `search_kb -> request_clarification -> search_kb` loops when
  retrieved KB content is too thin (< 30 chars). Capped by `retry_count`
  (max 2) so it can't loop forever - after that it proceeds to
  `draft_response` regardless.
- **Guardrails:** `guard_input` (before `classify_ticket`) judges the raw
  ticket text as in-scope/off-topic and safe/unsafe (prompt-injection or
  jailbreak attempts) via structured LLM output. Off-topic-but-safe tickets
  are politely rejected (`reject_ticket -> END`); unsafe ones are escalated
  straight to human review instead of processed. `guard_output` (after
  `draft_response`) judges the drafted reply as grounded in `kb_results` and
  free of PII/internal-detail leakage before it's allowed to auto-send -
  failing either check forces human review regardless of `confidence`.
- **Conditional edges:** `route_after_input_guard`, `route_after_search`
  (thin KB vs. enough KB), and `route_after_output_guard` (guardrail result
  + confidence-based auto-send vs. human escalation).
- **Human-in-the-loop gate:** the graph is compiled with
  `interrupt_before=["flag_for_human"]`, so execution pauses *before* the
  escalation node runs - a reviewer can inspect `draft_response`,
  `confidence`, and `kb_results` first. Resuming (`graph.invoke(None,
  config)`) continues straight to escalation, but a reviewer can also call
  `graph.update_state(config, {...})` before resuming - e.g. raising
  `confidence` above the 0.6 threshold (or setting `output_guardrail_passed`
  to `True`) causes `route_after_output_guard` to re-evaluate and route to
  `send_response` instead, effectively overriding the escalation rather than
  just annotating it.
- **Persistence:** the graph is compiled with a `SqliteSaver` checkpointer
  backed by `./data/checkpoints.db` (not in-memory), so paused/completed
  tickets survive server restarts. Each ticket's `ticket_id` is used as the
  LangGraph `thread_id`.

## API

All ticket endpoints are under `/tickets`.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check. |
| POST | `/tickets` | Submit a new ticket (`{"ticket_id": str, "ticket_text": str}`). Runs the graph to completion or until it pauses for human review; returns the resulting state. |
| GET | `/tickets/{ticket_id}/stream` | Streaming variant of `POST /tickets` for a **new** submission only (not resume). Server-Sent Events; `ticket_text` is a query param since browser `EventSource` only issues GET requests. Emits one `event: state` per graph step (`stream_mode="values"`), so a client can show "Classifying..." -> "Searching KB..." -> "Drafting..." live. |
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
