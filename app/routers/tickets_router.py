import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import TicketRequest

router = APIRouter(prefix="/tickets", tags=["tickets"])


def _state_to_response(state: dict) -> dict:
    return {
        "category": state.get("category", ""),
        "draft_response": state.get("draft_response", ""),
        "confidence": state.get("confidence", 0.0),
        "awaiting_human": state.get("awaiting_human", False),
        "input_guardrail": {
            "in_scope": state.get("input_in_scope", True),
            "is_safe": state.get("input_is_safe", True),
            "reason": state.get("input_guardrail_reason", ""),
        },
        "output_guardrail": {
            "passed": state.get("output_guardrail_passed", True),
            "reason": state.get("output_guardrail_reason", ""),
        },
        "messages": [
            {"type": message.type, "content": message.content}
            for message in state.get("messages", [])
        ],
    }


@router.post("")
def create_ticket(payload: TicketRequest, request: Request) -> dict:
    graph = request.app.state.graph

    initial_state = {
        "ticket_id": payload.ticket_id,
        "ticket_text": payload.ticket_text,
        "category": "",
        "kb_results": "",
        "draft_response": "",
        "confidence": 0.0,
        "messages": [],
        "awaiting_human": False,
        "retry_count": 0,
        "input_in_scope": True,
        "input_is_safe": True,
        "input_guardrail_reason": "",
        "output_guardrail_passed": True,
        "output_guardrail_reason": "",
    }
    config = {"configurable": {"thread_id": payload.ticket_id}}

    result = graph.invoke(initial_state, config=config)
    return _state_to_response(result)


@router.get("/{ticket_id}/stream")
def stream_ticket(ticket_id: str, ticket_text: str, request: Request) -> EventSourceResponse:
    # Streaming variant of POST /tickets for a NEW submission only (not
    # resume) - GET + query param because browser EventSource only issues
    # GET requests and can't send a JSON body.
    graph = request.app.state.graph

    initial_state = {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "category": "",
        "kb_results": "",
        "draft_response": "",
        "confidence": 0.0,
        "messages": [],
        "awaiting_human": False,
        "retry_count": 0,
        "input_in_scope": True,
        "input_is_safe": True,
        "input_guardrail_reason": "",
        "output_guardrail_passed": True,
        "output_guardrail_reason": "",
    }
    config = {"configurable": {"thread_id": ticket_id}}

    def event_generator():
        for state in graph.stream(initial_state, config=config, stream_mode="values"):
            yield {"event": "state", "data": json.dumps(_state_to_response(state))}

    return EventSourceResponse(event_generator())


@router.get("/{ticket_id}")
def get_ticket(ticket_id: str, request: Request) -> dict:
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": ticket_id}}

    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return _state_to_response(snapshot.values)


@router.post("/{ticket_id}/resume")
def resume_ticket(ticket_id: str, request: Request) -> dict:
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": ticket_id}}

    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if not snapshot.next:
        raise HTTPException(status_code=400, detail="Ticket is not paused; nothing to resume")

    # Passing None as input resumes the graph from its last checkpoint
    # instead of starting a new run - it continues past the interrupted
    # node (flag_for_human) using the state already persisted for this
    # thread_id.
    result = graph.invoke(None, config=config)
    return _state_to_response(result)
