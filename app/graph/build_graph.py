from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    classify_ticket,
    draft_response,
    flag_for_human,
    guard_input,
    guard_output,
    reject_ticket,
    request_clarification,
    search_kb,
    send_response,
)
from app.graph.state import TicketState
from app.graph.timing import timed_node

MIN_KB_RESULTS_CHARS = 30
MAX_CLARIFICATION_RETRIES = 2
CONFIDENCE_THRESHOLD = 0.6


def route_after_input_guard(state: TicketState) -> str:
    # Unsafe (prompt-injection/jailbreak) attempts are escalated to a human
    # reviewer rather than silently rejected, since that's a signal worth a
    # person seeing. Off-topic-but-safe tickets are just politely rejected.
    if not state["input_is_safe"]:
        return "unsafe"
    if not state["input_in_scope"]:
        return "reject"
    return "ok"


def route_after_search(state: TicketState) -> str:
    kb_too_thin = len(state["kb_results"].strip()) < MIN_KB_RESULTS_CHARS
    if kb_too_thin and state["retry_count"] < MAX_CLARIFICATION_RETRIES:
        return "clarify"
    return "draft"


def route_after_output_guard(state: TicketState) -> str:
    if not state["output_guardrail_passed"]:
        return "needs_human"
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "needs_human"
    return "auto_send"


def build_graph(checkpointer):
    builder = StateGraph(TicketState)

    builder.add_node("guard_input", timed_node("guard_input")(guard_input))
    builder.add_node("reject_ticket", timed_node("reject_ticket")(reject_ticket))
    builder.add_node("classify_ticket", timed_node("classify_ticket")(classify_ticket))
    builder.add_node("search_kb", timed_node("search_kb")(search_kb))
    builder.add_node("request_clarification", timed_node("request_clarification")(request_clarification))
    builder.add_node("draft_response", timed_node("draft_response")(draft_response))
    builder.add_node("guard_output", timed_node("guard_output")(guard_output))
    builder.add_node("flag_for_human", timed_node("flag_for_human")(flag_for_human))
    builder.add_node("send_response", timed_node("send_response")(send_response))

    builder.set_entry_point("guard_input")

    builder.add_conditional_edges(
        "guard_input",
        route_after_input_guard,
        {"reject": "reject_ticket", "unsafe": "flag_for_human", "ok": "classify_ticket"},
    )
    builder.add_edge("reject_ticket", END)

    builder.add_edge("classify_ticket", "search_kb")

    builder.add_conditional_edges(
        "search_kb",
        route_after_search,
        {"clarify": "request_clarification", "draft": "draft_response"},
    )
    # Loops back to search_kb; route_after_search's retry_count check
    # (capped at MAX_CLARIFICATION_RETRIES) is what stops this from cycling
    # forever.
    builder.add_edge("request_clarification", "search_kb")

    builder.add_edge("draft_response", "guard_output")

    builder.add_conditional_edges(
        "guard_output",
        route_after_output_guard,
        {"needs_human": "flag_for_human", "auto_send": "send_response"},
    )

    builder.add_edge("flag_for_human", END)
    builder.add_edge("send_response", END)

    # interrupt_before=["flag_for_human"] pauses the graph right before the
    # escalation node runs, so a human reviewer sees the full state
    # (draft_response, confidence, kb_results, etc.) BEFORE the ticket is
    # formally handed off. From that paused state, the reviewer can either
    # resume the graph past flag_for_human to finalize the escalation, edit
    # the state and send the draft anyway, or take the ticket over
    # manually - instead of the escalation being finalized with no chance
    # to intervene.
    return builder.compile(checkpointer=checkpointer, interrupt_before=["flag_for_human"])
