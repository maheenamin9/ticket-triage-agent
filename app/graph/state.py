from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class TicketState(TypedDict):
    """Shared state threaded through the ticket-triage LangGraph.

    ticket_id: Unique identifier of the support ticket being processed.
    ticket_text: The raw text of the customer's ticket/message.
    category: Ticket classification - "billing" | "technical" | "general" | ""
        (empty string before the classifier node has run).
    kb_results: Text of the knowledge-base chunks retrieved for this ticket.
    draft_response: The agent-generated reply drafted for the ticket.
    confidence: Model's confidence (0.0-1.0) in the draft response, used to
        decide whether the ticket needs human review.
    messages: Running audit trail of the graph's steps (LLM calls, tool
        results, etc.). Uses add_messages as a reducer so each node's update
        appends to the list instead of replacing it.
    awaiting_human: Whether the ticket is currently parked for human
        review/approval instead of being auto-resolved.
    retry_count: Number of times the graph has looped back through
        search_kb after asking the customer for clarification, used to
        cap the clarification loop and prevent it from running forever.
    input_in_scope: Whether guard_input judged the ticket an on-topic
        PulseTrack support question (not necessarily unsafe if False - just
        unrelated).
    input_is_safe: Whether guard_input judged the ticket free of
        prompt-injection/jailbreak attempts. Kept separate from
        input_in_scope so an off-topic-but-safe ticket can be politely
        rejected while an unsafe one is escalated to a human instead.
    input_guardrail_reason: One-sentence explanation for the input verdict.
    output_guardrail_passed: Whether guard_output judged the draft grounded
        in kb_results and free of PII/internal-detail leakage.
    output_guardrail_reason: One-sentence explanation for the output verdict.
    """

    ticket_id: str
    ticket_text: str
    category: str
    kb_results: str
    draft_response: str
    confidence: float
    messages: Annotated[list, add_messages]
    awaiting_human: bool
    retry_count: int
    input_in_scope: bool
    input_is_safe: bool
    input_guardrail_reason: str
    output_guardrail_passed: bool
    output_guardrail_reason: str
