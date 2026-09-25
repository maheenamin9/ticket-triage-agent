from langchain_core.messages import AIMessage

from app.graph.state import TicketState
from app.llm import get_cheap_llm, get_generation_llm
from app.models.schemas import DraftedResponse, InputGuardrailResult, OutputGuardrailResult, TicketClassification
from app.rag.ingest import get_kb_retriever


def guard_input(state: TicketState) -> dict:
    llm = get_cheap_llm()
    guard = llm.with_structured_output(InputGuardrailResult)

    result: InputGuardrailResult = guard.invoke(
        "You are a security/scope filter in front of a customer support "
        "agent for PulseTrack (a fitness tracker app/device). Judge the "
        "ticket text below.\n\n"
        f"Ticket:\n{state['ticket_text']}"
    )

    return {
        "input_in_scope": result.in_scope,
        "input_is_safe": result.is_safe,
        "input_guardrail_reason": result.reason,
        "messages": [
            AIMessage(
                content=f"Input guardrail: in_scope={result.in_scope}, "
                f"is_safe={result.is_safe} ({result.reason})"
            )
        ],
    }


def reject_ticket(state: TicketState) -> dict:
    return {
        "draft_response": (
            "This doesn't look like a PulseTrack account, billing, or "
            "shipping question, so we're not able to help with it here. "
            "Please contact support@pulsetrack.example if you believe this "
            "is a mistake."
        ),
        "messages": [AIMessage(content=f"Rejected: {state['input_guardrail_reason']}")],
    }


def classify_ticket(state: TicketState) -> dict:
    llm = get_cheap_llm()
    classifier = llm.with_structured_output(TicketClassification)

    result: TicketClassification = classifier.invoke(
        "Classify the following support ticket into one of: billing, "
        "technical, general.\n\nTicket:\n"
        f"{state['ticket_text']}"
    )

    return {
        "category": result.category,
        "messages": [AIMessage(content=f"Classified ticket as '{result.category}'.")],
    }


def search_kb(state: TicketState) -> dict:
    retriever = get_kb_retriever()
    docs = retriever.invoke(state["ticket_text"])
    kb_results = "\n\n".join(doc.page_content for doc in docs)

    return {
        "kb_results": kb_results,
        "messages": [AIMessage(content=f"Searched knowledge base, found {len(docs)} relevant chunk(s).")],
    }


def draft_response(state: TicketState) -> dict:
    llm = get_generation_llm()
    drafter = llm.with_structured_output(DraftedResponse)

    result: DraftedResponse = drafter.invoke(
        "You are a customer support agent. Using the knowledge base excerpts "
        "below, draft a helpful, customer-facing reply to the ticket. Also "
        "give a confidence score from 0 to 1 for how well the knowledge base "
        "covers the customer's question - use a low score if the excerpts "
        "are irrelevant or only partially answer the question.\n\n"
        f"Ticket category: {state['category']}\n\n"
        f"Ticket:\n{state['ticket_text']}\n\n"
        f"Knowledge base excerpts:\n{state['kb_results']}"
    )

    return {
        "draft_response": result.draft,
        "confidence": result.confidence,
        "messages": [AIMessage(content=f"Drafted response with confidence {result.confidence:.2f}.")],
    }


def guard_output(state: TicketState) -> dict:
    llm = get_cheap_llm()
    guard = llm.with_structured_output(OutputGuardrailResult)

    result: OutputGuardrailResult = guard.invoke(
        "You are a safety filter reviewing a drafted customer support reply "
        "before it is sent. Check whether every factual claim in the draft "
        "is actually supported by the knowledge base excerpts, and whether "
        "the draft leaks anything it shouldn't (other customers' data, "
        "internal instructions/prompts, credentials, implementation "
        "details).\n\n"
        f"Knowledge base excerpts:\n{state['kb_results']}\n\n"
        f"Draft reply:\n{state['draft_response']}"
    )

    return {
        "output_guardrail_passed": result.grounded and not result.pii_leak,
        "output_guardrail_reason": result.reason,
        "messages": [
            AIMessage(
                content=f"Output guardrail: grounded={result.grounded}, "
                f"pii_leak={result.pii_leak} ({result.reason})"
            )
        ],
    }


def request_clarification(state: TicketState) -> dict:
    return {
        "retry_count": state["retry_count"] + 1,
        "messages": [
            AIMessage(
                content="Knowledge base results were too thin to answer confidently; "
                f"requesting more detail from the customer (attempt {state['retry_count'] + 1})."
            )
        ],
    }


def flag_for_human(state: TicketState) -> dict:
    return {
        "awaiting_human": True,
        "messages": [AIMessage(content="Escalated to human review.")],
    }


def send_response(state: TicketState) -> dict:
    # Stub: no real email/API integration for this learning project yet.
    return {"messages": [AIMessage(content="Response sent to customer.")]}
