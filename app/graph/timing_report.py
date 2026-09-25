"""
Run 5 sample tickets through the (now-instrumented) graph and report which
stage consistently eats the most latency.

Run:
    python -m app.graph.timing_report
"""

from __future__ import annotations

from collections import defaultdict

from langgraph.checkpoint.memory import MemorySaver

from app.graph.build_graph import build_graph
from app.graph.timing import NODE_TIMINGS, reset_timings

INITIAL_STATE_DEFAULTS = {
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

QUERIES = [
    ("q1", "How do I reset my password? The link never showed up."),
    ("q2", "How much does PulseTrack Plus cost per month?"),
    ("q3", "What's the flat rate for expedited shipping?"),
    ("q4", "What's your policy on refunds?"),
    ("q5", "My PulseTrack device won't pair with my new phone, what do I do?"),
]


def main() -> None:
    reset_timings()
    graph = build_graph(MemorySaver())

    for ticket_id, text in QUERIES:
        config = {"configurable": {"thread_id": ticket_id}}
        state = {**INITIAL_STATE_DEFAULTS, "ticket_id": ticket_id, "ticket_text": text}
        graph.invoke(state, config)

    by_node = defaultdict(list)
    for record in NODE_TIMINGS:
        by_node[record["node"]].append(record["seconds"])

    rows = [
        (node, len(times), sum(times) / len(times), min(times), max(times), sum(times))
        for node, times in by_node.items()
    ]
    rows.sort(key=lambda r: r[2], reverse=True)  # sort by mean latency, descending

    print("\n=== Per-node latency across 5 queries ===")
    print(f"{'node':<24}{'n':>4}{'mean(s)':>10}{'min(s)':>10}{'max(s)':>10}{'total(s)':>10}")
    for node, n, mean, lo, hi, total in rows:
        print(f"{node:<24}{n:>4}{mean:>10.3f}{lo:>10.3f}{hi:>10.3f}{total:>10.3f}")

    slowest = rows[0]
    print(f"\nSlowest stage by mean latency: {slowest[0]} ({slowest[2]:.3f}s avg over {slowest[1]} calls)")
    print(f"Total graph time across all 5 queries: {sum(r[5] for r in rows):.3f}s")


if __name__ == "__main__":
    main()
