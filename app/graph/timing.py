"""
Per-node latency instrumentation for the ticket-triage LangGraph app.

Wraps each node function with a timer so individual-stage latency is
visible, instead of only seeing total request time.
"""

from __future__ import annotations

import functools
import time

NODE_TIMINGS: list[dict] = []


def reset_timings() -> None:
    NODE_TIMINGS.clear()


def timed_node(name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            start = time.perf_counter()
            result = fn(state)
            elapsed = time.perf_counter() - start
            NODE_TIMINGS.append({"node": name, "ticket_id": state.get("ticket_id", ""), "seconds": elapsed})
            print(f"[timing] {name:<22} {elapsed:6.3f}s  (ticket={state.get('ticket_id', '')})")
            return result

        return wrapper

    return decorator
