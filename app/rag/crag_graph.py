"""
CRAG as an actual LangGraph graph with cycles (vs. the plain-Python-if
version in app.rag.crag):

    retrieve -> grade_documents -> (generate | rewrite_query | web_search_fallback)

grade_documents routes to generate when it found relevant docs, to
rewrite_query (which loops back to retrieve) when it found none and retries
remain, or to web_search_fallback once retries are exhausted.

Uses the same Chroma store/embeddings as app.rag.ingest, and DuckDuckGo
search (via `ddgs`) as the web fallback.

Run:
    python -m app.rag.crag_graph
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from ddgs import DDGS
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.llm import get_cheap_llm, get_generation_llm
from app.rag.ingest import PERSIST_DIRECTORY

# Deliberately small so a vaguely-phrased query can miss the right chunk on
# the first pass in this 9-chunk corpus, giving the rewrite loop something
# real to do.
TOP_K = 3
MAX_REWRITES = 2

_embeddings = OpenAIEmbeddings()
_cheap_llm = get_cheap_llm()
_generation_llm = get_generation_llm()


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    retry_count: int


def _vectorstore() -> Chroma:
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=_embeddings)


def _label(doc: Document) -> str:
    source = doc.metadata.get("source", "web")
    if source != "web":
        source = Path(source).name
    return f"[{source}] {doc.page_content[:70].strip()!r}..."


class GradeResult(BaseModel):
    relevant: bool = Field(description="True only if the document actually helps answer the question")


# --- nodes -------------------------------------------------------------


def retrieve(state: RAGState) -> dict:
    print(f"[retrieve] query={state['question']!r}")
    docs = [d for d, _ in _vectorstore().similarity_search_with_score(state["question"], k=TOP_K)]
    for d in docs:
        print(f"  retrieved {_label(d)}")
    return {"documents": docs}


def grade_documents(state: RAGState) -> dict:
    grader = _cheap_llm.with_structured_output(GradeResult)
    relevant = []
    for doc in state["documents"]:
        result: GradeResult = grader.invoke(
            "You are grading whether a retrieved document is relevant to a user "
            "question. Be strict: only mark it relevant if it contains information "
            "that actually helps answer the question, not just shared topic words.\n\n"
            f"Question: {state['question']}\n\nDocument:\n{doc.page_content}"
        )
        tag = "RELEVANT  " if result.relevant else "irrelevant"
        print(f"  [grade_documents] {tag} {_label(doc)}")
        if result.relevant:
            relevant.append(doc)
    print(f"[grade_documents] {len(relevant)}/{len(state['documents'])} relevant")
    return {"documents": relevant}


def rewrite_query(state: RAGState) -> dict:
    rewritten = _cheap_llm.invoke(
        "The search query below returned no relevant documents from a support "
        "knowledge base about PulseTrack account/billing/shipping policies. "
        "Rewrite it as a clearer, more specific search query using the kind of "
        "vocabulary a policy document would use. Return only the rewritten "
        "query, no commentary.\n\n"
        f"Original query: {state['question']}"
    ).content.strip()
    print(f"[rewrite_query] {state['question']!r} -> {rewritten!r}")
    return {"question": rewritten, "retry_count": state["retry_count"] + 1}


def web_search_fallback(state: RAGState) -> dict:
    print(f"[web_search_fallback] query={state['question']!r}")
    results = DDGS().text(state["question"], max_results=3)
    docs = [
        Document(page_content=f"{r['title']}\n{r['body']}", metadata={"source": "web"})
        for r in results
    ]
    for d in docs:
        print(f"  web {_label(d)}")
    return {"documents": docs}


def generate(state: RAGState) -> dict:
    context = "\n\n".join(d.page_content for d in state["documents"])
    if context:
        answer = _generation_llm.invoke(
            "Answer the question using only the context below. If the context "
            "doesn't actually contain the answer, say so plainly instead of "
            "guessing.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}"
        ).content
    else:
        answer = "No relevant information found."
    print(f"[generate] {answer}")
    return {"generation": answer}


# --- conditional edge ---------------------------------------------------


def route_after_grade(state: RAGState) -> str:
    if state["documents"]:
        return "generate"
    if state["retry_count"] < MAX_REWRITES:
        return "rewrite"
    return "web_search"


def build_crag_graph(checkpointer):
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("generate", generate)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("web_search_fallback", web_search_fallback)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "grade_documents")

    builder.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite_query", "web_search": "web_search_fallback"},
    )
    # Loops back to retrieve with the rewritten query; route_after_grade's
    # retry_count check (capped at MAX_REWRITES) is what stops this from
    # cycling forever.
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("web_search_fallback", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_crag_graph(MemorySaver())
    config = {"configurable": {"thread_id": "crag-demo-1"}}

    initial_state: RAGState = {
        "question": "It says my hardware is already claimed by someone else when I try to set it up",
        "documents": [],
        "generation": "",
        "retry_count": 0,
    }
    graph.invoke(initial_state, config)

    print("\n=== get_state_history(thread_id='crag-demo-1') ===")
    for snapshot in reversed(list(graph.get_state_history(config))):
        print(
            f"step={snapshot.metadata['step']:>2} "
            f"just_ran={snapshot.metadata.get('writes') and list(snapshot.metadata['writes'].keys())} "
            f"next={snapshot.next} "
            f"retry_count={snapshot.values.get('retry_count')} "
            f"question={snapshot.values.get('question')!r}"
        )
