"""
Simplified Corrective RAG (CRAG) loop, plain Python control flow (no LangGraph).

    retrieve -> grade each doc relevant/irrelevant -> if zero relevant,
    fall back to a web search tool instead of generating from empty context.

Uses the same Chroma store/embeddings as app.rag.ingest (data/kb_docs), and
DuckDuckGo search (via `ddgs`, no API key needed) as the web fallback.

Run:
    python -m app.rag.crag
"""

from __future__ import annotations

from pathlib import Path

from ddgs import DDGS
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from app.llm import get_cheap_llm, get_generation_llm
from app.rag.ingest import PERSIST_DIRECTORY

TOP_K = 5

_embeddings = OpenAIEmbeddings()
_cheap_llm = get_cheap_llm()
_generation_llm = get_generation_llm()


def _vectorstore() -> Chroma:
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=_embeddings)


def _label(doc: Document) -> str:
    source = Path(doc.metadata.get("source", "?")).name if "source" in doc.metadata else "web"
    return f"[{source}] {doc.page_content[:70].strip()!r}..."


class GradeResult(BaseModel):
    relevant: bool = Field(description="True only if the document actually helps answer the question")


def retrieve(query: str, k: int = TOP_K) -> list[Document]:
    return [d for d, _ in _vectorstore().similarity_search_with_score(query, k=k)]


def grade_document(query: str, doc: Document) -> bool:
    grader = _cheap_llm.with_structured_output(GradeResult)
    result: GradeResult = grader.invoke(
        "You are grading whether a retrieved document is relevant to a user question.\n"
        "Be strict: only mark it relevant if it contains information that actually "
        "helps answer the question, not just shared topic words.\n\n"
        f"Question: {query}\n\nDocument:\n{doc.page_content}"
    )
    return result.relevant


def grade_documents(query: str, docs: list[Document]) -> tuple[list[Document], list[Document]]:
    relevant, irrelevant = [], []
    for doc in docs:
        (relevant if grade_document(query, doc) else irrelevant).append(doc)
    return relevant, irrelevant


def web_search(query: str, max_results: int = 3) -> list[Document]:
    results = DDGS().text(query, max_results=max_results)
    return [
        Document(page_content=f"{r['title']}\n{r['body']}", metadata={"source": r["href"]})
        for r in results
    ]


def generate_answer(query: str, docs: list[Document], source: str) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    return _generation_llm.invoke(
        f"Answer the question using only the context below, which came from {source}.\n"
        "If the context doesn't actually contain the answer, say so plainly instead "
        "of guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    ).content


def crag(query: str) -> str:
    print(f"\n=== CRAG: {query!r} ===")

    docs = retrieve(query)
    print(f"Retrieved {len(docs)} doc(s) from KB:")
    for d in docs:
        print(f"  {_label(d)}")

    relevant, irrelevant = grade_documents(query, docs)
    print(f"Graded: {len(relevant)} relevant / {len(irrelevant)} irrelevant")
    for d in relevant:
        print(f"  relevant:   {_label(d)}")
    for d in irrelevant:
        print(f"  irrelevant: {_label(d)}")

    if relevant:
        print("-> KB has relevant context, generating from KB")
        answer = generate_answer(query, relevant, source="the internal knowledge base")
    else:
        print("-> zero relevant KB docs, falling back to web search")
        web_docs = web_search(query)
        for d in web_docs:
            print(f"  web: {_label(d)}")
        if web_docs:
            answer = generate_answer(query, web_docs, source="a web search")
        else:
            answer = "No relevant information found in the knowledge base or the web."

    print(f"\nAnswer:\n{answer}")
    return answer


if __name__ == "__main__":
    crag("How long is a password reset link valid for?")
    crag("What is the current world record time for the marathon?")
