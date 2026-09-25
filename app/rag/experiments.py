"""
Hands-on exercise: HyDE and RAG Fusion vs. direct retrieval.

Uses the same Chroma store/embeddings as app.rag.ingest, populated from
data/kb_docs (PulseTrack account/billing/shipping policies).

Run:
    python -m app.rag.experiments
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.llm import get_cheap_llm
from app.rag.ingest import PERSIST_DIRECTORY

TOP_K = 5
RRF_K = 60  # standard RRF damping constant

_embeddings = OpenAIEmbeddings()
# Both calls below are retrieval-support (produce text used only to embed a
# better query), not customer-facing generation, so the cheap tier applies.
_llm = get_cheap_llm()


def _vectorstore() -> Chroma:
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=_embeddings)


def _doc_key(doc: Document) -> str:
    return f"{doc.metadata.get('source', '?')}|{doc.page_content[:60]}"


def _fmt(doc: Document, score: float | None = None) -> str:
    source = Path(doc.metadata.get("source", "?")).name
    snippet = doc.page_content.strip().replace("\n", " ")[:100]
    score_str = f" score={score:.4f}" if score is not None else ""
    return f"  [{source}]{score_str} {snippet}..."


def direct_retrieve(question: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Baseline: embed the raw question and search."""
    return _vectorstore().similarity_search_with_score(question, k=k)


def hyde_retrieve(question: str, k: int = TOP_K) -> tuple[str, list[tuple[Document, float]]]:
    """HyDE: generate a hypothetical answer, embed *that*, and search."""
    hypothetical = _llm.invoke(
        "Write a short, confident paragraph (3-5 sentences) that directly answers "
        "the question below, phrased as if it were an excerpt from an official "
        "company policy document. State specifics even if you have to invent "
        "plausible ones - do not hedge, and do not say you don't know.\n\n"
        f"Question: {question}"
    ).content
    results = _vectorstore().similarity_search_with_score(hypothetical, k=k)
    return hypothetical, results


def generate_query_variants(question: str, n: int = 4) -> list[str]:
    """RAG Fusion step 1: generate n reformulations covering different angles."""
    text = _llm.invoke(
        f"Generate {n} different search-query rephrasings of the question below, "
        "so that together they cover different plausible angles or interpretations "
        "a user might mean. One per line, no numbering, no extra commentary.\n\n"
        f"Question: {question}"
    ).content
    variants = [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
    return variants[:n]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = RRF_K
) -> list[tuple[Document, float]]:
    """RAG Fusion step 2: combine multiple rankings with RRF."""
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_by_key.setdefault(key, doc)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(doc_by_key[key], score) for key, score in fused]


def rag_fusion_retrieve(
    question: str, k: int = TOP_K
) -> tuple[list[str], list[tuple[Document, float]]]:
    variants = generate_query_variants(question)
    all_queries = [question] + variants
    vs = _vectorstore()
    ranked_lists = [[d for d, _ in vs.similarity_search_with_score(q, k=k)] for q in all_queries]
    fused = reciprocal_rank_fusion(ranked_lists)
    return all_queries, fused[:k]


# ---------------------------------------------------------------------------


def demo_hyde(question: str) -> None:
    print(f"\n=== HyDE vs. direct: {question!r} ===")
    direct = direct_retrieve(question)
    print("Direct (raw question embedded):")
    for doc, score in direct:
        print(_fmt(doc, score))

    hypothetical, hyde = hyde_retrieve(question)
    print(f"\nHypothetical answer generated:\n  {hypothetical!r}")
    print("HyDE (hypothetical answer embedded):")
    for doc, score in hyde:
        print(_fmt(doc, score))


def demo_rag_fusion(question: str) -> None:
    print(f"\n=== RAG Fusion vs. direct: {question!r} ===")
    direct = direct_retrieve(question)
    print("Direct top-5 (single query):")
    for doc, score in direct:
        print(_fmt(doc, score))

    queries, fused = rag_fusion_retrieve(question)
    print(f"\nQuery variants used: {queries}")
    print("RAG Fusion top-5 (RRF over all variants):")
    for doc, score in fused:
        print(_fmt(doc, score))


def demo_precise_query(question: str) -> None:
    print(f"\n=== Precise/narrow query: {question!r} ===")
    direct = direct_retrieve(question)
    print("Direct:")
    for doc, score in direct:
        print(_fmt(doc, score))

    hypothetical, hyde = hyde_retrieve(question)
    print(f"\nHypothetical answer generated:\n  {hypothetical!r}")
    print("HyDE:")
    for doc, score in hyde:
        print(_fmt(doc, score))

    queries, fused = rag_fusion_retrieve(question)
    print(f"\nQuery variants used: {queries}")
    print("RAG Fusion:")
    for doc, score in fused:
        print(_fmt(doc, score))


if __name__ == "__main__":
    conceptual_questions = [
        "How does PulseTrack handle it when a customer is unhappy with a purchase?",
        "What happens if something goes wrong between placing an order and it arriving?",
        "How does PulseTrack keep a customer's account safe from unauthorized access?",
    ]
    for q in conceptual_questions:
        demo_hyde(q)

    demo_rag_fusion("What's your policy on refunds?")

    demo_precise_query("What is the flat rate for expedited shipping?")
