"""
Evaluate the KB pipeline against the golden dataset with RAGAS's four core
metrics: faithfulness, answer_relevancy, context_precision, context_recall.

Runs two pipelines so an improvement can be confirmed by re-running eval:
  - naive_pipeline: plain top-k similarity search, generate from every
    retrieved chunk unfiltered (the Unit 6-ish baseline).
  - reranked_pipeline: over-fetch candidates and cross-encoder rerank them
    down to k (data/kb_docs' existing rerank=True path in ingest.py) - a
    retrieval-side fix for when context_precision/recall are the weak link.

Run:
    python -m app.rag.eval_ragas
"""

from __future__ import annotations

import sys
import types

# ragas unconditionally imports langchain_community.chat_models.vertexai,
# which no longer ships in current langchain-community (Vertex AI support
# moved to the standalone langchain-google-vertexai package). We never use
# Vertex AI, so stub the module out rather than pulling in that dependency.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # unused placeholder, just needs to be importable
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from app.llm import get_generation_llm
from app.rag.golden_dataset import GOLDEN_DATASET
from app.rag.ingest import get_kb_retriever

# Defaults (max_workers=16, timeout=180) fire too many concurrent judge
# calls at once and trip rate limits; context_precision in particular judges
# each retrieved chunk with its own LLM call, so a 5-chunk row can take
# several sequential calls. Lower concurrency and a generous timeout avoid
# TimeoutErrors turning into silent NaN scores.
_RUN_CONFIG = RunConfig(timeout=300, max_workers=2)

# Pipeline's own answer generation - the "generation" role, best model.
_llm = get_generation_llm()
# RAGAS's judge model for grading faithfulness/relevancy/precision/recall -
# a "grading" role, but left on OpenAI rather than the local Ollama model:
# RAGAS's judge prompts expect reliable structured-output following, which
# hasn't been validated against a 3B local model, and re-validating would
# mean re-running the ~45min eval blind. gpt-4o-mini is still the cheaper
# OpenAI tier, just not the free local one.
_eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_eval_embeddings = OpenAIEmbeddings()

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _generate(question: str, contexts: list[str]) -> str:
    context = "\n\n".join(contexts)
    if not context.strip():
        return "I don't have information about that in the knowledge base."
    return _llm.invoke(
        "Answer the question using only the context below. If the context "
        "doesn't actually contain the answer, say so plainly instead of "
        "guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    ).content


def naive_pipeline(question: str) -> tuple[str, list[str]]:
    """Plain top-5 similarity search; every retrieved chunk goes into the prompt unfiltered."""
    retriever = get_kb_retriever(k=5)
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    return _generate(question, contexts), contexts


def reranked_pipeline(question: str) -> tuple[str, list[str]]:
    """Over-fetch then cross-encoder rerank down to k - retrieval-side fix."""
    retriever = get_kb_retriever(k=3, rerank=True, rerank_candidates=9)
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    return _generate(question, contexts), contexts


def build_eval_records(pipeline_fn) -> list[dict]:
    records = []
    for item in GOLDEN_DATASET:
        answer, contexts = pipeline_fn(item["question"])
        records.append(
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )
    return records


def run_eval(records: list[dict]):
    dataset = EvaluationDataset.from_list(records)
    return evaluate(
        dataset, metrics=METRICS, llm=_eval_llm, embeddings=_eval_embeddings, run_config=_RUN_CONFIG
    )


def print_report(label: str, result, records: list[dict]) -> dict:
    df = result.to_pandas()
    means = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()

    print(f"\n=== {label}: aggregate scores ===")
    for metric, score in means.items():
        print(f"  {metric:<20} {score:.3f}")

    print(f"\n=== {label}: per-question breakdown ===")
    for i, row in df.iterrows():
        note = GOLDEN_DATASET[i].get("note", "")
        tag = f"  [{note.split(' - ')[0]}]" if note else ""
        print(f"  Q: {row['user_input']!r}{tag}")
        print(
            f"    faithfulness={row['faithfulness']:.2f}  "
            f"answer_relevancy={row['answer_relevancy']:.2f}  "
            f"context_precision={row['context_precision']:.2f}  "
            f"context_recall={row['context_recall']:.2f}"
        )

    return means.to_dict()


if __name__ == "__main__":
    print("Running naive_pipeline over the golden dataset...")
    naive_records = build_eval_records(naive_pipeline)
    naive_result = run_eval(naive_records)
    naive_means = print_report("naive_pipeline", naive_result, naive_records)

    print("\nRunning reranked_pipeline over the golden dataset...")
    reranked_records = build_eval_records(reranked_pipeline)
    reranked_result = run_eval(reranked_records)
    reranked_means = print_report("reranked_pipeline", reranked_result, reranked_records)

    print("\n=== naive_pipeline vs. reranked_pipeline ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        delta = reranked_means[metric] - naive_means[metric]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<20} {naive_means[metric]:.3f} -> {reranked_means[metric]:.3f}  ({sign}{delta:.3f})")
