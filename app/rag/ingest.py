from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from app.llm import get_cheap_llm

load_dotenv()

PERSIST_DIRECTORY = "./data/chroma_kb"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder() -> HuggingFaceCrossEncoder:
    # Cached so the model weights load once per process, not once per request.
    return HuggingFaceCrossEncoder(model_name=RERANK_MODEL_NAME)


def _load_docs(folder_path: str):
    docs = []
    for path in sorted(Path(folder_path).glob("*")):
        if path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
    return docs


def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def ingest_kb_docs(folder_path: str = "data/kb_docs") -> Chroma:
    docs = _load_docs(folder_path)
    chunks = _split_docs(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )
    print(f"Ingested {len(docs)} document(s) into {len(chunks)} chunks -> {PERSIST_DIRECTORY}")
    return vectorstore


def get_kb_retriever(
    k: int = 3,
    search_type: str = "similarity",
    score_threshold: float | None = None,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    folder_path: str = "data/kb_docs",
    dense_weight: float = 0.5,
    rerank: bool = False,
    rerank_candidates: int = 10,
    multi_query: bool = False,
):
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings(),
    )

    # When reranking, over-fetch candidates with the base retriever so the
    # reranker has a real pool to choose the final top-k from.
    base_k = rerank_candidates if rerank else k

    if search_type == "hybrid":
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": base_k})

        bm25_retriever = BM25Retriever.from_documents(_split_docs(_load_docs(folder_path)))
        bm25_retriever.k = base_k

        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[1 - dense_weight, dense_weight],
        )
    else:
        search_kwargs = {"k": base_k}
        if search_type == "similarity_score_threshold":
            if score_threshold is None:
                raise ValueError("score_threshold is required when search_type='similarity_score_threshold'")
            search_kwargs["score_threshold"] = score_threshold
        elif search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k
            search_kwargs["lambda_mult"] = lambda_mult

        base_retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    # Rephrases the query into several variants, runs each through base_retriever,
    # and unions the (deduped) results — so the final count can exceed base_k.
    if multi_query:
        base_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=get_cheap_llm(),
        )

    if not rerank:
        return base_retriever

    reranker = CrossEncoderReranker(model=_get_cross_encoder(), top_n=k)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


if __name__ == "__main__":
    ingest_kb_docs()
