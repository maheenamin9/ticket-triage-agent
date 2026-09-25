"""
Shared LLM factories, split by role:
  - get_cheap_llm: grading/routing/rewrite calls (short, structured-output
    judgments) - a free local Ollama model, no per-call cost.
  - get_generation_llm: customer-facing generation - the paid OpenAI model.
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_cheap_llm(temperature: float = 0) -> ChatOllama:
    settings = get_settings()
    return ChatOllama(model=settings.cheap_model, base_url=settings.ollama_base_url, temperature=temperature)


def get_generation_llm(temperature: float = 0) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(model=settings.generation_model, temperature=temperature)
