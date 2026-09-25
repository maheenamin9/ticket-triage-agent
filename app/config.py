from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "ticket-triage-agent"

    # Model tiers: grading/routing/rewrite calls (short, structured-output
    # judgments) run on a free local Ollama model; customer-facing
    # generation stays on the paid OpenAI model. Centralized here so every
    # script picks the same pair rather than hardcoding model strings.
    cheap_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    generation_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
