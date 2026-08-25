"""
Central, env-driven configuration for paraAI.

Every other module pulls its settings from `settings = Settings()` here
rather than reading `os.environ` directly, so the whole app can be
re-pointed (different model provider, different vector store path, etc.)
by editing `.env` alone.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    model_provider: str = "anthropic"          # "anthropic" | "openai"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # --- Indian Kanoon ---
    ik_api_token: str = ""
    ik_base_url: str = "https://api.indiankanoon.org"

    # --- RAG ---
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    vector_store_path: str = "./data/vector_store"
    watch_dir: str = "./data/incoming"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_mb: int = 50

    def ensure_dirs(self) -> None:
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
        Path(self.watch_dir).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def get_chat_model():
    """
    Returns a LangChain chat model instance based on MODEL_PROVIDER.
    Kept in one place so every agent stays model-agnostic — agents import
    `get_chat_model()`, never a specific provider class.
    """
    settings = get_settings()

    if settings.model_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=4096,
        )
    elif settings.model_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {settings.model_provider}")
