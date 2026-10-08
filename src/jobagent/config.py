from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["local", "gemini", "auto"] = "local"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_num_ctx: int = 8192  # context window requested from Ollama
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    redact_before_remote: bool = True
    otel_exporter_otlp_endpoint: str = ""
    otel_project: str = "jobagent"
    phoenix_public_url: str = ""  # browser-facing Phoenix URL; defaults to the OTLP endpoint
    llm_timeout_s: float = 180.0
    llm_max_retries: int = 2
    embed_model: str = "nomic-embed-text"
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    greenhouse_boards: str = ""
    lever_boards: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
