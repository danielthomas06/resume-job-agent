from jobagent.config import Settings, get_settings

from .base import LLMClient, LLMError
from .gemini import GeminiClient
from .ollama import OllamaClient


def build_client(provider: str, s: Settings) -> LLMClient:
    if provider == "local":
        return OllamaClient(s.ollama_host, s.ollama_model, s.llm_timeout_s, s.ollama_num_ctx)
    if provider == "gemini":
        return GeminiClient(s.gemini_api_key, s.gemini_model)
    raise ValueError(f"unknown provider {provider}")


def get_llm(provider: str | None = None, model: str | None = None) -> LLMClient:
    s = get_settings()
    if model:  # override the model for this client only (e.g. a bigger local model as judge)
        s = s.model_copy(update={"ollama_model": model, "gemini_model": model})
    p = provider or s.llm_provider
    if p == "auto":
        return FallbackClient(build_client("local", s), lambda: build_client("gemini", s))
    return build_client(p, s)


class FallbackClient(LLMClient):
    """Try local first; on failure, lazily build and use the remote client."""

    provider = "auto"

    def __init__(self, primary: LLMClient, make_secondary):
        self.primary, self._make_secondary, self._secondary = primary, make_secondary, None
        self.model = primary.model
        self.is_remote = False

    def _generate(self, system, user, schema, temperature):
        try:
            return self.primary._generate(system, user, schema, temperature)
        except LLMError:
            if self._secondary is None:
                self._secondary = self._make_secondary()
            self.is_remote = True
            return self._secondary._generate(system, user, schema, temperature)
