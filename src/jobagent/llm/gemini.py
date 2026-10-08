import time

from google import genai
from google.genai import types

from .base import LLMClient, LLMError, LLMResult

# Input context windows (tokens); unknown models fall back to 1M
CONTEXT_WINDOWS = {"gemini-2.5-flash": 1_048_576, "gemini-2.5-pro": 1_048_576, "gemini-2.0-flash": 1_048_576}


class GeminiClient(LLMClient):
    provider = "gemini"
    is_remote = True

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def _generate(self, system, user, schema, temperature) -> LLMResult:
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if schema else None,
            response_json_schema=schema,
        )
        t0 = time.perf_counter()
        try:
            r = self._client.models.generate_content(model=self.model, contents=user, config=cfg)
        except Exception as e:  # SDK raises several error types
            raise LLMError(f"Gemini request failed: {e}") from e
        u = r.usage_metadata
        return LLMResult(
            text=r.text or "",
            model=self.model,
            provider=self.provider,
            input_tokens=getattr(u, "prompt_token_count", 0) or 0,
            output_tokens=getattr(u, "candidates_token_count", 0) or 0,
            latency_s=time.perf_counter() - t0,
            context_window=CONTEXT_WINDOWS.get(self.model, 1_000_000),
            finish_reason=str(getattr(r.candidates[0].finish_reason, "name", "")) if r.candidates else "",
            reasoning_tokens=getattr(u, "thoughts_token_count", 0) or 0,  # Gemini 2.5 "thinking" tokens
        )
