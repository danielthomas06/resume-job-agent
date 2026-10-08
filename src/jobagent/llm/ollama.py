import time

import httpx

from .base import LLMClient, LLMError, LLMResult


class OllamaClient(LLMClient):
    provider = "ollama"
    is_remote = False

    def __init__(self, host: str, model: str, timeout_s: float = 180.0, num_ctx: int = 8192):
        self.host, self.model, self.timeout_s, self.num_ctx = host.rstrip("/"), model, timeout_s, num_ctx

    def _generate(self, system, user, schema, temperature) -> LLMResult:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},  # explicit: runtime default may be smaller than the model supports
        }
        if schema:
            body["format"] = schema  # Ollama enforces JSON schema via grammar
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_s)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        d = r.json()
        return LLMResult(
            text=d["message"]["content"],
            model=self.model,
            provider=self.provider,
            input_tokens=d.get("prompt_eval_count", 0),
            output_tokens=d.get("eval_count", 0),
            latency_s=time.perf_counter() - t0,
            context_window=self.num_ctx,
            finish_reason=d.get("done_reason", ""),
        )
