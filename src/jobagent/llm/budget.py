from dataclasses import dataclass, field

from .base_errors import LLMError


class BudgetExceeded(LLMError):
    pass


@dataclass
class Budget:
    """Per-run guard against runaway loops/cost. Attached to an LLMClient; also meters every call."""

    max_calls: int = 80
    max_tokens: int = 400_000
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    log: list[dict] = field(default_factory=list)  # one row per LLM request
    on_event: object = None  # optional callable(dict): live progress hook (UI/CLI)

    def _emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # a broken progress display must never break the run
                pass

    def started(self, purpose: str, attempt: int, model: str) -> None:
        self._emit({"type": "llm_start", "purpose": purpose, "attempt": attempt, "model": model})

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget exhausted ({self.max_calls})")
        if self.input_tokens + self.output_tokens >= self.max_tokens:
            raise BudgetExceeded(f"LLM token budget exhausted ({self.max_tokens})")

    def record(self, res, purpose: str = "", attempt: int = 1, truncated: bool = False) -> None:
        self.calls += 1
        self.input_tokens += res.input_tokens
        self.output_tokens += res.output_tokens
        self.latency_s += res.latency_s
        used = res.input_tokens + res.output_tokens
        row = {
            "n": self.calls, "purpose": purpose, "attempt": attempt, "provider": res.provider, "model": res.model,
            "prompt_tokens": res.input_tokens, "completion_tokens": res.output_tokens,
            "reasoning_tokens": res.reasoning_tokens, "context_window": res.context_window,
            "context_utilization": round(used / res.context_window, 3) if res.context_window else None,
            "finish_reason": res.finish_reason, "latency_s": round(res.latency_s, 2), "truncated": truncated,
        }
        self.log.append(row)
        self._emit({"type": "llm_end", "row": row, "totals": {
            "calls": self.calls, "prompt": self.input_tokens, "completion": self.output_tokens}})

    def summary(self) -> dict:
        utils = [r["context_utilization"] for r in self.log if r["context_utilization"] is not None]
        return {"llm_calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "llm_latency_s": round(self.latency_s, 1), "peak_context_utilization": max(utils, default=None),
                "truncated_calls": sum(r["truncated"] for r in self.log), "calls": self.log}
