import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from jobagent.observability import tracer

from .base_errors import LLMError  # re-exported

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("jobagent.llm")
TRACE_TEXT_LIMIT = 30_000  # prompts are stored in full on spans; this only guards pathological sizes

__all__ = ["LLMClient", "LLMError", "LLMResult"]


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    context_window: int = 0  # tokens the model/runtime will actually accept (0 = unknown)
    finish_reason: str = ""  # stop | length | ...
    reasoning_tokens: int = 0  # hidden "thinking" tokens, when the provider reports them
    meta: dict = field(default_factory=dict)


class LLMClient(ABC):
    provider: str
    model: str
    is_remote: bool = False
    budget = None  # optional llm.budget.Budget, set per run

    @abstractmethod
    def _generate(
        self, system: str, user: str, schema: dict | None, temperature: float
    ) -> LLMResult: ...

    def generate(
        self, system: str, user: str, *, schema: dict | None = None, temperature: float = 0.0,
        purpose: str = "", attempt: int = 1,
    ) -> LLMResult:
        with tracer().start_as_current_span(f"llm.{purpose or self.provider}") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.provider)
            span.set_attribute("llm.model_name", self.model)
            span.set_attribute("llm.purpose", purpose)
            span.set_attribute("llm.attempt", attempt)
            span.set_attribute("llm.temperature", temperature)
            span.set_attribute("llm.structured_output", schema is not None)
            # full prompts, in OpenInference message form so Phoenix renders the conversation
            span.set_attribute("llm.input_messages.0.message.role", "system")
            span.set_attribute("llm.input_messages.0.message.content", system[:TRACE_TEXT_LIMIT])
            span.set_attribute("llm.input_messages.1.message.role", "user")
            span.set_attribute("llm.input_messages.1.message.content", user[:TRACE_TEXT_LIMIT])
            span.set_attribute("input.value", user[:TRACE_TEXT_LIMIT])
            if self.budget:
                self.budget.check()
                self.budget.started(purpose, attempt, self.model)
            res = self._generate(system, user, schema, temperature)

            used = res.input_tokens + res.output_tokens
            util = used / res.context_window if res.context_window else 0.0
            # a prompt that fills the window is silently cut by the runtime: flag it loudly
            truncated = bool(res.context_window) and (
                res.input_tokens >= 0.98 * res.context_window or res.finish_reason == "length")
            if truncated:
                log.warning("LLM call %s/%s may have been truncated: %d prompt tokens, window %d, finish=%s",
                            self.provider, purpose or "-", res.input_tokens, res.context_window, res.finish_reason)
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute("llm.output_messages.0.message.content", res.text[:TRACE_TEXT_LIMIT])
            span.set_attribute("output.value", res.text[:TRACE_TEXT_LIMIT])
            span.set_attribute("llm.token_count.prompt", res.input_tokens)
            span.set_attribute("llm.token_count.completion", res.output_tokens)
            span.set_attribute("llm.token_count.total", used)
            span.set_attribute("llm.token_count.completion_details.reasoning", res.reasoning_tokens)
            span.set_attribute("llm.context_window", res.context_window)
            span.set_attribute("llm.context_utilization", round(util, 4))
            span.set_attribute("llm.input_truncated", truncated)
            span.set_attribute("llm.finish_reason", res.finish_reason)
            span.set_attribute("llm.latency_s", res.latency_s)
            if used and res.latency_s:
                span.set_attribute("llm.tokens_per_second", round(res.output_tokens / res.latency_s, 1))
            if self.budget:
                self.budget.record(res, purpose=purpose, attempt=attempt, truncated=truncated)
            return res

    def generate_structured(
        self, system: str, user: str, model_cls: type[T], *, max_retries: int = 2, purpose: str = ""
    ) -> T:
        """Generate JSON constrained to model_cls; on validation failure feed the error back."""
        schema = model_cls.model_json_schema()
        # Small local models need the schema in the prompt too, not only as a decoding constraint.
        system = f"{system}\n\nRespond with JSON matching this schema; fill every field you can:\n{json.dumps(schema)}"
        with tracer().start_as_current_span(f"structured.{purpose or model_cls.__name__}") as span:
            span.set_attribute("llm.schema", model_cls.__name__)
            prompt, last_err = user, None
            for attempt in range(1, max_retries + 2):
                res = self.generate(system, prompt, schema=schema, purpose=purpose, attempt=attempt)
                try:
                    out = model_cls.model_validate_json(_strip_fences(res.text))
                    span.set_attribute("structured.attempts", attempt)
                    return out
                except (ValidationError, json.JSONDecodeError) as e:
                    last_err = e
                    span.add_event("validation_failed", {"attempt": attempt, "error": str(e)[:400]})
                    prompt = (
                        f"{user}\n\nYour previous output was invalid:\n{res.text[:1500]}\n"
                        f"Error: {str(e)[:600]}\nReturn corrected JSON only."
                    )
            span.set_attribute("structured.attempts", max_retries + 1)
            raise LLMError(f"Structured output failed after retries: {last_err}")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()
