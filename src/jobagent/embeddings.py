import httpx
import numpy as np

from jobagent.config import get_settings
from jobagent.observability import tracer


def embed(texts: list[str], kind: str = "document") -> np.ndarray:
    """Embed via Ollama. nomic-embed-text expects a task prefix. Returns L2-normalized rows."""
    s = get_settings()
    prefix = "search_query: " if kind == "query" else "search_document: "
    with tracer().start_as_current_span("embed") as span:
        span.set_attribute("openinference.span.kind", "EMBEDDING")
        span.set_attribute("embedding.model_name", s.embed_model)
        span.set_attribute("embedding.count", len(texts))
        r = httpx.post(f"{s.ollama_host}/api/embed", timeout=s.llm_timeout_s,
                       json={"model": s.embed_model, "input": [prefix + t[:2000] for t in texts]})
        r.raise_for_status()
        m = np.array(r.json()["embeddings"], dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)
