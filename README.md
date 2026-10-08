# jobagent: resume → jobs, an agentic AI capstone

Upload a resume and get ranked job matches with evidence for each one. The agent extracts a profile,
plans its own searches, queries several job sources through a **custom MCP server**, ranks results with
embeddings plus an **LLM judge whose claims are checked against the source text**, and reports skill
gaps. It runs fully local (Ollama) or with Gemini, and every step is traced end to end.

<!-- TODO before publishing: add docs/img/ui.png (results page), docs/img/timeline.png (live
     timeline), docs/img/trace.png (Phoenix trace), and a 2-3 minute demo video link here. -->

## What this project demonstrates
| Skill | Where to look |
|---|---|
| **Agent orchestration**: stateful graph with a conditional loop, live progress events | `pipeline/graph.py` |
| **MCP**: a server I wrote (6 sources, normalized schema) and a client that propagates trace context | `mcp_servers/`, `mcp_client.py` |
| **Provider-agnostic LLM layer**: Ollama and Gemini, fallback, schema-constrained output with retry, per-run budget | `llm/` |
| **Grounding / hallucination control**: a critic that removes skill claims not present in the profile or posting | `pipeline/ranking.py` (`critic`) |
| **Security**: prompt-injection screening of untrusted postings, PII redaction before remote calls, input validation | `guardrails.py`, `resume/redact.py` |
| **Observability**: OpenTelemetry → Phoenix; one trace across two processes; per-call tokens, context window, full prompts | `observability/`, `docs/DESIGN_DECISIONS.md` |
| **Evaluation**: golden resumes, labeled job pool incl. adversarial cases, nDCG, calibrated LLM-judge | `evals/` |
| **Engineering**: tests, CI, Docker (app + Phoenix, Ollama on host) | `tests/`, `Dockerfile`, `.github/` |

**Design decisions, trade-offs and what went wrong along the way:** [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)

## Status and honest limits
- The eval set is small (5 golden resumes, 45 labeled jobs), so metrics are directional, not proof.
- On the hard pool, the hand-weighted ranking blend does **not** beat plain embeddings (see results below).
- The Gemini path is implemented and unit-tested, but I have not yet run the local-vs-Gemini eval comparison.
- It is a deterministic workflow with LLM steps rather than an open-ended agent; see the design doc for why.

## Architecture
```
resume ─ validate ─ parse ─ (PII redaction if remote LLM) ─ profile extractor ─┐
                                                                               ▼
        LangGraph:  plan queries → search (MCP client → jobs-mcp) → dedupe ─┐ too few? broaden once
                    → prelim rank (embedding + skills + title; injection scan) → LLM judge top-10
                    → critic (grounding) → report                          ◄──┘
```

| Part | Where |
|---|---|
| LLM layer (Ollama, Gemini, auto-fallback, structured output + retry, budget) | `src/jobagent/llm/` |
| MCP server (6 sources: Adzuna, Remotive, RemoteOK, Arbeitnow, Greenhouse, Lever) | `src/jobagent/mcp_servers/` |
| Pipeline + ranking | `src/jobagent/pipeline/` |
| Guardrails (input validation, prompt injection, URL safety, PII, budget) | `src/jobagent/guardrails.py` |
| Observability (OpenTelemetry → Arize Phoenix) | `src/jobagent/observability/` |
| Evals (golden resumes, labeled job pool, nDCG, groundedness, safety) | `src/jobagent/evals/`, `evals/data/` |
| UI | `src/jobagent/ui/app.py` |

## Run
```bash
uv sync --python 3.12
cp .env.example .env            # add ADZUNA_APP_ID/KEY, optionally GEMINI_API_KEY
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
docker compose up -d phoenix    # traces at http://localhost:6006 (give it ~15s to boot)
uv run jobagent ui              # Streamlit
uv run jobagent run samples/sample_resume.txt
uv run jobagent eval --provider local     # later: --provider gemini, compare reports in evals/results/
uv run pytest
```

## Run with Docker
Ollama stays on the host; the compose stack runs the app and Phoenix and reaches Ollama through
`host.docker.internal`.
```bash
ollama pull qwen2.5:7b && ollama pull nomic-embed-text     # on the host
cp .env.example .env                                       # ADZUNA_APP_ID/KEY, optional GEMINI_API_KEY
docker compose up -d --build      # UI http://localhost:8501, traces http://localhost:6006
docker compose run --rm app jobagent run samples/sample_resume.txt
docker compose run --rm app jobagent eval --provider local
```
Requirements: host Ollama must accept connections from containers (`OLLAMA_HOST=0.0.0.0`, e.g. via a
systemd override on Linux; default is 127.0.0.1 only). Set `APP_PORT` if 8501 is taken and
`DOCKER_OLLAMA_HOST` to point at a different Ollama. Run records and eval reports persist in `./data`
and `./evals/results`. If your host uid isn't 1000, export `HOST_UID`/`HOST_GID`.

## Traceability
- **One run = one trace**, across processes: the agent sends W3C trace context to the `jobs-mcp`
  subprocess in MCP `_meta`, so tool calls and each source's HTTP call (`source.adzuna`, ...) are
  children of the agent's spans. Phoenix projects: `jobagent` (app runs), `jobagent-evals` (evals).
- **Run record:** every run gets a `run_id` and is saved to `data/runs/<run_id>.json` with its
  `trace_id`, deep link, provider/model, embedding model, ranking weights, resume hash (not content),
  token usage. `uv run jobagent runs` lists them; the UI can reload past runs.
- **Decision trail:** each ranked job stores its feature breakdown and an ordered score trail
  (prelim components → LLM score → critic adjustments → cap → final). `jobagent run --explain`
  prints it; the UI shows it under "Why this rank?". Per-job `rerank.job` and `critic` spans carry
  the same numbers.
- **Live timeline:** the UI (and CLI) stream pipeline events while a run is in progress: current step,
  the LLM request in flight, each job as it is judged, running token/context totals (`ui/live.py`).
- Logs carry `trace_id`; failures are recorded on spans (including per-source errors).

## Ranking
`prelim = 0.5·embedding + 0.3·skill coverage + 0.2·title match` over the whole pool →
the LLM scores the top 10 (pointwise, 0-10, matched/missing skills) → the critic removes skills not
literally in the profile/job text and caps unevidenced high scores →
`final = 0.4·prelim + 0.6·llm/10`. Postings containing injection phrases are flagged, stripped before
the LLM sees them, and demoted below every clean job.

## Eval results (qwen2.5:7b; judge: qwen2.5:14b-instruct; 5 golden resumes)
Pool = 30 core jobs + 15 hard near-misses (wrong seniority, right keywords but wrong role, synonym-only
matches, a subtle injection, a benign posting that discusses injection). `--pool core|hard|all`.

| metric | core (30 jobs) | all (45 jobs) |
|---|---|---|
| nDCG@5: embedding only | 0.919 | 0.854 |
| nDCG@5: blended prelim | 0.946 | 0.790 |
| nDCG@5: + LLM judge & critic | 0.962 | 0.809 |
| nDCG@10 (same three) | 0.963 / 0.962 / 0.968 | 0.874 / 0.866 / 0.875 |

| other metric | value |
|---|---|
| skill recall / invented skills | 0.96 / 0.00 |
| adversarial postings in top-5 / flagged | 0 / 100% |
| benign posting wrongly flagged as injection | 5 of 5 runs (known tradeoff of regex screening) |
| LLM explanations judged faithful | 73% |
| judge catches a fabricated requirement | 100% (calibration check) |
| LLM skill claims surviving critic | 73% |

Findings: on the hard pool the hand-weighted blend does **not** beat embedding-only (keyword and title
features reward "Java Trainer" and "DevOps Sales Engineer"), and the LLM stage recovers only part of it.
A weight sweep suggests ~0.7/0.2/0.1 helps a little, but with 5 personas that is within noise, so the
weights are left unchanged until the golden set is larger.
