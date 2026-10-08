FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# dependencies first: this layer is cached until pyproject/uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY evals ./evals
COPY samples ./samples
RUN uv sync --frozen --no-dev   # installs the project itself (editable, so evals/ resolves relative to /app)

ENV PATH="/app/.venv/bin:$PATH"
RUN useradd -m app && mkdir -p data && chown -R app /app/data /app/evals
USER app

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/jobagent/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
