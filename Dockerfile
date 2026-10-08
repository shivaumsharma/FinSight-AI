# Backend image (FastAPI, app/api). Deployed to AWS Elastic Beanstalk by
# .github/workflows/deploy-aws.yml; Railway and DigitalOcean also work. Reads
# $PORT at runtime (see the CMD line).
#
# Not minimal on purpose: RAG (chromadb + sentence-transformers) and FinBERT need
# torch/transformers whatever LLM_PROVIDER is, so the image is multi-GB. Only
# llama-cpp-python is conditional (LLM_PROVIDER=local).
#
# Multi-stage: build tools stay in the builder stage and never reach the final image.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

# curl: Railway's own healthcheck (railway.json) hits this over the
# network from outside the container, not from in here; Cloud Run's
# healthcheck likewise hits it over the network -- kept only in case a
# future in-container healthcheck/debugging needs it. Cheap enough to
# keep; drop it if that never materializes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

# Non-root: the app writes its persistent state (jobs.db, reports/,
# llm_logs/, logs/, filings_cache/, vector_db/, and mlruns/ if a
# training script ever runs here) directly under WORKDIR by default --
# see app/core/paths.py's own DATA_DIR docstring: DATA_DIR only points
# elsewhere (a mounted volume) when explicitly set, and defaults to the
# repo root otherwise. chown the whole tree rather than enumerating
# each write path individually, since a new one could be added later
# without this Dockerfile being updated to match.
RUN useradd --create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# $PORT is assigned at runtime by whichever platform this runs on
# (Cloud Run sets it to 8080, Railway sets its own value), not fixed
# here -- the ${PORT:-8000} fallback only matters for `docker run`
# without -e PORT set (local testing convenience). Shell form
# (not exec-form CMD ["..."]) is required for ${PORT} to actually
# expand -- exec form passes the literal string through with no shell
# to interpolate it.
CMD uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
