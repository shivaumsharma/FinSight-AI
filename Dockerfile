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

# Runs as root, as the images deployed before 2026-10-08 did. A non-root user crashed production on startup
# ("attempt to write a readonly database"): Elastic Beanstalk creates the /var/app/data host volume (mounted at
# DATA_DIR=/data) owned by root, so appuser cannot write jobs.db there. To drop privileges again, chown the volume
# first (entrypoint or EB platform hook) and test that on a built image with a root-owned volume before re-enabling.

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
