# FastAPI research service (app/api/) for Railway deployment.
#
# Deliberately NOT a minimal image: RAG retrieval (chromadb +
# sentence-transformers) and FinBERT sentiment scoring are separate
# local models used regardless of LLM_PROVIDER -- torch/transformers/
# chromadb are real, load-bearing dependencies here, not leftovers.
# Only llama-cpp-python (app/rag/report_generator.py's local narrative
# backend) is conditional: LocalLlamaProvider lazy-imports it, so it's
# only ever touched when LLM_PROVIDER=local actually generates text --
# verified with LLM_PROVIDER=hosted set, across three real code paths
# (API import, a real ToolRegistry construction, and an actual
# generate() call): llama_cpp never appears in sys.modules, while
# torch/transformers/chromadb/sentence_transformers correctly do (RAG +
# FinBERT need them regardless of LLM_PROVIDER). This is why the image
# is still multi-GB despite that guard working correctly -- torch alone
# is ~500MB -- not evidence the guard failed. See EVALUATION.md and
# app/core/llm_provider.py for the full picture.
#
# Multi-stage: build-essential/cmake (needed only if llama-cpp-python
# or chromadb's native deps fall back to a source build) stay in the
# builder stage and never reach the final image.
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
# network from outside the container, not from in here -- kept only in
# case a future in-container healthcheck/debugging needs it. Cheap
# enough to keep; drop it if that never materializes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# $PORT is assigned by Railway at runtime, not fixed here -- the
# ${PORT:-8000} fallback only matters for `docker run` without -e PORT
# set (local testing convenience), Railway always sets it. Shell form
# (not exec-form CMD ["..."]) is required for ${PORT} to actually
# expand -- exec form passes the literal string through with no shell
# to interpolate it.
CMD uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
