FROM python:3.12-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
RUN python -m venv "${VIRTUAL_ENV}" \
    && pip install --no-cache-dir --require-hashes -r requirements.txt

COPY app ./app
COPY evals ./evals
COPY packs ./packs
RUN pip install --no-cache-dir --no-deps --no-build-isolation .

FROM python:3.12-slim AS runtime

ARG GIT_SHA=dev
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_SHA="${GIT_SHA}" \
    PORT=10000

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY packs ./packs
COPY evals/baseline.json ./evals/baseline.json
COPY reports/sample_latest.json ./reports/sample_latest.json
RUN mkdir -p .cache/embeddings reports \
    && chown -R app:app /app

USER app

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '10000') + '/health', timeout=2)"]

EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
