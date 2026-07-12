# Slim serving image for the Render free tier (512 MB RAM budget).
# Build from the REPO ROOT so src/ and the bundled model are in context:
#   docker build -f serving/Dockerfile -t churn-serving .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Only what serving needs: the src package (config + champion resolver) and the
# app itself, which carries the bundled model — the primary deployment path.
COPY src/ src/
COPY serving/ serving/

RUN useradd --create-home appuser \
    && mkdir -p /app/shadow \
    && chown appuser:appuser /app/shadow
USER appuser

ENV SHADOW_LOG_PATH=/app/shadow/comparisons.jsonl \
    PORT=8000
EXPOSE 8000

# Shell form so Render's injected $PORT expands.
CMD uvicorn serving.app:app --host 0.0.0.0 --port "$PORT"
