FROM python:3.11-slim

WORKDIR /app

# Install the package first so this layer is cached across source/artifact changes.
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -e '.[api,recs-pip]'

COPY configs/ configs/
# Staged by docker/prepare_serve_artifacts.sh — only the ~25MB the served API reads,
# not the full (multi-GB) artifacts/ tree.
COPY docker/artifacts/ artifacts/

# Universal Sentence Encoder download cache (~1GB on first request); mount a volume here
# in compose so repeated `docker compose up` doesn't re-download it every time.
ENV TFHUB_CACHE_DIR=/app/.tfhub_cache

EXPOSE 8000

CMD ["uvicorn", "steam_review_ml.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
