# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for biopython / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY spatialppiv2/ spatialppiv2/

# Install CPU-only torch first (smaller image), then the package
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir torch-geometric \
 && pip install --no-cache-dir -e ".[api]"


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages and source
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

# Config and checkpoint dirs (mount via -v in production)
RUN mkdir -p config checkpoint data/processed/pdbs results

COPY config/default.yaml config/default.yaml

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONUNBUFFERED=1

CMD ["sppi-api", "--host", "0.0.0.0", "--port", "8000"]
