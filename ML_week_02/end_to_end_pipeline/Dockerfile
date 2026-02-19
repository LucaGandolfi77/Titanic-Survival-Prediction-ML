# ============================================================
# Dockerfile — Titanic MLOps Prediction Service
# ============================================================
# Multi-stage build for a slim production image.
#
# Build:  docker build -t titanic-api .
# Run:    docker run -p 8000:8000 titanic-api
# ============================================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim

LABEL maintainer="Luca Gandolfi"
LABEL description="Titanic survival prediction — FastAPI serving"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config/ config/
COPY src/ src/
COPY models/ models/
COPY data/processed/ data/processed/
COPY serve.py .

# Create logs directory
RUN mkdir -p logs

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "serve.py"]
