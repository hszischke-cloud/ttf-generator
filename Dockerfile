# ── Build stage ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed to compile some Python packages (OpenCV, Pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libgl1 potrace \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 potrace \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/ ./backend/
COPY test_ui.html ./test_ui.html

# Render sets PORT env var; default to 8000
ENV PORT=8000
ENV JOBS_DIR=/tmp/jobs

EXPOSE $PORT

# Run from inside the backend directory so relative imports work
WORKDIR /app/backend

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
