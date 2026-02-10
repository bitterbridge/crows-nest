# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Builder - Install dependencies with uv
# =============================================================================
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up uv environment
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Install the project
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Development - Full tooling for testing
# =============================================================================
FROM python:3.12-slim AS dev

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install all dependencies including dev
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy source and tests
COPY src/ ./src/
COPY tests/ ./tests/
COPY .pre-commit-config.yaml ./

# Install the project with dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Default command for dev: run tests
CMD ["uv", "run", "pytest"]

# =============================================================================
# Stage 3: Runtime - Minimal production image
# =============================================================================
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd --gid 1000 crows && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home crows

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src ./src

# Set up path to use venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER crows

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import crows_nest; print('ok')" || exit 1

# Default command: start the API server
CMD ["uvicorn", "crows_nest.api:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000
