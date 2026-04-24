FROM ghcr.io/meta-pytorch/openenv-base:latest

# HF Spaces enforces UID 1000. Base image runs as root; create appuser.
RUN useradd -u 1000 -m -s /bin/bash appuser

WORKDIR /app

# ── Dep-cache layer ───────────────────────────────────────────────────────────
# Full runtime dep list mirroring pyproject.toml. This layer is cached until
# version pins change. pip no-ops on packages already in the base image
# (fastapi, pydantic, httpx, sse-starlette, PyYAML); real work is openenv-core,
# aiohttp, and upgrading uvicorn from the base's 0.44.0 to >= 0.45.0.
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.3" \
    "fastapi>=0.136.0" \
    "uvicorn>=0.45.0" \
    "pydantic>=2.13.0" \
    "httpx>=0.28.1" \
    "sse-starlette>=3.0.0" \
    "aiohttp==3.13.5" \
    "pyyaml>=6.0.2"

# ── Source + wheel layer ──────────────────────────────────────────────────────
# Copies all source. pip --no-deps builds the setuptools wheel; package_data
# copies server/dashboard/index.html into site-packages so Path(__file__).parent
# in sse.py resolves to the installed location (not /app/).
COPY . .
RUN pip install --no-deps --no-cache-dir . && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# HF Spaces uses this to know when the container is ready.
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "sre_arena_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
