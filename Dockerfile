# Demo image (also the Hugging Face Spaces Docker image): FastAPI + built React dashboard,
# synthetic demo data, mock Telegram notifier. CPU only, no torch.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --ignore-scripts
COPY web/ ./
RUN npx vite build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

# Spaces run containers as uid 1000; keep everything under /app.
RUN useradd -m -u 1000 app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DEMO_MODE=1 \
    PORT=7860

COPY --chown=app pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project && chown -R app /app

COPY --chown=app . .
COPY --from=web --chown=app /web/dist ./web/dist
USER app

EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.dashboard.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
