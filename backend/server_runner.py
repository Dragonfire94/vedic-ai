"""Uvicorn launcher with environment-driven concurrency controls."""

import os

import uvicorn


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = _env_int("PORT", 8000, minimum=1)
    workers = _env_int("WEB_CONCURRENCY", 1, minimum=1)
    backlog = _env_int("UVICORN_BACKLOG", 2048, minimum=16)
    timeout_keep_alive = _env_int("UVICORN_TIMEOUT_KEEP_ALIVE", 5, minimum=1)
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info").strip() or "info"
    limit_concurrency = _env_optional_int("UVICORN_LIMIT_CONCURRENCY")

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        workers=workers,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
        limit_concurrency=limit_concurrency,
        log_level=log_level,
    )
