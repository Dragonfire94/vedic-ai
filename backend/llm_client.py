"""LLM client initialization utilities."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger("vedic_ai")


def _first_nonempty_env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_openai_base_url() -> Optional[str]:
    configured = _first_nonempty_env("OPENAI_BASE_URL", "OPENAI_API_BASE")
    if not configured:
        return None
    lowered = configured.lower()
    if "localhost" in lowered or "127.0.0.1" in lowered:
        logger.error("Invalid OPENAI base URL '%s' detected; falling back to default OpenAI endpoint", configured)
        return None
    return configured


def build_openai_client(api_key: str) -> tuple[Optional[AsyncOpenAI], Optional[httpx.AsyncClient]]:
    if not api_key:
        return None, None

    base_url = _resolve_openai_base_url()
    proxy_url = _first_nonempty_env("OPENAI_PROXY_URL", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=120.0)

    try:
        if proxy_url:
            http_client = httpx.AsyncClient(
                timeout=timeout,
                trust_env=True,
                proxy=proxy_url,
            )
        else:
            http_client = httpx.AsyncClient(
                timeout=timeout,
                trust_env=True,
            )
        logger.info(
            "OpenAI transport configured: proxy=%s trust_env=%s",
            proxy_url if proxy_url else "NONE",
            True,
        )
        client_kwargs: dict[str, Any] = {"api_key": api_key, "http_client": http_client}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)
        logger.info(
            "OpenAI client initialized base_url=%s proxy_configured=%s",
            str(getattr(client, "base_url", "default")),
            "True" if bool(proxy_url) else "False",
        )
        return client, http_client
    except Exception as e:
        logger.warning("OpenAI client initialization failed: %s", e)
        return None, None
