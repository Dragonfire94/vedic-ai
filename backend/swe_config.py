"""Shared Swiss Ephemeris initialization helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

import swisseph as swe


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _probe_ephemeris_backend(log: logging.Logger) -> dict[str, Any]:
    """Probe whether Swiss Ephemeris binary files are used or Moshier fallback is active."""
    status: dict[str, Any] = {
        "ephemeris_backend": "unknown",
        "ephemeris_verified": False,
        "ephemeris_retflag": None,
    }
    try:
        if not (hasattr(swe, "calc_ut") and hasattr(swe, "MOON")):
            status["probe_error"] = "calc_ut or MOON not available"
            return status

        jd = swe.julday(2024, 1, 1, 0.0) if hasattr(swe, "julday") else 2460310.5
        flags = 0
        if hasattr(swe, "FLG_SWIEPH"):
            flags |= swe.FLG_SWIEPH
        if hasattr(swe, "FLG_SIDEREAL"):
            flags |= swe.FLG_SIDEREAL

        _, retflag = swe.calc_ut(jd, swe.MOON, flags)
        status["ephemeris_retflag"] = int(retflag)

        uses_moshier = bool(getattr(swe, "FLG_MOSEPH", 0) and (retflag & swe.FLG_MOSEPH))
        uses_swieph = bool(getattr(swe, "FLG_SWIEPH", 0) and (retflag & swe.FLG_SWIEPH))

        if uses_swieph and not uses_moshier:
            status["ephemeris_backend"] = "swieph"
            status["ephemeris_verified"] = True
        elif uses_moshier:
            status["ephemeris_backend"] = "moshier"
            status["ephemeris_verified"] = False
        else:
            status["ephemeris_backend"] = "unknown"
            status["ephemeris_verified"] = False
        return status
    except Exception as exc:
        log.warning("Failed to probe ephemeris backend: %s", exc)
        status["probe_error"] = str(exc)
        return status


def initialize_swe_context(logger: logging.Logger | None = None) -> dict[str, Any]:
    """Initialize Swiss Ephemeris context consistently across modules.

    Returns a status dictionary so callers can expose diagnostics in health checks.
    The function is defensive so test stubs without full Swiss Ephemeris APIs
    do not crash module import.
    """
    log = logger or logging.getLogger("swe_config")
    ephe_path = os.getenv("SWE_EPHE_PATH", "/usr/share/libswe/ephe")
    require_swieph = _is_truthy(os.getenv("SWE_REQUIRE_SWIEPH", "0"))
    status: dict[str, Any] = {
        "ephemeris_path": ephe_path,
        "sidereal_mode": "unknown",
        "sidereal_initialized": False,
        "ephemeris_backend": "unknown",
        "ephemeris_verified": False,
        "ephemeris_retflag": None,
        "require_swieph": require_swieph,
    }

    try:
        if hasattr(swe, "set_ephe_path"):
            swe.set_ephe_path(ephe_path)
    except Exception as exc:
        log.warning("Failed to set ephemeris path: %s", exc)
        status["ephe_path_error"] = str(exc)

    if not (hasattr(swe, "set_sid_mode") and hasattr(swe, "SIDM_LAHIRI")):
        log.warning("Swiss Ephemeris sidereal APIs not fully available; skipping sidereal initialization.")
        status.update(_probe_ephemeris_backend(log))
        return status

    try:
        # Explicit Lahiri lock for deterministic sidereal calculations.
        swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)
        status["sidereal_initialized"] = True
        status["sidereal_mode"] = "lahiri"
    except TypeError:
        # Compatibility fallback for wrappers that only accept one positional arg.
        swe.set_sid_mode(swe.SIDM_LAHIRI)
        status["sidereal_initialized"] = True
        status["sidereal_mode"] = "lahiri"
    except Exception as exc:
        log.warning("Failed to set sidereal mode to Lahiri: %s", exc)
        status["sidereal_error"] = str(exc)

    status.update(_probe_ephemeris_backend(log))

    if require_swieph and not status.get("ephemeris_verified", False):
        raise RuntimeError(
            "Swiss Ephemeris data files are not available. "
            f"Configured path: {ephe_path}. Backend: {status.get('ephemeris_backend')}."
        )

    return status
