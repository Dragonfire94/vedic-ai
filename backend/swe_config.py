"""Shared Swiss Ephemeris initialization helpers."""

from __future__ import annotations

import logging

import swisseph as swe


def initialize_swe_context(logger: logging.Logger | None = None) -> bool:
    """Initialize Swiss Ephemeris context consistently across modules.

    Returns True if sidereal mode initialization was attempted successfully,
    False otherwise. This function is defensive so test stubs without full
    Swiss Ephemeris APIs do not crash module import.
    """
    log = logger or logging.getLogger("swe_config")

    try:
        if hasattr(swe, "set_ephe_path"):
            swe.set_ephe_path(None)
    except Exception as exc:
        log.warning("Failed to set ephemeris path: %s", exc)

    if not (hasattr(swe, "set_sid_mode") and hasattr(swe, "SIDM_LAHIRI")):
        log.warning("Swiss Ephemeris sidereal APIs not fully available; skipping sidereal initialization.")
        return False

    try:
        # Explicit Lahiri lock for deterministic sidereal calculations.
        swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)
        return True
    except TypeError:
        # Compatibility fallback for wrappers that only accept one positional arg.
        swe.set_sid_mode(swe.SIDM_LAHIRI)
        return True
    except Exception as exc:
        log.warning("Failed to set sidereal mode to Lahiri: %s", exc)
        return False

