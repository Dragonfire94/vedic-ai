from backend.astro_engine import ENGINE_SIGNATURE, ENGINE_VERSION

EXPECTED_VERSION = "1.0.1"
EXPECTED_SIGNATURE = "STRUCTURAL_CORE_V1"

# NOTE:
# Version and signature are intentionally hard-locked.
# Any promotion requires manual update of EXPECTED_* constants.
# This is a deliberate release friction policy.
def validate_engine_integrity() -> bool:
    errors: list[str] = []

    if ENGINE_VERSION != EXPECTED_VERSION:
        errors.append(f"Version mismatch: {ENGINE_VERSION} != {EXPECTED_VERSION}")

    if ENGINE_SIGNATURE != EXPECTED_SIGNATURE:
        errors.append("Engine structural signature mismatch.")

    if errors:
        raise RuntimeError("ENGINE INTEGRITY FAILURE:\n" + "\n".join(errors))

    return True
