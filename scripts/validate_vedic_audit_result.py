#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "backend" / "audit_templates" / "vedic_technical_audit_output_schema_v1.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _type_name(value: Any) -> str:
    return type(value).__name__


def _require_key(obj: dict[str, Any], key: str, errors: list[str], path: str) -> None:
    if key not in obj:
        errors.append(f"{path}.{key}: missing")


def _manual_validate(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["$: must be object"]

    required_top = [
        "verdict",
        "availability",
        "technical_findings",
        "commercial_contradictions",
        "commercial_minimal_edits",
        "notes_for_engineers",
    ]
    for key in required_top:
        _require_key(result, key, errors, "$")

    verdict = result.get("verdict")
    if not isinstance(verdict, dict):
        errors.append(f"$.verdict: must be object, got {_type_name(verdict)}")
    else:
        for key in ("technical_ok", "commercial_consistent_with_technical", "confidence", "blocking_issues_count"):
            _require_key(verdict, key, errors, "$.verdict")
        if "technical_ok" in verdict and not isinstance(verdict.get("technical_ok"), bool):
            errors.append("$.verdict.technical_ok: must be boolean")
        if "commercial_consistent_with_technical" in verdict and not isinstance(verdict.get("commercial_consistent_with_technical"), bool):
            errors.append("$.verdict.commercial_consistent_with_technical: must be boolean")
        confidence = verdict.get("confidence")
        if not isinstance(confidence, (int, float)):
            errors.append("$.verdict.confidence: must be number")
        elif not (0.0 <= float(confidence) <= 1.0):
            errors.append("$.verdict.confidence: must be within [0,1]")
        blocking = verdict.get("blocking_issues_count")
        if not isinstance(blocking, int):
            errors.append("$.verdict.blocking_issues_count: must be integer")
        elif blocking < 0:
            errors.append("$.verdict.blocking_issues_count: must be >= 0")

    availability = result.get("availability")
    allowed_reasons = {"missing_chart_context", "missing_varga", "missing_dasha", "missing_transits", "partial_data", "unknown", None}
    if not isinstance(availability, dict):
        errors.append(f"$.availability: must be object, got {_type_name(availability)}")
    else:
        for key in ("ok", "reason", "missing_fields", "audit_limitations"):
            _require_key(availability, key, errors, "$.availability")
        if "ok" in availability and not isinstance(availability.get("ok"), bool):
            errors.append("$.availability.ok: must be boolean")
        if availability.get("reason") not in allowed_reasons:
            errors.append("$.availability.reason: invalid enum value")
        if "missing_fields" in availability and not (
            isinstance(availability.get("missing_fields"), list) and all(isinstance(x, str) for x in availability.get("missing_fields", []))
        ):
            errors.append("$.availability.missing_fields: must be list[str]")
        if "audit_limitations" in availability and not (
            isinstance(availability.get("audit_limitations"), list) and all(isinstance(x, str) for x in availability.get("audit_limitations", []))
        ):
            errors.append("$.availability.audit_limitations: must be list[str]")

    findings = result.get("technical_findings")
    if not isinstance(findings, dict):
        errors.append(f"$.technical_findings: must be object, got {_type_name(findings)}")
    else:
        for key in ("critical", "minor"):
            _require_key(findings, key, errors, "$.technical_findings")
        for bucket in ("critical", "minor"):
            items = findings.get(bucket, [])
            if not isinstance(items, list):
                errors.append(f"$.technical_findings.{bucket}: must be list")
                continue
            for idx, item in enumerate(items):
                path = f"$.technical_findings.{bucket}[{idx}]"
                if not isinstance(item, dict):
                    errors.append(f"{path}: must be object")
                    continue
                evidence = item.get("evidence_paths")
                if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
                    errors.append(f"{path}.evidence_paths: must be list[str]")
                    continue
                if len(evidence) == 0:
                    errors.append(f"{path}.evidence_paths: must not be empty")
                    continue
                if not any(x.startswith("vedic_technical_data.") for x in evidence):
                    errors.append(f"{path}.evidence_paths: at least one path must start with 'vedic_technical_data.'")

    for key in ("commercial_contradictions", "commercial_minimal_edits", "notes_for_engineers"):
        value = result.get(key)
        if key == "notes_for_engineers":
            if not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                errors.append("$.notes_for_engineers: must be list[str]")
        else:
            if not isinstance(value, list):
                errors.append(f"$.{key}: must be list")

    return errors


def _schema_validate_if_available(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator  # type: ignore
    except Exception:
        return []
    validator = Draft202012Validator(schema)
    errs = []
    for err in sorted(validator.iter_errors(result), key=lambda e: e.path):
        path = "$." + ".".join(str(p) for p in err.path) if err.path else "$"
        errs.append(f"{path}: {err.message}")
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Vedic audit result JSON.")
    parser.add_argument("--result", required=True, help="Path to audit_result.json")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Path to output schema JSON")
    args = parser.parse_args()

    result_path = Path(args.result).resolve()
    schema_path = Path(args.schema).resolve()
    if not result_path.exists():
        print(f"ERROR: result_not_found: {result_path}")
        return 1
    if not schema_path.exists():
        print(f"ERROR: schema_not_found: {schema_path}")
        return 1

    try:
        result = _load_json(result_path)
        schema = _load_json(schema_path)
    except Exception as exc:
        print(f"ERROR: json_load_failed: {exc}")
        return 1

    errors: list[str] = []
    if not isinstance(result, dict):
        errors.append("$: result root must be object")
    if not isinstance(schema, dict):
        errors.append("$: schema root must be object")

    if not errors:
        errors.extend(_schema_validate_if_available(result, schema))
        errors.extend(_manual_validate(result))

    if errors:
        print("INVALID")
        for e in errors:
            print(f"- {e}")
        return 1

    verdict = result.get("verdict", {})
    print("VALID")
    print(
        "summary technical_ok=%s commercial_consistent=%s blocking_issues=%s confidence=%s"
        % (
            verdict.get("technical_ok"),
            verdict.get("commercial_consistent_with_technical"),
            verdict.get("blocking_issues_count"),
            verdict.get("confidence"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
