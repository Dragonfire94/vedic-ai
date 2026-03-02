#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "backend" / "audit_templates" / "vedic_technical_audit_prompt_v1.md"
DEFAULT_SCHEMA = ROOT / "backend" / "audit_templates" / "vedic_technical_audit_output_schema_v1.json"
TEMPLATE_VERSION = "v1"
SCHEMA_VERSION = "v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_newlines(text: str) -> str:
    # Hash normalization is newline-only; preserve all other whitespace.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256_normalized_text(text: str) -> str:
    return _sha256_text(_normalize_newlines(text))


def _canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_canonical_json(payload: Any) -> str:
    return _sha256_text(_canonical_json_dumps(payload))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed_to_load_json: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object: {path}")
    return payload


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _four_backtick_block(content: str, lang: str) -> str:
    safe = content if isinstance(content, str) and content.strip() else "—"
    return f"````{lang}\n{safe}\n````"


def _build_payload(
    source: dict[str, Any],
    *,
    input_path: Path,
    truncate_technical_md: int | None,
    prefer_data_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vedic_data = source.get("vedic_technical_data")
    if not isinstance(vedic_data, dict):
        raise ValueError("missing_required_field: vedic_technical_data")

    polished = source.get("polished_reading")
    reading = source.get("reading")
    if isinstance(polished, str) and polished.strip():
        commercial_text = polished
        commercial_source = "polished_reading"
    elif isinstance(reading, str) and reading.strip():
        commercial_text = reading
        commercial_source = "reading"
    else:
        commercial_text = ""
        commercial_source = "none"

    technical_reading_raw = source.get("vedic_technical_reading") if isinstance(source.get("vedic_technical_reading"), str) else ""
    technical_reading = technical_reading_raw
    technical_reading_included = True
    technical_reading_truncated = False
    technical_reading_original_chars = len(technical_reading_raw)
    if prefer_data_only:
        technical_reading = ""
        technical_reading_included = False
    elif isinstance(truncate_technical_md, int) and truncate_technical_md > 0 and len(technical_reading) > truncate_technical_md:
        technical_reading = technical_reading[:truncate_technical_md].rstrip() + "\n...[TRUNCATED]"
        technical_reading_truncated = True

    raw_text = _canonical_json_dumps(source)
    payload = {
        "meta": {
            "created_utc": _utc_now_iso(),
            "template_version": TEMPLATE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "commercial_source": commercial_source,
        },
        "source_paths": {
            "input_json": str(input_path),
        },
        "source_hashes": {
            "input_json_sha256": _sha256_text(raw_text),
        },
        "commercial_source": commercial_source,
        "vedic_technical_data": vedic_data,
        "vedic_technical_reading": technical_reading,
        "commercial_reading": commercial_text,
    }

    stats = {
        "technical_reading_included": technical_reading_included,
        "technical_reading_truncated": technical_reading_truncated,
        "technical_reading_original_chars": technical_reading_original_chars,
        "technical_reading_final_chars": len(technical_reading),
        "commercial_chars": len(commercial_text),
        "technical_data_top_level_keys_count": len(vedic_data.keys()),
    }
    return payload, stats


def _fill_template(
    template_text: str,
    *,
    commercial_source: str,
    vedic_technical_data: dict[str, Any],
    vedic_technical_reading: str,
    commercial_reading: str,
) -> str:
    mapping = {
        "{{COMMERCIAL_SOURCE}}": commercial_source,
        "{{VEDIC_TECHNICAL_DATA_BLOCK}}": _four_backtick_block(
            json.dumps(vedic_technical_data, ensure_ascii=False, indent=2),
            "json",
        ),
        "{{VEDIC_TECHNICAL_READING_BLOCK}}": _four_backtick_block(vedic_technical_reading, "md"),
        "{{COMMERCIAL_READING_BLOCK}}": _four_backtick_block(commercial_reading, "md"),
    }
    out = template_text
    for key, value in mapping.items():
        out = out.replace(key, value)
    missing_tokens = [token for token in mapping.keys() if token in out]
    if missing_tokens:
        raise ValueError(f"unreplaced_template_tokens: {missing_tokens}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Vedic technical audit package from ai_reading response JSON.")
    parser.add_argument("--input", required=True, help="Path to ai_reading response JSON.")
    parser.add_argument("--outdir", required=True, help="Output directory for audit package files.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Prompt template path.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Output schema path.")
    parser.add_argument(
        "--truncate-technical-md",
        type=int,
        default=None,
        help="Optional character cap for vedic_technical_reading before insertion.",
    )
    parser.add_argument(
        "--prefer-data-only",
        action="store_true",
        help="Exclude vedic_technical_reading block and keep authoritative data-only mode.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    template_path = Path(args.template).resolve()
    schema_path = Path(args.schema).resolve()

    if not input_path.exists():
        print(f"ERROR: input_not_found: {input_path}")
        return 1
    if not template_path.exists():
        print(f"ERROR: template_not_found: {template_path}")
        return 1
    if not schema_path.exists():
        print(f"ERROR: schema_not_found: {schema_path}")
        return 1

    try:
        source = _load_json(input_path)
        payload, stats = _build_payload(
            source,
            input_path=input_path,
            truncate_technical_md=args.truncate_technical_md,
            prefer_data_only=bool(args.prefer_data_only),
        )
        template_text = template_path.read_text(encoding="utf-8")
        filled_prompt = _fill_template(
            template_text,
            commercial_source=str(payload.get("commercial_source") or "none"),
            vedic_technical_data=payload["vedic_technical_data"],
            vedic_technical_reading=str(payload.get("vedic_technical_reading") or ""),
            commercial_reading=str(payload.get("commercial_reading") or ""),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    outdir.mkdir(parents=True, exist_ok=True)
    payload_path = outdir / "audit_input_payload.json"
    prompt_path = outdir / "audit_prompt_filled.md"
    manifest_path = outdir / "audit_manifest.json"

    _json_dump(payload_path, payload)
    prompt_path.write_text(filled_prompt, encoding="utf-8")

    payload_text = payload_path.read_text(encoding="utf-8")
    payload_sha = _sha256_normalized_text(payload_text)
    commercial_text = str(payload.get("commercial_reading") or "")
    technical_md_text = str(payload.get("vedic_technical_reading") or "")
    technical_data = payload.get("vedic_technical_data") if isinstance(payload.get("vedic_technical_data"), dict) else {}
    commercial_sha = _sha256_normalized_text(commercial_text)
    technical_md_sha = _sha256_normalized_text(technical_md_text)
    technical_data_sha = _sha256_canonical_json(technical_data)
    manifest = {
        "created_utc": _utc_now_iso(),
        "input_path": str(input_path),
        "template_file": str(template_path),
        "template_version": TEMPLATE_VERSION,
        "schema_file": str(schema_path),
        "schema_version": SCHEMA_VERSION,
        "payload_file": str(payload_path),
        "prompt_file": str(prompt_path),
        "payload_sha256": payload_sha,
        "commercial_source": payload.get("commercial_source"),
        "commercial_sha256": commercial_sha,
        "technical_data_sha256": technical_data_sha,
        "technical_md_sha256": technical_md_sha,
        "options": {
            "truncate_technical_md": args.truncate_technical_md,
            "prefer_data_only": bool(args.prefer_data_only),
        },
        "stats": stats,
        "source_paths": payload.get("source_paths"),
        "source_hashes": payload.get("source_hashes"),
    }
    _json_dump(manifest_path, manifest)

    print(f"OK: package_created: {outdir}")
    print(f"- payload: {payload_path}")
    print(f"- prompt:  {prompt_path}")
    print(f"- manifest:{manifest_path}")
    print(
        "- telemetry: payload_chars=%s commercial_chars=%s technical_md_chars=%s technical_data_top_level_keys_count=%s"
        % (
            len(payload_text),
            len(commercial_text),
            len(technical_md_text),
            len(technical_data.keys()),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
