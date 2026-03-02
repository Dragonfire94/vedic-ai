#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_JSON="$ROOT/logs/qa_phase11_run7_20260222_005336/ai_reading_sample_with_technical_appendix_v14.json"
OUTDIR="$ROOT/logs/qa_phase11_run7_20260222_005336/audit_package_v1"
SCHEMA="$ROOT/backend/audit_templates/vedic_technical_audit_output_schema_v1.json"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_JSON="$OUTDIR/audit_result_${TS}.json"

echo "[1/3] Build audit package"
python "$ROOT/scripts/build_vedic_audit_package.py" \
  --input "$INPUT_JSON" \
  --outdir "$OUTDIR"

echo "[2/3] Open prompt file and paste it into your Vedic auditor LLM UI:"
echo "      $OUTDIR/audit_prompt_filled.md"
echo "      Then save model output JSON to:"
echo "      $RESULT_JSON"

echo "[3/3] Validate audit result (run after saving audit_result.json)"
python "$ROOT/scripts/validate_vedic_audit_result.py" \
  --result "$RESULT_JSON" \
  --schema "$SCHEMA"
