$ErrorActionPreference = "Stop"

$ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path
$INPUT_JSON = Join-Path $ROOT "logs\qa_phase11_run7_20260222_005336\ai_reading_sample_with_technical_appendix_v14.json"
$OUTDIR = Join-Path $ROOT "logs\qa_phase11_run7_20260222_005336\audit_package_v1"
$SCHEMA = Join-Path $ROOT "backend\audit_templates\vedic_technical_audit_output_schema_v1.json"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$RESULT_JSON = Join-Path $OUTDIR ("audit_result_{0}.json" -f $TS)

Write-Host "[1/3] Build audit package"
python (Join-Path $ROOT "scripts\build_vedic_audit_package.py") `
  --input $INPUT_JSON `
  --outdir $OUTDIR

Write-Host "[2/3] Open prompt file and paste it into your Vedic auditor LLM UI:"
Write-Host "      $OUTDIR\audit_prompt_filled.md"
Write-Host "      Then save model output JSON to:"
Write-Host "      $RESULT_JSON"

Write-Host "[3/3] Validate audit result (run after saving audit_result.json)"
python (Join-Path $ROOT "scripts\validate_vedic_audit_result.py") `
  --result $RESULT_JSON `
  --schema $SCHEMA
