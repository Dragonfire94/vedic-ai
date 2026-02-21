# Quality Gates (Standard)

This project uses a two-speed gate strategy to keep developer cycle time fast while blocking regression leaks.

## PR Gate (required)

Run:

```bash
python -m backend.golden_sample_runner --mode structural
python -m backend.fast_llm_gate --samples 2 --profile-mode extremes
```

Fail conditions:
- `backend.fast_llm_gate` exits non-zero
- `forbidden_hits_total > 0` in `logs/golden_samples_fast_gate/fast_gate_summary.json`

Notes:
- `--profile-mode extremes` prioritizes `highest_stability` and `lowest_stability` for representative contrast.
- This gate is intentionally lighter than full golden run.
- PDF scanner is intentionally excluded from PR gate.

## Nightly / Release Gate (required)

Run PR Gate, then add:

```bash
python -m pytest backend/test_pdf_output_scanner.py -q
```

And before final release:

```bash
python -m backend.golden_sample_runner --mode full
```

Fail conditions:
- PDF scanner test fails
- Full golden run fails
- In Nightly/Release, PDF scanner is mandatory (no skip policy for release decision).

## Environment Notes

- Ensure a runner with `pypdf` installed exists for Nightly/Release PDF scanning.
- If PR runners do not install `pypdf`, keep PDF scan out of PR gate to avoid false skip confidence.
