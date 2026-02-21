import json
import traceback
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app, get_chart
from backend.astro_engine import build_structural_summary
from backend.golden_sample_runner import generate_golden_charts, _candidate_metrics, select_profiles
from backend.llm_service import audit_llm_output
from backend.llm_output_scanner import scan_forbidden_patterns


def main():
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f'logs/qa_phase11_run7_{run_ts}')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'RUN_DIR={run_dir}')

    inputs = generate_golden_charts()
    candidates = []
    for payload in inputs:
        chart = get_chart(
            year=payload['year'], month=payload['month'], day=payload['day'], hour=payload['hour'],
            lat=payload['lat'], lon=payload['lon'], house_system=payload['house_system'],
            include_nodes=payload['include_nodes'], include_d9=payload['include_d9'],
            include_vargas=payload['include_vargas'], gender=payload['gender'], timezone=payload['timezone']
        )
        structural_summary = build_structural_summary(chart, analysis_mode=payload['analysis_mode'])
        candidates.append(_candidate_metrics(payload, chart, structural_summary))

    selected = dict(select_profiles(candidates))
    ordered = [
        ('highest_stability', selected['highest_stability']),
        ('lowest_stability', selected['lowest_stability']),
        ('most_balanced', selected['most_balanced']),
    ]

    client = TestClient(app)
    rows = []

    for idx, (name, cand) in enumerate(ordered, start=1):
        p = cand['input']
        structural_summary = cand['structural_summary']
        print(f'processing {idx} {name} seed={cand.get("seed")}')

        params = {
            'year': p['year'], 'month': p['month'], 'day': p['day'], 'hour': p['hour'],
            'lat': p['lat'], 'lon': p['lon'], 'house_system': p['house_system'],
            'include_nodes': p['include_nodes'], 'include_d9': p['include_d9'], 'include_vargas': p['include_vargas'],
            'language': 'ko', 'gender': p['gender'],
            'analysis_mode': p['analysis_mode'], 'detail_level': 'full', 'use_cache': 1
        }
        if p.get('timezone') is not None:
            params['timezone'] = p.get('timezone')

        ai_resp = client.get('/ai_reading', params=params, timeout=300)
        ai_resp.raise_for_status()
        ai = ai_resp.json()
        ai_text = (ai.get('polished_reading') or ai.get('reading') or ai.get('report_text') or '').strip()

        audit = audit_llm_output(ai_text, structural_summary)
        ai_forbidden_hits = len(scan_forbidden_patterns(ai_text))

        pdf_resp = client.get('/pdf', params={**params, 'include_ai': 1}, timeout=300)
        pdf_resp.raise_for_status()

        subdir = run_dir / f"{idx:02d}_{name}"
        subdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        pdf_path = subdir / f"{name}_{stamp}.pdf"
        ai_path = subdir / f"{name}_{stamp}.ai_reading.json"
        aud_path = subdir / f"{name}_{stamp}.audit.json"

        pdf_path.write_bytes(pdf_resp.content)
        ai_path.write_text(json.dumps(ai, ensure_ascii=False, indent=2), encoding='utf-8')
        aud_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

        row = {
            'profile_name': name,
            'seed': cand.get('seed'),
            'audit_overall': int(audit.get('overall_score', 0)),
            'audit_structural': int(audit.get('structural_integrity_score', 0)),
            'audit_tone': int(audit.get('tone_alignment_score', 0)),
            'audit_density': int(audit.get('density_score', 0)),
            'ai_forbidden_hits': ai_forbidden_hits,
            'pdf_path': str(pdf_path),
        }
        rows.append(row)

    summary = {
        'run_dir': str(run_dir),
        'rows': rows,
        'avg_audit_overall': round(sum(r['audit_overall'] for r in rows) / len(rows), 2),
        'ai_forbidden_hits_total': sum(r['ai_forbidden_hits'] for r in rows),
    }

    (run_dir / 'phase11_run7_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUMMARY_JSON=' + json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', type(e).__name__, str(e))
        traceback.print_exc()
        raise
