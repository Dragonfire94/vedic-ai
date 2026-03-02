# PATCH: v1.1.5 → v1.1.6 (micro)

> **적용 방법**: `PRODUCT_SPEC_PRD_v1_1_4.md` + `PATCH_v1_1_4_to_v1_1_5.md` 위에 이 패치를 추가 적용합니다.
> 이 패치가 명시하지 않은 모든 내용은 v1.1.5 기준을 따릅니다.

---

## 변경 로그 — 최상단에 아래 블록 삽입

```markdown
### v1.1.6 *(현재)*

#### [수정] PLANET_LABEL_MAP 상수 명시적 추가 (섹션 9.4, 7.1.3)
- v1.1.5 패치 테스트에서 `PLANET_LABEL_MAP` 검증을 요구했으나
  상수 추가 블록에는 `PLANET_DOMAIN_MAP`만 존재해 참조 불일치 발생.
- `PLANET_LABEL_MAP` (행성 코드 → 소비자 주제 라벨 dict)을 상수로 명시 추가.
  7.1.3 테이블을 코드로 고정한 것이므로 테이블과 항상 동기화.

#### [수정] next_mahadasha_date 결측/파싱 실패 시 fallback 규칙 추가 (섹션 7.1.9, 섹션 14.5)
- `next_mahadasha_date`가 null이거나 파싱에 실패하면 valid_until 계산이
  불가능해지는 미정의 상황 존재.
- fallback: `next_mahadasha_date` invalid/null이면 `as_of_local + 3년`으로 고정.

#### [수정] 렌더/후처리 계층 명시 — 엔진 산식 비범위와 충돌 방지 (섹션 7.1.4, 7.1.5, 7.1.6)
- `assign_life_stages`, `compute_life_highs_lows`, `compute_repeat_patterns`가
  엔진 산식 변경으로 오해될 수 있음.
- 이 함수들은 "엔진이 이미 계산한 다샤 날짜·점수를 받아 소비자 텍스트로
  변환하는 상업 렌더/후처리 계층"임을 명시. 엔진 내부 산식과 무관.

#### [수정] 전환 강도 분위 tie-break 및 엣지케이스 규칙 추가 (섹션 7.1.5)
- 동률, N<3, 경계값(정확히 33/67%) 처리가 미정의여서 구현자마다 라벨이 달라질 수 있음.
- 모든 엣지케이스를 결정론으로 고정.

#### [수정] 자동 테스트 / 수동 QA 항목 분리 (섹션 10.1, 10.2)
- v1.1.5 패치 `섹션 10.1 자동 테스트` 블록에 수동 QA 항목이 혼재.
- 자동 테스트와 수동 QA 항목을 각각의 섹션으로 분리.
```

---

## 섹션 9.4 — 패턴/정규식 정의 (항목 교체)

**v1.1.5 패치의 "행성 도메인 분류" 블록을 아래로 교체:**

```python
# ── 행성 도메인 분류 (LOCK, 반복 패턴 분석용) ────────────────────────────────
# 도메인 → [행성 코드] 매핑. 같은 도메인 행성이 2회 이상 등장하면 반복 패턴.
PLANET_DOMAIN_MAP: dict[str, list[str]] = {
    "관계":       ["VE", "MO"],
    "돈·커리어":  ["JU", "SU", "ME"],
    "건강·에너지": ["MA", "RA", "KE"],
}

# ── 행성 주제 라벨 (LOCK, 소비자 텍스트 변환용) ──────────────────────────────
# 섹션 7.1.3 테이블을 코드로 고정. 테이블과 반드시 동기화 유지.
# 변경은 PRD 버전업으로만 허용.
PLANET_LABEL_MAP: dict[str, dict] = {
    "SU": {"label": "자아 확립·리더십",   "tone": "상승"},
    "MO": {"label": "감정·관계·직관",     "tone": "중립"},
    "MA": {"label": "에너지·도전·행동",   "tone": "혼합"},
    "RA": {"label": "확장·혼돈·야망",     "tone": "혼합"},
    "JU": {"label": "성장·지혜·풍요",     "tone": "상승"},
    "SA": {"label": "압축·카르마·책임",   "tone": "경계"},
    "ME": {"label": "소통·분석·학습",     "tone": "중립"},
    "KE": {"label": "내면·해방·영성",     "tone": "경계"},
    "VE": {"label": "관계·창의·물질",     "tone": "상승"},
}
# 행성 코드 유효성 검사용
VALID_PLANET_CODES: frozenset[str] = frozenset(PLANET_LABEL_MAP.keys())

# ── 인생 단계 라벨 (LOCK) ─────────────────────────────────────────────────────
LIFE_STAGE_LABELS: list[str] = ["기반 형성", "방향 탐색", "사회적 확장", "영향력 축적"]

# ── 전환 강도 분위 임계치 (LOCK) ─────────────────────────────────────────────
# 사용 방법: 섹션 7.1.5 tie-break 규칙 참조.
TRANSITION_INTENSITY_THRESHOLDS: tuple[float, float] = (0.33, 0.67)
```

---

## 섹션 7.1.5 — 고점/저점 지도 산출 기준 (교체)

**v1.1.5 패치의 "전환 강도 라벨 기준 (LOCK)" 블록을 아래로 교체:**

```markdown
> **전환 강도 라벨 기준 + tie-break 규칙 (LOCK)**

**계산 방식**:
- 전환점 후보 = 인접 마하다샤 쌍의 `delta` 값 리스트.
- 정렬 후 `TRANSITION_INTENSITY_THRESHOLDS = (0.33, 0.67)` 분위수 기준으로 3등분.

**tie-break 및 엣지케이스 규칙 (LOCK)**:

| 상황 | 처리 규칙 |
|---|---|
| 전환점 총 수 N = 0 | 전환 강도 라벨 없음. 전환점 섹션 생략. |
| 전환점 총 수 1 ≤ N ≤ 2 | 전부 "중"으로 고정. 분위 계산 skip. |
| delta 동률이 분위 경계에 걸림 | 동률 그룹은 **모두 상위 등급**으로 처리. |
| 경계값 정확히 33번째 백분위 | **하** 등급에 포함 (경계값은 하위 등급). |
| 경계값 정확히 67번째 백분위 | **중** 등급에 포함 (경계값은 하위 등급). |

```python
# commercial_gate_helpers.py

def _get_transition_intensity(delta: float, sorted_deltas: list[float]) -> str:
    """
    단일 delta 값의 전환 강도 라벨 산출.
    sorted_deltas: 오름차순 정렬된 전체 delta 리스트.

    tie-break: 동률이 경계에 걸리면 상위 등급으로.
    경계값 자체는 하위 등급에 포함 (≤ 기준).
    N <= 2이면 호출부에서 "중"으로 고정하므로 이 함수는 호출하지 않음.
    """
    n = len(sorted_deltas)
    low_thresh  = sorted_deltas[int(n * TRANSITION_INTENSITY_THRESHOLDS[0])]
    high_thresh = sorted_deltas[int(n * TRANSITION_INTENSITY_THRESHOLDS[1])]

    if delta > high_thresh:
        return "상"
    if delta > low_thresh:
        return "중"
    return "하"
```
```

---

## 섹션 7.1.4 — 인생 단계 그룹핑 알고리즘 (계층 명시 추가)

**v1.1.5 패치 `assign_life_stages` 함수 독스트링 앞에 아래 블록 삽입:**

```markdown
> **계층 정의 (LOCK)**: `assign_life_stages`, `compute_life_highs_lows`, `compute_repeat_patterns`는
> **엔진이 이미 계산한** 다샤 날짜·pressure_score를 입력받아
> 소비자 텍스트 구조로 변환하는 **상업 렌더/후처리 계층**입니다.
> Vimshottari Dasha 날짜 계산, 행성 점수 산식 등 엔진 내부 로직과 무관합니다.
> 비범위 조항("엔진 산식 변경 금지")은 이 함수들에 적용되지 않습니다.
```

---

## 섹션 7.1.9 — valid_until 결측 fallback 추가

**v1.1.5 패치 `7.1.9 CTA/업셀 규칙` 블록 직전에 아래 삽입:**

```markdown
#### 7.1.8-a valid_until 결측 fallback 규칙 (LOCK)

`next_mahadasha_date` 처리 규칙:

| 상황 | valid_until 값 |
|---|---|
| `next_mahadasha_date` 정상값 존재 | `min(as_of_local + 3년, next_mahadasha_date)` |
| `next_mahadasha_date` null | `as_of_local + 3년` |
| `next_mahadasha_date` 파싱 실패 (형식 오류 등) | `as_of_local + 3년` + `meta.valid_until_fallback = true` 기록 |
| `next_mahadasha_date < as_of_local` (과거 날짜 입력 오류) | `as_of_local + 3년` + 운영 알림 트리거 |

```python
# 배포 파이프라인 / 렌더 계층

from datetime import date
from dateutil.relativedelta import relativedelta

def compute_valid_until_lifecycle(
    as_of_local: date,
    next_mahadasha_date: date | None,
) -> tuple[date, bool]:
    """
    인생 주기 리포트 valid_until 계산.
    반환: (valid_until, fallback_applied)
    fallback_applied=True이면 gate_summary에 valid_until_fallback=true 기록.
    """
    default = as_of_local + relativedelta(years=3)

    if next_mahadasha_date is None:
        return default, True

    if next_mahadasha_date <= as_of_local:
        trigger_ops_alert("next_mahadasha_date_past", str(next_mahadasha_date))
        return default, True

    return min(default, next_mahadasha_date), False
```
```

---

## 섹션 10.1 / 10.2 — 자동 테스트 / 수동 QA 분리

**v1.1.5 패치 "섹션 10.1 — 자동 테스트 추가 항목" 블록 전체를 아래로 교체:**

```markdown
## 섹션 10.1 — 자동 테스트 추가 항목 (인생 주기 리포트 전용)

- `assign_life_stages()`:
  - 다샤 9개 → 그룹 크기 [2, 2, 2, 3] 정확 분할
  - 다샤 4개 → 그룹 크기 [1, 1, 1, 1]
  - 다샤 12개 → 그룹 크기 [3, 3, 3, 3]
  - 다샤 0개 → 빈 리스트 반환 (오류 없음)

- `compute_life_highs_lows()`:
  - 다샤 9개: 상위 3 / 하위 3 / 전환점 5 정확 선정
  - 다샤 5개: 전환점이 4개(N-1)만 생성되는지 확인
  - 다샤 3개: 전환점이 2개 생성되는지 확인

- `_get_transition_intensity()` — 전환 강도 분위 케이스:
  - N=1 또는 N=2 → 호출부에서 "중" 고정, 함수 미호출 확인
  - 경계값 정확히 33번째 백분위 → "하" 반환
  - 경계값 정확히 67번째 백분위 → "중" 반환
  - 동률이 경계에 걸림 → 동률 그룹 전체 상위 등급 처리 확인

- `compute_repeat_patterns()`:
  - 동일 도메인 행성 2회 이상 등장 → 패턴 인식
  - 동일 도메인 행성 1회만 등장 → 패턴 제외

- `PLANET_LABEL_MAP` 변환:
  - 9개 행성 코드 각각 → 소비자 주제 라벨 + tone 정확 변환 (7.1.3 테이블 기준)
  - 유효하지 않은 코드 입력 → `VALID_PLANET_CODES` 검사로 오류 처리 확인

- `compute_valid_until_lifecycle()` — valid_until 케이스:
  - `next_mahadasha_date` 정상값: `min(+3년, next)` 중 이른 날짜 선택
  - `next_mahadasha_date = None` → `+3년`, `fallback_applied=True`
  - `next_mahadasha_date < as_of_local` → `+3년`, 운영 알림 트리거, `fallback_applied=True`

- 다음 3년 구체화 슬롯:
  - 3년 이내 부크티 전환 3개 → 3개 슬롯 출력 (패딩 없음)
  - 3년 이내 부크티 전환 0개 → 슬롯 없음, 고정 문구만 출력

- 모듈 분리 정적 검사:
  - `assign_life_stages`, `compute_life_highs_lows`, `compute_repeat_patterns`가
    `commercial_gate_helpers.py`에 위치하고 `commercial_quality_constants.py`를 import해 사용하는지
```

```markdown
## 섹션 10.2 — 수동 QA 추가 항목 (인생 주기 리포트 전용)

- 각 마하다샤 섹션 분량이 4~6줄 캡 이내인지 확인
- 인생 고점/저점 지도의 경계 구간에 대안 행동이 있는지 확인
- 예언 리스크 금지 표현 8종 육안 점검:
  - "반드시 일어납니다" 등 단정 표현 없는지
  - "큰돈을 법니다" 등 투자 유사 표현 없는지
  - "결혼이 옵니다" 등 관계 확정 예언 없는지
- "다음 3년 구체화" 섹션 끝에 업데이트 유도 고정 문구 존재 여부 확인
- `valid_until_fallback = true` 리포트: 소비자 본문에 fallback 값이 자연스럽게 노출되는지 확인
```

---

## 섹션 14.5 — P1 valid_until 로직 업데이트 (교체)

**v1.1.5 패치 `14.5 P1 — valid_until 로직 업데이트` 전체를 아래로 교체:**

```markdown
### 14.5 P1 — valid_until 로직 업데이트

1. `compute_valid_until_lifecycle()` 함수 구현 (섹션 7.1.8-a 코드 참조).
2. `next_mahadasha_date` 메타 필드 파이프라인 주입.
3. 결측/파싱 실패/과거 날짜 각각의 fallback 동작 단위 테스트 통과 확인.
4. `fallback_applied=True` 시 `gate_summary["valid_until_fallback"] = True` 기록.
5. `next_mahadasha_date < as_of_local` 시 운영 알림 트리거 동작 확인.
```

---

## 섹션 14.7 검증/수용 기준 (항목 추가)

**v1.1.5 패치 `14.7` 끝에 아래 추가:**

```markdown
7. `PLANET_LABEL_MAP` 9개 행성 코드 변환 테스트 통과.
8. 전환 강도 분위 tie-break 케이스 (N≤2, 경계값, 동률) 단위 테스트 통과.
9. valid_until fallback 3케이스 (null / 파싱 실패 / 과거 날짜) 단위 테스트 통과.
10. 자동/수동 테스트 항목 분리 확인 (10.1 자동, 10.2 수동에 혼재 없음).
```

---

## 로드맵 — 해당 행 교체

```markdown
| v1.1.5 | 인생 주기 리포트(Life Cycle Map) 제품 재설계 — 행성 주제 라벨 / 단계 그룹핑 / 고점저점 / 반복 패턴 / 예언 리스크 방어 |
| **v1.1.6 (현재)** | **PLANET_LABEL_MAP 명시 / next_mahadasha_date fallback / 렌더 계층 명시 / 전환 강도 tie-break / 테스트 분리** |
```

---

*패치 끝. 이 문서에서 명시하지 않은 모든 내용은 `PATCH_v1_1_4_to_v1_1_5.md` 및 `PRODUCT_SPEC_PRD_v1_1_4.md`를 따릅니다.*
