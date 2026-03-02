# PATCH: v1.1.6 → v1.1.7 (micro)

> **적용 방법**: v1.1.4 + v1.1.5 패치 + v1.1.6 패치 위에 이 패치를 추가 적용합니다.
> 이 패치가 명시하지 않은 모든 내용은 v1.1.6 기준을 따릅니다.

---

## 변경 로그 — 최상단에 아래 블록 삽입

```markdown
### v1.1.7 *(현재)*

#### [수정] 전환 강도 tie-break — 문서/코드 불일치 해소 (섹션 7.1.5)
- **v1.1.6 충돌**: 문서 "동률 그룹은 모두 상위 등급으로 처리"와
  코드 `if delta > high_thresh` (strict greater-than) 가 반대 동작.
  `>` 기준이면 동률은 하위 등급으로 내려가므로 문서와 모순.
- **v1.1.7 결정**: **코드 기준(`>` strict)으로 통일**.
  경계값은 하위 등급에 포함. "동률 상위 등급" 규칙 삭제.
  이유: 구현이 단순하고, 동률 재승격 로직 없이 예측 가능한 동작을 보장.

#### [수정] 파싱 실패 fallback — 계층 계약 명시 (섹션 7.1.8-a)
- **v1.1.6 불일치**: `compute_valid_until_lifecycle(next_mahadasha_date: date | None)`
  시그니처는 "파싱 실패" 상태를 직접 표현할 수 없음.
  함수 내부에서 파싱을 시도하지 않으므로 실패 상태가 유입되지 않아야 하는데
  이 계약이 문서에 없었음.
- **v1.1.7 결정**: **파싱은 호출 전 계층(메타 파이프라인)에서 책임진다**는 계약을 고정.
  파싱 실패 시 파이프라인에서 `None`으로 변환 후 함수를 호출.
  시그니처는 `date | None`으로 유지.
```

---

## 섹션 7.1.5 — 전환 강도 분위 tie-break 규칙 (해당 블록 전체 교체)

**v1.1.6 패치의 "전환 강도 라벨 기준 + tie-break 규칙 (LOCK)" 블록을 아래로 교체:**

```markdown
> **전환 강도 라벨 기준 (LOCK)**

**계산 방식**:
- 전환점 후보 = 인접 마하다샤 쌍의 `delta` 값 리스트.
- 정렬 후 `TRANSITION_INTENSITY_THRESHOLDS = (0.33, 0.67)` 분위수 기준으로 3등분.

**등급 판정 기준 (strict greater-than, LOCK)**:

| delta 값 | 등급 |
|---|---|
| `delta > high_thresh` | **상** |
| `low_thresh < delta ≤ high_thresh` | **중** |
| `delta ≤ low_thresh` | **하** |

**엣지케이스 규칙 (LOCK)**:

| 상황 | 처리 규칙 |
|---|---|
| 전환점 총 수 N = 0 | 전환 강도 라벨 없음. 전환점 섹션 생략. |
| 전환점 총 수 1 ≤ N ≤ 2 | 전부 "중"으로 고정. 분위 계산 skip. |
| 경계값 정확히 `low_thresh` | **하** 등급 (`≤` 기준) |
| 경계값 정확히 `high_thresh` | **중** 등급 (`≤` 기준) |

> **동률 처리**: 별도 재승격 없음. strict greater-than 기준 그대로 적용.
> 동률인 delta 값은 모두 동일한 등급을 받는다.

```python
# commercial_gate_helpers.py

def _get_transition_intensity(delta: float, sorted_deltas: list[float]) -> str:
    """
    단일 delta 값의 전환 강도 등급 산출.
    sorted_deltas: 오름차순 정렬된 전체 delta 리스트.

    등급 기준 (LOCK):
    - delta > high_thresh → "상"
    - low_thresh < delta ≤ high_thresh → "중"  (경계값 포함)
    - delta ≤ low_thresh → "하"                (경계값 포함)

    N ≤ 2이면 호출부에서 "중"으로 고정하므로 이 함수는 호출하지 않음.
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

## 섹션 7.1.8-a — valid_until 결측 fallback 규칙 (파싱 계약 항목 교체)

**v1.1.6 패치의 "valid_until 결측 fallback 규칙" 표에서 "파싱 실패" 행과 함수 정의를 아래로 교체:**

**표 — "파싱 실패" 행 교체:**

```markdown
| 상황 | valid_until 값 |
|---|---|
| `next_mahadasha_date` 정상값 존재 | `min(as_of_local + 3년, next_mahadasha_date)` |
| `next_mahadasha_date` null | `as_of_local + 3년`, `fallback_applied=True` |
| `next_mahadasha_date` 파싱 실패 | **파이프라인 계층에서 `None`으로 변환 후 호출** → null 케이스와 동일하게 처리 |
| `next_mahadasha_date < as_of_local` | `as_of_local + 3년`, 운영 알림 트리거, `fallback_applied=True` |
```

**파싱 계층 계약 (LOCK — 한 줄로 고정):**
```markdown
> **파싱 계층 계약 (LOCK)**: `next_mahadasha_date`의 파싱(문자열 → date 변환)은
> `compute_valid_until_lifecycle()` 호출 **이전** 계층(메타 파이프라인)에서 책임진다.
> 파싱에 실패하면 파이프라인에서 `None`으로 변환한 뒤 함수를 호출한다.
> `compute_valid_until_lifecycle()`은 파싱을 수행하지 않으며,
> `date | None`만 수신한다고 가정한다.
```

**함수 정의 (시그니처 및 독스트링 교체):**

```python
# 배포 파이프라인 / 렌더 계층

from datetime import date
from dateutil.relativedelta import relativedelta

def compute_valid_until_lifecycle(
    as_of_local: date,
    next_mahadasha_date: date | None,  # 파싱 실패는 호출 전 None으로 변환 완료
) -> tuple[date, bool]:
    """
    인생 주기 리포트 valid_until 계산.

    전제 (LOCK):
    - next_mahadasha_date는 호출 전 파싱이 완료된 date 또는 None.
    - 파싱 자체는 이 함수의 책임 밖.

    반환: (valid_until, fallback_applied)
    - fallback_applied=True → gate_summary["valid_until_fallback"] = True 기록.
    """
    default = as_of_local + relativedelta(years=3)

    if next_mahadasha_date is None:
        return default, True

    if next_mahadasha_date <= as_of_local:
        trigger_ops_alert("next_mahadasha_date_past", str(next_mahadasha_date))
        return default, True

    return min(default, next_mahadasha_date), False
```

---

## 섹션 10.1 — 자동 테스트 (케이스 교체)

**v1.1.6 패치 `_get_transition_intensity()` 테스트 블록을 아래로 교체:**

```markdown
- `_get_transition_intensity()` — 전환 강도 등급 케이스:
  - N=1 또는 N=2 → 호출부에서 "중" 고정, 함수 미호출 확인
  - delta가 `low_thresh`와 정확히 같음 → "하" 반환 (경계값 하위 등급)
  - delta가 `high_thresh`와 정확히 같음 → "중" 반환 (경계값 하위 등급)
  - delta가 `high_thresh` 초과 → "상" 반환
  - delta가 `low_thresh` 초과, `high_thresh` 이하 → "중" 반환
  - delta가 `low_thresh` 이하 → "하" 반환
  - 동률 delta 값 2개가 경계에 걸림 → 둘 다 동일 등급 (재승격 없음) 확인
```

**v1.1.6 패치 `compute_valid_until_lifecycle()` 테스트 블록을 아래로 교체:**

```markdown
- `compute_valid_until_lifecycle()` — valid_until 케이스:
  - `next_mahadasha_date` 정상값, +3년보다 이름 → `next_mahadasha_date` 반환, `fallback_applied=False`
  - `next_mahadasha_date` 정상값, +3년보다 늦음 → `as_of_local + 3년` 반환, `fallback_applied=False`
  - `next_mahadasha_date = None` → `+3년`, `fallback_applied=True`
  - `next_mahadasha_date < as_of_local` → `+3년`, 운영 알림 트리거, `fallback_applied=True`
  - 파싱 실패 케이스: 파이프라인에서 `None` 변환 후 함수 호출 → null 케이스와 동일 동작 확인
```

---

## 로드맵 — 해당 행 교체

```markdown
| v1.1.6 | PLANET_LABEL_MAP 명시 / next_mahadasha_date fallback / 렌더 계층 명시 / 전환 강도 tie-break / 테스트 분리 |
| **v1.1.7 (현재)** | **전환 강도 문서·코드 통일(strict >) / 파싱 계층 계약 고정** |
```

---

*패치 끝. 이 문서에서 명시하지 않은 모든 내용은 v1.1.6 기준을 따릅니다.*
