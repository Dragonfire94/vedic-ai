# PATCH: v1.1.2 → v1.1.3 (micro)

> **적용 방법**: `PRODUCT_SPEC_PRD_v1_1_1.md` + `PATCH_v1_1_1_to_v1_1_2.md` 위에 이 패치를 추가 적용합니다.  
> 이 패치가 명시하지 않은 모든 내용은 v1.1.2 기준을 따릅니다.

---

## 변경 로그 — 최상단에 아래 블록 삽입

```markdown
### v1.1.3 *(현재)*

#### [수정] HF16 짧은 이름(≤ 2글자) 오탐 방지 (섹션 9.3)
- **v1.1.2 문제**: `name_norm in summary_norm` 단순 substring 매칭은
  이름이 "민", "이" 같이 1–2글자일 때 "민감한", "이번 시즌" 등
  무관한 단어에서 오탐(False PASS) 발생 가능.
- **v1.1.3 조치**: 정규화 후 이름 길이 ≤ 2글자이면 substring 대신
  단어 경계 + 호칭 경계 조건(`_name_boundary_re()`)으로 매칭.
  길이 ≥ 3글자는 기존 substring 방식 유지.

#### [수정] H2=0 누적 배포 임계치 명시 (섹션 9.3, 9.6)
- **v1.1.2 문제**: 개별 건 skip+알림은 처리했지만,
  배포 단위 누적 비율이 높아도 경고 기준이 없어 운영 드리프트 위험.
- **v1.1.3 조치**: 단일 배포 기준 `h2_section_count = 0` 비율이
  **5% 초과** 시 배포 경고(deploy_warn) 트리거. 규칙을 LOCK으로 고정.

#### [수정] _normalize_header_for_match() 테스트 케이스 추가 (섹션 10.1)
- `## [면책]` (대괄호 + H2 마커 혼합) 케이스를 명시 테스트에 추가.
```

---

## 섹션 9.3 — HARD FAIL 16 판정 알고리즘 (교체)

**v1.1.2 패치의 `check_name_token_exposure()` 함수를 아래로 교체:**

```markdown
##### 짧은 이름 경계 매칭 규칙 (≤ 2글자, LOCK)

정규화 후 이름 길이에 따라 매칭 방식을 분기합니다. **분기 기준(2글자) 변경 금지.**

| 이름 길이 (정규화 후) | 매칭 방식 | 이유 |
|---|---|---|
| ≥ 3글자 | substring (`name_norm in summary_norm`) | 오탐 위험 낮음 |
| ≤ 2글자 | 단어 경계 + 호칭 경계 정규식 | "민감한", "이번" 등 오탐 차단 |
| 정규화 후 빈 문자열 | 검사 스킵 → PASS | 입력값 무효 처리 |

**단어 경계 + 호칭 경계 조건 (≤ 2글자, LOCK):**  
이름 양쪽이 아래 중 하나이면 PASS로 인정합니다.

- 공백 (`\s`)
- 문자열 시작/끝
- 호칭 접미사 (`님|씨|군|양|선생|고객님|사용자`)
- 문장 부호 (`[,.!?。、]`)

```python
# commercial_quality_constants.py

NAME_BOUNDARY_SUFFIX_RE_STR = r"(?:님|씨|군|양|선생님?|고객님|사용자)?"
NAME_BOUNDARY_CHARS_RE_STR  = r"[\s,.!?。、]"

def _name_boundary_re(name_norm: str) -> re.Pattern:
    """
    ≤ 2글자 이름용 단어 경계 + 호칭 경계 패턴 생성.
    패턴: (문자열시작|경계문자) + 이름 + (호칭접미사)(경계문자|문자열끝)
    """
    escaped = re.escape(name_norm)
    boundary = rf"(?:^|{NAME_BOUNDARY_CHARS_RE_STR})"
    suffix   = NAME_BOUNDARY_SUFFIX_RE_STR
    trail    = rf"(?:{suffix})(?:{NAME_BOUNDARY_CHARS_RE_STR}|$)"
    return re.compile(rf"{boundary}{escaped}{trail}", re.MULTILINE)
```

```python
# commercial_gate_helpers.py

def check_name_token_exposure(text: str, name_input: str) -> bool:
    """
    이름 입력값이 있을 때 한 장 요약 섹션에
    정규화된 이름이 1회 이상 등장하면 PASS.
    ≤ 2글자 이름은 단어 경계 + 호칭 경계 조건으로 매칭.
    """
    if not name_input.strip():
        return True

    name_norm = _normalize_name(name_input)
    if not name_norm:
        return True

    m = SUMMARY_SECTION_RE.search(text)
    if not m:
        return False  # HARD FAIL 16

    next_h2 = SECTION_HEADER_RE.search(text, m.end())
    summary_end = next_h2.start() if next_h2 else len(text)
    summary_norm = _normalize_name(text[m.start():summary_end])

    if len(name_norm) <= 2:
        # 짧은 이름: 단어 경계 + 호칭 경계 패턴 매칭
        pattern = _name_boundary_re(name_norm)
        return bool(pattern.search(summary_norm))
    else:
        # 일반 이름: substring 매칭
        return name_norm in summary_norm
```
```

---

## 섹션 9.3 — 공통 섹션 경계 파싱 규칙 (H2=0 출고 정책 항목 교체)

**v1.1.2 패치의 "출고 정책" 블록을 아래로 교체:**

```markdown
> **출고 정책 및 누적 임계치 (LOCK)**:
>
> | 조건 | 조치 |
> |---|---|
> | 개별 리포트 `h2_section_count = 0` | HF11/12/15/16 skip, 운영 알림 트리거, 출고 허용 |
> | 단일 배포 기준 `h2_section_count = 0` 비율 **> 5%** | 배포 경고(deploy_warn) 트리거 — 배포 자동 차단은 하지 않으나 담당자 확인 필수 |
> | 단일 배포 기준 `h2_section_count = 0` 비율 **> 20%** | 배포 차단(deploy_block) 트리거 — 담당자 수동 해제 없이 배포 불가 |
>
> - 비율 계산 기준: `해당 배포의 전체 리포트 생성 건수` 분모.
> - 임계치(5% / 20%) 변경은 PRD 버전업으로만 허용.
> - `deploy_warn` / `deploy_block` 이벤트 발생 시 `gate_summary`에
>   `deploy_alert: "warn"` 또는 `"block"` 을 기록.
```

---

## 섹션 9.6 — 관찰 지표 (항목 교체)

**v1.1.2 패치에서 추가한 `h2_section_count` 항목을 아래로 교체:**

```markdown
- `h2_section_count`: H2 섹션 수. 0이면 `metrics_valid = false`, HF11/12/15/16 skip
- `gate_skip_reason`: H2 0개 등 비정상 상황 사유 기록 (예: `"no_h2_sections"`)
- `h2_zero_ratio_per_deploy`: 배포 단위 H2=0 리포트 비율. 5% 초과 → deploy_warn, 20% 초과 → deploy_block
- `deploy_alert`: 배포 경고 수준 (`"none"` / `"warn"` / `"block"`)
- `name_token_exposure_norm_applied`: HF16 정규화 비교 사용 여부 (true/false)
- `name_token_short_boundary_applied`: HF16 짧은 이름(≤ 2글자) 경계 매칭 사용 여부 (true/false)
```

---

## 섹션 9.4 — 패턴/정규식 정의 (항목 추가)

**v1.1.2 패치 `commercial_quality_constants.py` 블록 끝에 아래 추가:**

```python
# ── HF16 짧은 이름 경계 매칭 (LOCK, v1.1.3 신규) ─────────────────────────────
NAME_BOUNDARY_SUFFIX_RE_STR = r"(?:님|씨|군|양|선생님?|고객님|사용자)?"
NAME_BOUNDARY_CHARS_RE_STR  = r"[\s,.!?。、]"
# _name_boundary_re() 함수는 commercial_gate_helpers.py에 위치
# (패턴이 동적 생성이라 상수 단독 노출 불필요)

# ── H2=0 배포 임계치 (LOCK, v1.1.3 신규) ─────────────────────────────────────
H2_ZERO_DEPLOY_WARN_THRESHOLD  = 0.05   # 5% 초과 → deploy_warn
H2_ZERO_DEPLOY_BLOCK_THRESHOLD = 0.20   # 20% 초과 → deploy_block
```

---

## 섹션 10.1 — 자동 테스트 추가 항목

**v1.1.2 패치 "헤더 정규화 테스트" 블록 끝에 아래 케이스 추가:**

```markdown
  - `"## [면책]"` → `"면책"` (대괄호 + H2 마커 혼합) → alias `"면책"` 매칭 → PASS
  - `"## [면책 / 윤리]: 확인사항"` → `"면책 윤리 확인사항"` → alias `"면책 윤리"` 매칭 PASS
  - `"## [공지] 중요안내"` → `"공지 중요안내"` → alias 없음 → 제외 안 됨 (오탐 방지)
```

**v1.1.2 패치 "HF16 name 정규화 테스트" 블록 끝에 아래 케이스 추가:**

```markdown
  - 짧은 이름 `"민"`: 요약에 `"민님의 이번"` → PASS (호칭 경계)
  - 짧은 이름 `"민"`: 요약에 `"민감한 시기"` → FAIL (오탐 차단)
  - 짧은 이름 `"이"`: 요약에 `"이 씨의 흐름"` → PASS (호칭 경계)
  - 짧은 이름 `"이"`: 요약에 `"이번 시즌은"` → FAIL (오탐 차단)
  - 3글자 이름 `"홍길"`: 요약에 `"홍길동의"` → FAIL (substring 불일치)
  - 3글자 이름 `"홍길동"`: 요약에 `"홍길동의"` → PASS (substring 포함)
```

**v1.1.2 패치 "H2 0개 문서 fallback 테스트" 블록 끝에 아래 케이스 추가:**

```markdown
  - 배포 100건 중 6건 h2_section_count=0 (6%) → `deploy_alert: "warn"` 기록 확인
  - 배포 100건 중 21건 h2_section_count=0 (21%) → `deploy_alert: "block"` 기록 확인
  - 배포 100건 중 4건 h2_section_count=0 (4%) → `deploy_alert: "none"` 기록 확인
```

---

## 섹션 13 — 변경 관리 (LOCK 목록 추가)

**v1.1.2 패치 LOCK 목록 끝에 아래 추가:**

```markdown
    / **HF16 짧은 이름 경계 분기 기준(2글자)** /
    **H2=0 배포 임계치(5%/20%)**
```

---

## 섹션 14 — 실행 우선순위 (v1.1.3 항목 추가)

**v1.1.2 패치 섹션 14 "14.4 P0" 뒤에 아래 삽입:**

```markdown
### 14.4-a P0 — HF16 짧은 이름 경계 매칭 구현

1. `commercial_quality_constants.py`에 `NAME_BOUNDARY_SUFFIX_RE_STR`,
   `NAME_BOUNDARY_CHARS_RE_STR` 추가.
2. `commercial_gate_helpers.py`에 `_name_boundary_re()` 함수 구현.
3. `check_name_token_exposure()` 내 분기 로직 (`len(name_norm) <= 2`) 추가.
4. `gate_summary`에 `name_token_short_boundary_applied` 기록.
5. 짧은 이름 케이스 단위 테스트 6케이스 통과 확인.

### 14.4-b P1 — H2=0 배포 임계치 모니터링 구현

1. `commercial_quality_constants.py`에 `H2_ZERO_DEPLOY_WARN_THRESHOLD = 0.05`,
   `H2_ZERO_DEPLOY_BLOCK_THRESHOLD = 0.20` 추가.
2. 배포 파이프라인에서 배포 단위 h2_zero_ratio 집계 및 임계치 비교 로직 추가.
3. `deploy_alert: "warn"` / `"block"` / `"none"` 기록 및 알림 트리거 연결.
4. 배포 비율 임계치 경계값 단위 테스트 3케이스 통과 확인.
```

**v1.1.2 패치 섹션 14 "14.5 검증/수용 기준" 끝에 아래 추가:**

```markdown
6. 짧은 이름(≤ 2글자) PASS/FAIL 경계값 테스트 통과.
7. 배포 임계치 5%/20% 경계값 테스트 통과.
8. `## [면책]` 혼합 헤더 정규화 테스트 통과.
```

---

## 로드맵 — 해당 행 교체

```markdown
| v1.1.2 | HF16 name 정규화 / H2 0개 fallback / 헤더 정규화 함수 / HF15 수치·패턴 확정 |
| **v1.1.3 (현재)** | **HF16 짧은 이름 경계 매칭 / H2=0 배포 임계치 / 혼합 헤더 테스트** |
```

---

*패치 끝. 이 문서에서 명시하지 않은 모든 내용은 `PATCH_v1_1_1_to_v1_1_2.md` 및 `PRODUCT_SPEC_PRD_v1_1_1.md`를 따릅니다.*
