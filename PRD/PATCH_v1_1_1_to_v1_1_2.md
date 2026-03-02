# PATCH: v1.1.1 → v1.1.2

> **적용 방법**: 이 파일을 `PRODUCT_SPEC_PRD_v1_1_1.md`와 함께 읽으십시오.  
> 아래 각 섹션은 v1.1.1 문서에서 **교체 또는 삽입**할 위치와 내용을 명시합니다.  
> 패치 미적용 섹션은 v1.1.1 원문 그대로 유효합니다.

---

## 문서 헤더 교체

```
# PRODUCT_SPEC_PRD v1.1.2 — 상업용 베딕 리포트 엔진 (Contract-Complete)

- **Status:** LOCKED (계약 변경 시 반드시 버전업 필요)
- **Last updated (Asia/Seoul):** 2026-03-02
- **Scope focus:** "그럴듯한 품질의 리포트를 안정적으로 출력" + HF16 정규화 / H2 fallback / 헤더 정규화 함수 / HF15 수치 확정
```

---

## 변경 로그 — 최상단에 아래 블록을 삽입

```markdown
### v1.1.2 *(현재)*

#### [수정] HF16 name 매칭 정규화 규칙 명시 (섹션 9.3, 9.4)
- **v1.1.1 문제**: `name_input.strip() in summary_text` 방식은 이름이
  "홍길동"으로 입력됐으나 리포트에 "홍길동님" 또는 " 홍길동 "으로 노출된 경우
  대소문자·공백·호칭 변화에 취약해 오탐(False FAIL) 가능.
- **v1.1.2 조치**: `_normalize_name()`으로 양쪽 정규화 후 비교.
  정규화 규칙(공백 압축, 괄호 제거, 호칭 제거, 소문자)을 PRD에 고정.

#### [수정] H2가 0개인 문서에 대한 파서 fallback 규칙 추가 (섹션 9.3)
- **v1.1.1 문제**: `parse_sections()`이 H2 전용이므로 H1 위주 산출물에서
  HF11/12/15/16이 "섹션 없음 → 전부 PASS"로 오탐될 수 있음.
- **v1.1.2 조치**: H2가 0개이면 `metrics_valid = false`로 마킹하고
  HF11/12/15/16 판정을 skip. 게이트 summary에 `h2_section_count: 0,
  gate_skip_reason: "no_h2_sections"` 기록. 출고 차단은 하지 않되,
  운영 알림을 트리거.

#### [수정] SECTION_EXCLUDE_ALIAS 헤더 정규화 함수 계약 추가 (섹션 9.3, 9.4)
- **v1.1.1 문제**: `header.strip().lower()`만 적용 — 마크다운 기호(`**`, `:`, `-`),
  다중 공백, 괄호가 포함된 헤더("**면책** / 윤리")는 alias 테이블 매칭에 실패.
- **v1.1.2 조치**: `_normalize_header_for_match()` 규칙을 PRD에 테이블로 고정.

#### [수정] HF15 품질 조건 수치·패턴 확정 (섹션 9.3, 9.4)
- **v1.1.1 문제**: "공감 문구 또는 행동 동사 1개"의 패턴 목록과
  "공백 제외" 문자수 산정 기준이 구현자 재량으로 남아 있음.
- **v1.1.2 조치**: `METHODOLOGY_CARD_QUALITY_RE` 전체 패턴을 확정 목록으로 고정.
  최소 길이 산정 기준을 "공백·줄바꿈 제거 후 글자 수"로 명시.
  스니펫 추출 범위(200자)도 명시.

#### [추가 보완] _normalize_header_for_match 마크다운 기호 제거
- v1.1.1에서 `_normalize_header_for_match()`가 `header.strip().lower()`만
  수행해 `**굵게**`, `## 면책:`, `면책 / 윤리` 등 마크다운 서식이 포함된
  헤더에서 alias 누락 가능. 정규화 규칙에 마크다운 기호 제거 단계 추가.
```

---

## 섹션 1 — LOCK 항목 요약 (12번 항목 교체)

**교체 대상 (v1.1.1 원문):**
> 12. **섹션 경계 파싱 규칙** (섹션 9.3): H2 기준 분리, H3 내부 블록 처리, 제외 섹션 exact match + alias 테이블

**교체 후:**
```markdown
12. **섹션 경계 파싱 규칙** (섹션 9.3): H2 기준 분리, H3 내부 블록 처리,
    `_normalize_header_for_match()` 정규화 규칙, 제외 섹션 exact match + alias 테이블,
    H2 0개 문서 fallback (`metrics_valid = false`, HF11/12/15/16 skip)
13. **HF16 name 정규화 규칙** (섹션 9.3): `_normalize_name()` 적용 기준 및
    호칭 제거 목록 고정
```

---

## 섹션 9.3 — 공통 섹션 경계 파싱 규칙 (전체 교체)

**v1.1.1 원문의 "공통 — 섹션 경계 파싱 규칙" 블록 전체를 아래로 교체:**

```markdown
#### 공통 — 섹션 경계 파싱 규칙 (v1.1.2 수정, LOCK)

> HF11/12/15/16은 **H2(`##`) 기준으로만 섹션을 분리**합니다.
> H3(`###`)은 섹션 내부 블록. H1은 문서 제목.

##### 헤더 정규화 함수 계약 (LOCK)

`_normalize_header_for_match()`는 헤더 문자열을 alias 테이블 비교 전에
아래 순서로 정규화합니다. **순서 변경 금지.**

| 순서 | 처리 | 예시 입력 → 출력 |
|---|---|---|
| 1 | 마크다운 강조 제거 (`**`, `*`, `__`, `_`) | `**면책**` → `면책` |
| 2 | 헤더 마커 제거 (앞쪽 `#` 문자 + 공백) | `## 면책` → `면책` |
| 3 | 구분 기호 정규화 (`/`, `\|`, `:`, `-`) → 단일 공백 | `면책/윤리:` → `면책 윤리 ` |
| 4 | 괄호류 제거 (`()`, `[]`, `{}`, `（）`) | `면책(필수)` → `면책 ` |
| 5 | 다중 공백 → 단일 공백 | `면책  윤리` → `면책 윤리` |
| 6 | 앞뒤 공백 제거 + 소문자 | `면책 윤리 ` → `면책 윤리` |

```python
# commercial_gate_helpers.py

import re

_HEADER_MD_BOLD_RE    = re.compile(r"\*{1,2}|_{1,2}")
_HEADER_MARKER_RE     = re.compile(r"^#+\s*")
_HEADER_DELIM_RE      = re.compile(r"[/|:\-]")
_HEADER_BRACKET_RE    = re.compile(r"[(){}\[\]（）]")
_HEADER_MULTI_SPACE   = re.compile(r"\s{2,}")

def _normalize_header_for_match(header: str) -> str:
    h = _HEADER_MD_BOLD_RE.sub("", header)
    h = _HEADER_MARKER_RE.sub("", h)
    h = _HEADER_DELIM_RE.sub(" ", h)
    h = _HEADER_BRACKET_RE.sub("", h)
    h = _HEADER_MULTI_SPACE.sub(" ", h)
    return h.strip().lower()
```

##### H2 0개 문서 fallback 규칙 (LOCK)

```python
# commercial_gate_helpers.py

def parse_sections(text: str) -> list[dict]:
    matches = list(SECTION_HEADER_RE.finditer(text))

    # H2가 0개인 문서: metrics_valid=False 마킹, 빈 리스트 반환
    if not matches:
        return []   # 호출부에서 빈 리스트 → h2_section_count=0 처리

    sections = []
    for i, m in enumerate(matches):
        header_raw = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        excluded = _is_excluded_header(header_raw)
        sections.append({"header": header_raw, "body": body, "excluded": excluded})
    return sections


def _is_excluded_header(header: str) -> bool:
    """정규화 후 SECTION_EXCLUDE_ALIAS exact match."""
    normalized = _normalize_header_for_match(header)
    return normalized in SECTION_EXCLUDE_ALIAS


# cheap_validation_gate.py 호출부 처리 패턴 (LOCK)
#
# sections = parse_sections(scored_surface_text)
# if not sections:
#     gate_summary["h2_section_count"] = 0
#     gate_summary["metrics_valid"] = False
#     gate_summary["gate_skip_reason"] = "no_h2_sections"
#     # HF11/12/15/16 판정 skip — 나머지 HF 1–10, 13, 14는 계속 실행
#     trigger_ops_alert("h2_section_count_zero", report_id)
# else:
#     gate_summary["h2_section_count"] = len(sections)
#     gate_summary["metrics_valid"] = True
#     # HF11/12/15/16 정상 실행
```

> **출고 정책**: H2 0개 문서는 HF11/12/15/16 판정을 skip하지만 **출고 차단하지 않습니다**.
> 대신 운영 알림을 트리거하고, 수동 QA 대상으로 분류합니다.
> 출고 차단 여부는 v1.2.x에서 운영 데이터 기반으로 결정합니다.
```

---

## 섹션 9.3 — HARD FAIL 16 판정 알고리즘 (전체 교체)

**v1.1.1 원문의 "HARD FAIL 16 — name_token_min_exposure" 블록 전체를 아래로 교체:**

```markdown
#### HARD FAIL 16 — name_token_min_exposure (v1.1.2 수정)

- **검사 조건**: `personalization_input.name` (①토큰) 비어있지 않을 때만 검사
- **검사 대상**: 한 장 요약 섹션 내 정규화된 이름 토큰 1회 이상 존재 여부

##### name 정규화 규칙 (LOCK)

`_normalize_name()`은 아래 순서로 정규화합니다. **순서 변경 금지.**

| 순서 | 처리 | 예시 입력 → 출력 |
|---|---|---|
| 1 | 앞뒤 공백 제거 | `" 홍길동 "` → `"홍길동"` |
| 2 | 내부 다중 공백 → 단일 공백 | `"홍  길동"` → `"홍 길동"` |
| 3 | 호칭 접미사 제거 (아래 목록) | `"홍길동님"` → `"홍길동"` |
| 4 | 괄호류 제거 | `"홍길동(가명)"` → `"홍길동"` |
| 5 | 소문자 변환 | `"HongGildong"` → `"honggildong"` |

**호칭 접미사 제거 목록 (LOCK):**
`님`, `씨`, `군`, `양`, `선생`, `선생님`, `고객님`, `사용자`, `씨 귀하`

```python
# commercial_quality_constants.py

NAME_HONORIFIC_RE = re.compile(
    r"(님|씨|군|양|선생님|선생|고객님|사용자|씨\s*귀하)\s*$"
)
```

```python
# commercial_gate_helpers.py

_NAME_BRACKET_RE    = re.compile(r"[（）()\[\]{}]")
_NAME_MULTI_SPACE   = re.compile(r"\s{2,}")

def _normalize_name(name: str) -> str:
    n = name.strip()
    n = _NAME_MULTI_SPACE.sub(" ", n)
    n = re.sub(NAME_HONORIFIC_RE, "", n).strip()
    n = _NAME_BRACKET_RE.sub("", n)
    return n.lower()


def check_name_token_exposure(text: str, name_input: str) -> bool:
    """
    이름 입력값이 있을 때 한 장 요약 섹션에
    정규화된 이름이 1회 이상 등장하면 PASS.
    """
    if not name_input.strip():
        return True  # 입력값 없으면 검사 스킵

    name_norm = _normalize_name(name_input)
    if not name_norm:
        return True  # 정규화 후 빈 문자열이면 스킵

    m = SUMMARY_SECTION_RE.search(text)
    if not m:
        return False  # 요약 섹션 없음 → HARD FAIL 16

    next_h2 = SECTION_HEADER_RE.search(text, m.end())
    summary_end = next_h2.start() if next_h2 else len(text)
    summary_text = text[m.start():summary_end]
    summary_norm = _normalize_name(summary_text)  # 본문도 동일 정규화

    return name_norm in summary_norm  # False → HARD FAIL 16
```
```

---

## 섹션 9.3 — HARD FAIL 15 판정 알고리즘 (전체 교체)

**v1.1.1 원문의 "HARD FAIL 15" 블록에서 `check_methodology_card()` 함수와 상수를 아래로 교체:**

```markdown
#### HARD FAIL 15 — 방법론 카드 품질 미달 (v1.1.2 수정)

> v1.1.1: 패턴 목록·문자수 산정 기준이 구현자 재량으로 남아 있던 문제 수정.
> v1.1.2: 패턴 목록 확정 + 문자수 산정 기준("공백·줄바꿈 제거 후") + 스니펫 범위(200자) 명시.

##### 최소 품질 조건 3가지 (LOCK)

| # | 조건 | 기준 |
|---|---|---|
| 1 | 항목 키워드(또는 대체어) 존재 | `text.lower()`에서 `alternates` 중 1개 이상 발견 |
| 2 | 키워드 발견 위치부터 200자 스니펫의 본문 최소 길이 | `re.sub(r"[\s\n]", "", snippet)` 기준 **25자 이상** |
| 3 | 스니펫 내 품질 패턴 1개 이상 | `METHODOLOGY_CARD_QUALITY_RE` 매칭 |

**`METHODOLOGY_CARD_QUALITY_RE` 확정 패턴 목록 (LOCK):**

아래 표는 "공감 문구 또는 행동/설명 동사"로 인정하는 패턴 전체 목록입니다.
목록 외 단어 추가는 PRD 버전업으로만 허용합니다.

| 카테고리 | 패턴 단어 |
|---|---|
| 설명/보여주기 | 설명, 보여, 알려, 보여줍니다, 알려줍니다 |
| 이해/도움 | 이해, 도움, 이해합니다, 도움이 됩니다 |
| 활용/준비 | 활용, 준비, 활용합니다, 준비합니다 |
| 서술형 종결 | 씁니다, 있습니다, 됩니다, 해줍니다, 드립니다, 입니다 |

```python
# commercial_quality_constants.py

METHODOLOGY_CARD_QUALITY_RE = re.compile(
    r"(설명|보여|알려|보여줍니다|알려줍니다"
    r"|이해|도움|이해합니다|도움이\s*됩니다"
    r"|활용|준비|활용합니다|준비합니다"
    r"|씁니다|있습니다|됩니다|해줍니다|드립니다|입니다)"
)

METHODOLOGY_CARD_SNIPPET_LEN = 200   # 스니펫 추출 최대 글자 수
METHODOLOGY_CARD_MIN_BODY_LEN = 25   # 공백·줄바꿈 제거 후 최소 글자 수
```

```python
# commercial_gate_helpers.py

def check_methodology_card(text: str) -> bool:
    """
    방법론 카드 6개 항목 각각에 대해 조건 3가지를 모두 만족해야 PASS.
    1. 키워드(또는 대체어) 존재
    2. 키워드 위치부터 METHODOLOGY_CARD_SNIPPET_LEN자 스니펫에서
       공백·줄바꿈 제거 후 METHODOLOGY_CARD_MIN_BODY_LEN자 이상
    3. 스니펫 내 METHODOLOGY_CARD_QUALITY_RE 패턴 1개 이상
    """
    text_lower = text.lower()
    for item in METHODOLOGY_CARD_ITEMS:
        found_pos = None
        for alt in item["alternates"]:
            idx = text_lower.find(alt)
            if idx != -1:
                found_pos = idx
                break

        if found_pos is None:
            return False  # 조건 1 실패 → HARD FAIL 15

        snippet = text[found_pos: found_pos + METHODOLOGY_CARD_SNIPPET_LEN]

        # 조건 2: 공백·줄바꿈 제거 후 글자 수
        body_len = len(re.sub(r"[\s\n]", "", snippet))
        if body_len < METHODOLOGY_CARD_MIN_BODY_LEN:
            return False  # 조건 2 실패 → HARD FAIL 15

        # 조건 3: 품질 패턴 존재
        if not re.search(METHODOLOGY_CARD_QUALITY_RE, snippet):
            return False  # 조건 3 실패 → HARD FAIL 15

    return True
```
```

---

## 섹션 9.4 — 패턴/정규식 정의 (교체 항목만)

**아래 항목들을 v1.1.1 원문의 해당 변수 정의와 교체합니다:**

```python
# ── 섹션 경계 파싱 — H2 전용 (LOCK, v1.1.2 동일 유지) ───────────────────────
SECTION_HEADER_RE = re.compile(r"(?m)^##\s+(.+)$")

# ── 헤더 정규화 (LOCK, v1.1.2 신규) — commercial_gate_helpers.py에 위치 ──────
# _normalize_header_for_match() 규칙은 섹션 9.3 참조.
# 아래 정규식들은 helpers에서만 사용하며 constants에 상수로 노출하지 않음.
# (정규식 자체가 단순하고 함수 내부에 캡슐화되어 외부 재사용 불필요)

# ── name 정규화 (LOCK, v1.1.2 신규) ──────────────────────────────────────────
NAME_HONORIFIC_RE = re.compile(
    r"(님|씨|군|양|선생님|선생|고객님|사용자|씨\s*귀하)\s*$"
)

# ── HF15 품질 조건 (LOCK, v1.1.2 확정) ──────────────────────────────────────
METHODOLOGY_CARD_QUALITY_RE = re.compile(
    r"(설명|보여|알려|보여줍니다|알려줍니다"
    r"|이해|도움|이해합니다|도움이\s*됩니다"
    r"|활용|준비|활용합니다|준비합니다"
    r"|씁니다|있습니다|됩니다|해줍니다|드립니다|입니다)"
)
METHODOLOGY_CARD_SNIPPET_LEN = 200
METHODOLOGY_CARD_MIN_BODY_LEN = 25   # 공백·줄바꿈 제거 후 글자 수 기준
```

---

## 섹션 9.6 — 관찰 지표 (항목 추가)

**v1.1.1 원문 관찰 지표 목록 끝에 아래 항목을 추가:**

```markdown
- `h2_section_count`: H2 섹션 수. 0이면 `metrics_valid = false`, HF11/12/15/16 skip
- `gate_skip_reason`: H2 0개 등 비정상 상황 사유 기록 (예: `"no_h2_sections"`)
- `name_token_exposure_norm_applied`: HF16 정규화 비교 사용 여부 (true/false)
```

---

## 섹션 10.1 — 자동 테스트 추가 항목

**v1.1.1 원문 "섹션 파싱 단위 테스트" 블록 끝에 아래를 추가:**

```markdown
- **헤더 정규화 테스트 (v1.1.2 신규)**:
  - `"**면책**"` → `"면책"` (강조 제거)
  - `"## 면책 / 윤리:"` → `"면책 윤리"` (구분 기호 → 공백, 콜론 제거)
  - `"면책(필수)"` → `"면책"` (괄호 제거)
  - `"  면책  윤리  "` → `"면책 윤리"` (다중 공백 → 단일, 앞뒤 제거)
  - 정규화 후 alias 테이블 exact match 통과 케이스
  - 정규화 후에도 alias 불일치 → 제외 안 됨 케이스 (오탐 방지)

- **H2 0개 문서 fallback 테스트 (v1.1.2 신규)**:
  - H1만 있는 문서 → `parse_sections()` 빈 리스트 반환
  - `gate_summary["h2_section_count"] == 0` 및 `metrics_valid == False` 기록 확인
  - `gate_skip_reason == "no_h2_sections"` 기록 확인
  - HF1–10, 13, 14는 정상 실행, HF11/12/15/16은 skip 확인
  - 출고 차단 없음 확인

- **HF16 name 정규화 테스트 (v1.1.2 신규)**:
  - 입력 `"홍길동"`, 요약 텍스트 `"홍길동님의 이번 시즌은"` → PASS
  - 입력 `"홍길동"`, 요약 텍스트 `" 홍길동 "` (공백 포함) → PASS
  - 입력 `"홍길동"`, 요약 텍스트 `"홍길동(가명)"` → PASS
  - 입력 `"홍길동"`, 요약 텍스트 `"이은정"` → FAIL
  - 입력 `""` (빈 문자열) → 검사 스킵 → PASS
  - 정규화 후 빈 문자열 → 검사 스킵 → PASS

- **HF15 수치 기준 테스트 (v1.1.2 신규)**:
  - 스니펫 공백 25자, 비공백 25자 → PASS (공백 제거 후 25자 충족)
  - 스니펫 공백 포함 60자이나 비공백 24자 → FAIL (공백 제거 후 24자)
  - 키워드 발견 위치 + 200자 스니펫 경계 정확도 케이스
  - `METHODOLOGY_CARD_QUALITY_RE` 패턴 목록 외 단어 → FAIL
  - 목록 내 단어 1개 → PASS
```

---

## 섹션 13 — 변경 관리 (LOCK 목록 끝에 항목 추가)

**v1.1.1 원문 금지 섹션 목록에 항목 추가:**

```markdown
  - 9 (...) / **`_normalize_header_for_match()` 정규화 순서** /
    **`_normalize_name()` 정규화 순서 및 호칭 접미사 목록** /
    **H2 0개 fallback 출고 정책**
```

---

## 섹션 14 — 실행 우선순위 (v1.1.2 전용)

**v1.1.1 섹션 14 전체를 아래로 교체:**

```markdown
## 14. v1.1.2 실행 우선순위 (Gap-to-Implementation)

> 목적: v1.1.1 오탐 위험 4가지를 제거하고 운영 공백 1가지를 추가 봉쇄한다.

### 14.1 P0 — _normalize_header_for_match() 구현

1. `commercial_gate_helpers.py`에 6단계 정규화 함수 추가 (섹션 9.3 코드 참조).
2. `_is_excluded_header()`가 `_normalize_header_for_match()` 경유 후 alias 테이블 비교하도록 교체.
3. 헤더 정규화 단위 테스트 6케이스 통과 확인 (섹션 10.1).

### 14.2 P0 — H2 0개 fallback 구현

1. `parse_sections()` 반환값이 빈 리스트일 때 호출부에서 `metrics_valid = false`, `h2_section_count = 0`, `gate_skip_reason = "no_h2_sections"` 기록.
2. HF11/12/15/16 판정 블록을 `if gate_summary["metrics_valid"]` 조건으로 감싸 skip 처리.
3. HF1–10, 13, 14는 조건 없이 계속 실행.
4. 운영 알림 트리거 연결 (출고 차단은 하지 않음).
5. H1만 있는 문서 단위 테스트 통과 확인.

### 14.3 P0 — HF16 _normalize_name() 구현

1. `commercial_quality_constants.py`에 `NAME_HONORIFIC_RE` 추가.
2. `commercial_gate_helpers.py`에 `_normalize_name()` 구현.
3. `check_name_token_exposure()` 내부를 정규화 비교로 교체.
4. `gate_summary`에 `name_token_exposure_norm_applied: true` 기록.
5. 정규화 케이스 단위 테스트 7케이스 통과 확인.

### 14.4 P0 — HF15 수치·패턴 확정 적용

1. `METHODOLOGY_CARD_QUALITY_RE` 를 확정 패턴 목록으로 교체.
2. `check_methodology_card()` 내 `re.sub(r"[\s\n]", "", snippet)` 기준으로 통일.
3. `METHODOLOGY_CARD_SNIPPET_LEN = 200` 상수 적용.
4. HF15 수치 기준 단위 테스트 4케이스 통과 확인.

### 14.5 검증/수용 기준

1. 기존 HF 16개 계약 회귀 없음.
2. 헤더 정규화 단위 테스트 전부 통과.
3. H2 0개 문서에서 HF11/12/15/16 skip + HF1–10,13,14 실행 확인.
4. HF16 정규화 비교: 호칭/공백 포함 이름 모두 PASS.
5. HF15 비공백 24자 → FAIL, 25자 → PASS 경계값 테스트 통과.
6. 모듈 분리 정적 검사 통과 (v1.1.1 14.6 유지).

### 14.6 비범위 (재확인)

- 엔진 산식 변경 금지.
- BTR ON 전환 금지.
- 프론트엔드/결제/로그인/대시보드 비범위.
- LLM 추가 호출 금지.
```

---

## 로드맵 — 해당 행 교체

**v1.1.1 원문:**
> | **v1.1.1 (현재)** | **HF16 명칭 확정 / 섹션 파싱 H2 고정 / 제외 exact match / HF15 품질 조건 추가 / 모듈 분리 / 결측 fallback** |

**교체 후:**
```markdown
| v1.1.1 | HF16 명칭 확정 / 섹션 파싱 H2 고정 / 제외 exact match / HF15 품질 조건 추가 / 모듈 분리 / 결측 fallback |
| **v1.1.2 (현재)** | **HF16 name 정규화 / H2 0개 fallback / 헤더 정규화 함수 / HF15 수치·패턴 확정** |
```

---

*패치 끝. 이 문서에서 명시하지 않은 모든 내용은 `PRODUCT_SPEC_PRD_v1_1_1.md` 원문을 따릅니다.*
