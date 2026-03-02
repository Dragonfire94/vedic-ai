from __future__ import annotations

from backend.main import (
    _active_chapter_order_for_style,
    _compute_body_paragraph_density_metrics,
    _reading_style_error_codes,
)


_BODY_PARA = (
    "이 문단은 검사 기준에서 본문으로 안정적으로 인식되도록 "
    "충분히 긴 길이와 문맥을 갖춘 설명 문장입니다."
)


def _build_doc(paragraph_counts: list[int]) -> str:
    headings = _active_chapter_order_for_style()[: len(paragraph_counts)]
    chunks: list[str] = []
    for heading, count in zip(headings, paragraph_counts):
        chunks.append(f"## [{heading}]")
        if count <= 0:
            chunks.append("")
            continue
        body = "\n\n".join(f"{_BODY_PARA} {idx + 1}." for idx in range(count))
        chunks.append(body)
    return "\n\n".join(chunks).strip()


def test_density_metrics_normalize_crlf_and_lf_consistently() -> None:
    lf = "## [Current Phase]\n\n" + _BODY_PARA + "\n\n" + _BODY_PARA
    crlf = lf.replace("\n", "\r\n")
    lf_metrics = _compute_body_paragraph_density_metrics(lf)
    crlf_metrics = _compute_body_paragraph_density_metrics(crlf)
    assert lf_metrics == crlf_metrics


def test_density_metrics_exclude_heading_list_and_short_caption_blocks() -> None:
    text = """## [Current Phase]

### Timing Map

- 항목 A
  이어지는 설명

짧다

이 문단은 검사 기준에서 본문으로 안정적으로 인식되도록 충분히 긴 길이와 문맥을 갖춘 설명 문장입니다.

이 문단도 검사 기준에서 본문으로 안정적으로 인식되도록 충분히 긴 길이와 문맥을 갖춘 설명 문장입니다.
"""
    metrics = _compute_body_paragraph_density_metrics(text)
    assert metrics["chapter_body_paragraph_count"]["Current Phase"] == 2
    assert metrics["excluded_blocks"]["heading"] == 1
    assert metrics["excluded_blocks"]["list"] == 1
    assert metrics["excluded_blocks"]["short"] == 1


def test_density_warning_band_is_non_hard() -> None:
    # 9 chapters * 2 + 1 chapter * 1 = 19 / 10 = 1.9
    doc = _build_doc([2, 2, 2, 2, 2, 2, 2, 2, 2, 1])
    codes = _reading_style_error_codes(doc)
    assert "warn_paragraph_density_low" in codes
    assert "paragraph_density_low" not in codes


def test_density_hard_fail_when_average_too_low() -> None:
    # 7 chapters * 2 + 3 chapters * 1 = 17 / 10 = 1.7
    doc = _build_doc([2, 2, 2, 2, 2, 2, 2, 1, 1, 1])
    codes = _reading_style_error_codes(doc)
    assert "paragraph_density_low" in codes


def test_density_hard_fail_when_zero_body_chapters_two_or_more() -> None:
    # Average >= 2.0 but 2 zero-body chapters should still hard-fail.
    doc = _build_doc([3, 3, 3, 3, 3, 3, 3, 3, 0, 0])
    metrics = _compute_body_paragraph_density_metrics(doc)
    assert metrics["avg_body_paragraphs_per_chapter"] >= 2.0
    codes = _reading_style_error_codes(doc)
    assert "paragraph_density_low" in codes
