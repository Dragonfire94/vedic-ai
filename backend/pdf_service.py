import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import logging
logger = logging.getLogger("vedic_ai")

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent

# ------------------------------------------------------------------------------
# Pretendard/system Korean font setup
# ------------------------------------------------------------------------------
FONT_REGULAR_CANDIDATES = [
    MODULE_DIR / "fonts" / "Pretendard-Regular.ttf",
    REPO_ROOT / "assets" / "fonts" / "Pretendard-Regular.ttf",
]
FONT_BOLD_CANDIDATES = [
    MODULE_DIR / "fonts" / "Pretendard-Bold.ttf",
    REPO_ROOT / "assets" / "fonts" / "Pretendard-Bold.ttf",
]
SYSTEM_KOREAN_FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]

KOREAN_FONT_AVAILABLE = False
PDF_FONT_REG = 'Helvetica'
PDF_FONT_BOLD = 'Helvetica-Bold'
PDF_FONT_MONO = 'Courier'
PDF_FEATURE_AVAILABLE = False
PDF_FEATURE_ERROR: Optional[str] = None


def _first_existing_path(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _fontconfig_match(family: str) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}\n", family],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    raw = result.stdout.strip()
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _discover_system_korean_font() -> Optional[Path]:
    direct = _first_existing_path(SYSTEM_KOREAN_FONT_CANDIDATES)
    if direct:
        return direct

    for family in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        matched = _fontconfig_match(family)
        if matched:
            return matched

    try:
        result = subprocess.run(
            ["fc-list", ":lang=ko", "file"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        path_text = line.split(":", 1)[0].strip()
        if not path_text:
            continue
        candidate = Path(path_text)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def init_fonts() -> None:
    """Initialize PDF fonts with a safe fallback chain."""
    global KOREAN_FONT_AVAILABLE, PDF_FONT_REG, PDF_FONT_BOLD, PDF_FONT_MONO
    global PDF_FEATURE_AVAILABLE, PDF_FEATURE_ERROR

    KOREAN_FONT_AVAILABLE = False
    PDF_FONT_REG = 'Helvetica'
    PDF_FONT_BOLD = 'Helvetica-Bold'
    PDF_FONT_MONO = 'Courier'
    PDF_FEATURE_AVAILABLE = False
    PDF_FEATURE_ERROR = None

    try:
        regular = _first_existing_path(FONT_REGULAR_CANDIDATES)
        if regular:
            pdfmetrics.registerFont(TTFont('Pretendard', str(regular)))
            bold = _first_existing_path(FONT_BOLD_CANDIDATES)
            if bold:
                pdfmetrics.registerFont(TTFont('Pretendard-Bold', str(bold)))
                PDF_FONT_BOLD = 'Pretendard-Bold'
            else:
                PDF_FONT_BOLD = 'Pretendard'
                logger.warning(
                    'Pretendard-Bold.ttf not found; using Pretendard regular for bold style.'
                )

            KOREAN_FONT_AVAILABLE = True
            PDF_FONT_REG = 'Pretendard'
            PDF_FONT_MONO = 'Pretendard'
            PDF_FEATURE_AVAILABLE = True
            logger.info('Pretendard font loaded.')
            return

        system_font = _discover_system_korean_font()
        if system_font:
            pdfmetrics.registerFont(TTFont('KoreanFallback', str(system_font)))
            KOREAN_FONT_AVAILABLE = True
            PDF_FONT_REG = 'KoreanFallback'
            PDF_FONT_BOLD = 'KoreanFallback'
            PDF_FONT_MONO = 'KoreanFallback'
            PDF_FEATURE_AVAILABLE = True
            logger.info('System Korean font loaded: %s', system_font)
            return

        raise FileNotFoundError(
            f'No Korean font found in bundle candidates={FONT_REGULAR_CANDIDATES} '
            f'or system candidates={SYSTEM_KOREAN_FONT_CANDIDATES}.'
        )
    except Exception as e:
        PDF_FEATURE_ERROR = str(e)
        logger.error('Font initialization failed; PDF feature disabled: %s', e)
        return



class SouthIndianChart(Flowable):
    """Draw a diamond-style chart panel (North Indian style)."""
    def __init__(self, chart_data, width=400, height=400, is_d9=False):
        Flowable.__init__(self)
        self.chart_data = chart_data
        self.width = width
        self.height = height
        self.is_d9 = is_d9
    
    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        cx, cy = w / 2, h / 2
        size = min(w, h) * 0.8
        
        # ------------------------------------------------------------------------------
        pts = [
            (cx, cy + size/2),      # top
            (cx + size/2, cy),      # right
            (cx, cy - size/2),      # bottom
            (cx - size/2, cy),      # left
        ]
        
        c.setStrokeColor(colors.black)
        c.setLineWidth(2)
        
        # Inner guide lines
        p = c.beginPath()
        p.moveTo(pts[0][0], pts[0][1])
        for i in range(1, 4):
            p.lineTo(pts[i][0], pts[i][1])
        p.close()
        c.drawPath(p, stroke=1, fill=0)
        
        # Inner guide lines
        c.line(pts[0][0], pts[0][1], pts[2][0], pts[2][1])
        c.line(pts[1][0], pts[1][1], pts[3][0], pts[3][1])
        
        # House number labels (slot 1 shows the Ascendant house).
        houses_layout = [
            (cx, cy - size*0.15),           # 1
            (cx - size*0.25, cy - size*0.3), # 2
            (cx - size*0.35, cy - size*0.1), # 3
            (cx - size*0.35, cy + size*0.1), # 4
            (cx - size*0.25, cy + size*0.3), # 5
            (cx, cy + size*0.15),            # 6
            (cx + size*0.25, cy + size*0.3), # 7
            (cx + size*0.35, cy + size*0.1), # 8
            (cx + size*0.35, cy - size*0.1), # 9
            (cx + size*0.25, cy - size*0.3), # 10
            (cx + size*0.15, cy - size*0.1), # 11
            (cx - size*0.15, cy - size*0.1), # 12
        ]
        
        # ------------------------------------------------------------------------------
        planets = self.chart_data.get("planets", {})
        house_contents = {i: [] for i in range(1, 13)}
        
        for name, data in planets.items():
            house_num = data.get("house")
            if house_num:
                abbrev = name[:2].upper() if len(name) <= 3 else name[:3]
                house_contents[house_num].append(abbrev)
        
        # ------------------------------------------------------------------------------
        c.setFont(PDF_FONT_REG, 9)
        for house_num, (x, y) in enumerate(houses_layout, 1):
            content = house_contents.get(house_num, [])
            if content:
                text = ", ".join(content)
                c.drawCentredString(x, y, text)

# ------------------------------------------------------------------------------
# PDF rendering pipeline
# ------------------------------------------------------------------------------
def create_pdf_styles():
    """Create PDF paragraph styles (with safe config defaults)."""
    config = load_pdf_layout_config()
    font_cfg = config["fonts"]
    color_cfg = config["colors"]

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Title'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["title"],
        leading=font_cfg["title"] + 8,
        alignment=TA_CENTER,
        spaceAfter=14,
        textColor=colors.HexColor(color_cfg["title"]),
    ))

    styles.add(ParagraphStyle(
        name='ReportSubtitle',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=font_cfg["small"],
        leading=font_cfg["small"] + 4,
        alignment=TA_CENTER,
        spaceAfter=14,
        textColor=colors.HexColor(color_cfg.get("body", "#444444")),
    ))

    styles.add(ParagraphStyle(
        name='ChapterTitle',
        parent=styles['Heading1'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["chapter"],
        leading=font_cfg["chapter"] + 5,
        spaceAfter=10,
        textColor=colors.HexColor(color_cfg["chapter"]),
    ))

    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Heading2'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["subtitle"],
        leading=font_cfg["subtitle"] + 5,
        spaceAfter=8,
        textColor=colors.HexColor(color_cfg["chapter"]),
    ))

    styles.add(ParagraphStyle(
        name='Body',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=font_cfg["body"],
        leading=18,
        alignment=TA_JUSTIFY,
        spaceAfter=9,
        textColor=colors.HexColor(color_cfg["body"]),
    ))

    styles.add(ParagraphStyle(
        name='SummaryLead',
        parent=styles['Normal'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["body"] + 0.5,
        leading=19,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        textColor=colors.HexColor(color_cfg.get("summary_accent", color_cfg.get("chapter", "#111111"))),
    ))

    styles.add(ParagraphStyle(
        name='Small',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=font_cfg["small"],
        leading=font_cfg["small"] + 2,
        textColor=colors.grey,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name='InsightSpike',
        parent=styles['Normal'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["body"],
        leading=16,
        textColor=colors.HexColor(color_cfg["insight_spike"]),
        leftIndent=10,
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name='MetaLabel',
        parent=styles['Normal'],
        fontName=PDF_FONT_BOLD,
        fontSize=max(8, font_cfg["small"]),
        leading=max(10, font_cfg["small"] + 2),
        textColor=colors.HexColor(color_cfg.get("chapter", "#111111")),
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name='MetaValue',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=max(8, font_cfg["small"]),
        leading=max(10, font_cfg["small"] + 2),
        textColor=colors.HexColor(color_cfg.get("body", "#444444")),
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name='TableHeaderCell',
        parent=styles['Normal'],
        fontName=PDF_FONT_BOLD,
        fontSize=max(8, font_cfg["small"]),
        leading=max(10, font_cfg["small"] + 2),
        textColor=colors.whitesmoke,
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name='TableBodyCell',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=max(8, font_cfg["small"]),
        leading=max(10, font_cfg["small"] + 2),
        textColor=colors.HexColor(color_cfg["body"]),
        alignment=TA_LEFT,
    ))

    return styles


def load_pdf_layout_config() -> dict[str, Any]:
    """Build PDF output with robust fallback handling."""
    config_path = Path(__file__).resolve().parent / "pdf_layout_config.json"
    default_config = {
        "page": {
            "size": "A4",
            "margin_top": 36,
            "margin_bottom": 36,
            "margin_left": 48,
            "margin_right": 48,
        },
        "fonts": {"title": 22, "chapter": 18, "subtitle": 14, "body": 12, "small": 10},
        "colors": {
            "title": "#1F2A44",
            "chapter": "#1E3A5F",
            "body": "#2D3748",
            "insight_spike": "#B91C1C",
            "choice_fork": "#1D4ED8",
            "predictive": "#0F766E",
            "separator": "#D1D9E6",
            "panel_bg": "#F8FAFC",
            "table_alt": "#F1F5F9",
        },
        "chapters": {},
    }

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                default_config.update({k: v for k, v in loaded.items() if k in default_config})
    except Exception as e:
        logger.warning(f"PDF layout config load failed. Using defaults: {e}")

    return default_config


def _sanitize_pdf_text(value: Any) -> str:
    """ReportLab Paragraph safe text conversion."""
    if value is None:
        return ""
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clip_pdf_cell_text(value: Any, max_chars: int = 700) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    if not clipped:
        clipped = text[:max_chars]
    return f"{clipped} ...[truncated]"


def _to_pdf_paragraph(value: Any, style) -> Paragraph:
    return Paragraph(_sanitize_pdf_text(value), style)


def _extract_summary_text(summary_value: Any) -> str:
    if isinstance(summary_value, str):
        return summary_value.strip()
    if isinstance(summary_value, dict):
        for key in ("key_takeaway", "key_takeaways", "summary", "executive_summary", "overview"):
            candidate = summary_value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(summary_value, list):
        parts = [str(item).strip() for item in summary_value if isinstance(item, str) and item.strip()]
        return "\n".join(parts).strip()
    return ""


def _extract_key_forecast_text(forecast_value: Any) -> str:
    if isinstance(forecast_value, str):
        return forecast_value.strip()
    if isinstance(forecast_value, dict):
        for key in ("headline", "summary", "forecast", "text"):
            candidate = forecast_value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(forecast_value, list):
        parts = [str(item).strip() for item in forecast_value if isinstance(item, str) and item.strip()]
        return "\n".join(parts).strip()
    return ""


def render_report_payload_to_pdf(report_payload: dict[str, Any], styles, config: dict[str, Any]) -> list:
    """Render deterministic report payload with chapter-aware PDF layout."""
    chapter_blocks = report_payload.get("chapter_blocks", {}) if isinstance(report_payload, dict) else {}
    if not isinstance(chapter_blocks, dict):
        return []

    elements: list[Any] = []
    chapter_config = config.get("chapters", {}) if isinstance(config.get("chapters"), dict) else {}
    color_cfg = config.get("colors", {}) if isinstance(config.get("colors"), dict) else {}
    page_cfg = config.get("page", {}) if isinstance(config.get("page"), dict) else {}
    separator_color = colors.HexColor(color_cfg.get("separator", "#CCCCCC"))
    choice_color = colors.HexColor(color_cfg.get("choice_fork", "#0033AA"))
    predictive_color = colors.HexColor(color_cfg.get("predictive", "#006633"))
    forecast_color = colors.HexColor(color_cfg.get("forecast", "#7C3AED"))
    panel_bg = colors.HexColor(color_cfg.get("panel_bg", "#F8FAFC"))
    table_alt = colors.HexColor(color_cfg.get("table_alt", "#F1F5F9"))
    page_width = float(A4[0])
    margin_left = float(page_cfg.get("margin_left", 48))
    margin_right = float(page_cfg.get("margin_right", 48))
    content_width = max(320.0, page_width - margin_left - margin_right)
    table_label_col = max(120.0, min(170.0, content_width * 0.30))
    table_value_col = max(180.0, content_width - table_label_col)
    summary_text = _extract_summary_text(report_payload.get("summary") if isinstance(report_payload, dict) else None)
    summary_box_style = ParagraphStyle(
        "KeyTakeawayLane",
        parent=styles["SummaryLead"],
        fontName=PDF_FONT_BOLD,
        fontSize=styles["SummaryLead"].fontSize + 1.8,
        leading=max(styles["SummaryLead"].leading, styles["SummaryLead"].fontSize + 8),
        alignment=TA_LEFT,
        spaceAfter=0,
    )

    if summary_text:
        summary_title = Paragraph("Key Takeaway", styles["Subtitle"])
        summary_body = Paragraph(convert_markdown_bold(_sanitize_pdf_text(summary_text)), summary_box_style)
        summary_box = Table([[summary_body]], colWidths=[content_width])
        summary_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F9FF")),
            ("BOX", (0, 0), (-1, -1), 1.1, colors.HexColor("#7A9CC6")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_title)
        elements.append(summary_box)
        elements.append(Spacer(1, 12))

    for chapter in REPORT_CHAPTERS:
        fragments = chapter_blocks.get(chapter, [])
        if not isinstance(fragments, list) or not fragments:
            continue

        chapter_forecast_lines: list[str] = []

        if elements and chapter_config.get(chapter, {}).get("break_before"):
            elements.append(PageBreak())

        elements.append(Paragraph(_sanitize_pdf_text(chapter), styles["ChapterTitle"]))
        chapter_rule = Table([[""]], colWidths=[content_width], rowHeights=[0.5])
        chapter_rule.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, -1), 0.6, separator_color),
        ]))
        elements.append(chapter_rule)
        elements.append(Spacer(1, 8))

        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue

            if "spike_text" in fragment:
                elements.append(Paragraph(_sanitize_pdf_text(fragment.get("spike_text", "")), styles["InsightSpike"]))
                elements.append(Spacer(1, 8))
                continue

            key_forecast = _extract_key_forecast_text(fragment.get("key_forecast"))
            if key_forecast:
                chapter_forecast_lines.extend(
                    line.strip(" -•")
                    for line in key_forecast.splitlines()
                    if isinstance(line, str) and line.strip()
                )
                forecast_table = Table(
                    [[
                        _to_pdf_paragraph("Forecast", styles["TableHeaderCell"]),
                        Paragraph(convert_markdown_bold(_sanitize_pdf_text(key_forecast)), styles["Body"]),
                    ]],
                    colWidths=[table_label_col, table_value_col],
                )
                forecast_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, 0), forecast_color),
                    ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#F5F3FF")),
                    ('FONTNAME', (0, 0), (0, 0), PDF_FONT_BOLD),
                    ('FONTNAME', (1, 0), (1, 0), PDF_FONT_REG),
                    ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.45, colors.HexColor("#C4B5FD")),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                elements.append(forecast_table)
                elements.append(Spacer(1, 7))

            for field in ("title", "summary", "analysis", "implication", "examples", "micro_scenario", "long_term_projection"):
                value = fragment.get(field)
                if not value:
                    continue
                if field == "title":
                    style_name = "Subtitle"
                elif field == "summary":
                    style_name = "SummaryLead"
                else:
                    style_name = "Body"

                if field in {"micro_scenario", "long_term_projection"}:
                    value = f"{field.replace('_', ' ').title()}: {value}"
                elif field == "examples":
                    value = f"Practice Note: {value}"
                elements.append(_to_pdf_paragraph(value, styles[style_name]))

            # Soft divider between narrative cards for magazine-like rhythm.
            fragment_rule = Table([[""]], colWidths=[content_width], rowHeights=[0.4])
            fragment_rule.setStyle(TableStyle([
                ('LINEABOVE', (0, 0), (-1, -1), 0.35, separator_color),
            ]))
            elements.append(fragment_rule)
            elements.append(Spacer(1, 7))

            choice_fork = fragment.get("choice_fork")
            if isinstance(choice_fork, str) and choice_fork.strip():
                elements.append(_to_pdf_paragraph(choice_fork, styles["Body"]))
            elif isinstance(choice_fork, dict):
                path_a = choice_fork.get("path_a", {}) if isinstance(choice_fork.get("path_a"), dict) else {}
                path_b = choice_fork.get("path_b", {}) if isinstance(choice_fork.get("path_b"), dict) else {}
                table_rows = [
                    [_to_pdf_paragraph("Path", styles["TableHeaderCell"]), _to_pdf_paragraph("Trajectory", styles["TableHeaderCell"])],
                    [
                        _to_pdf_paragraph(f"A: {path_a.get('label', '-')}", styles["TableBodyCell"]),
                        _to_pdf_paragraph(_clip_pdf_cell_text(path_a.get("trajectory", "-")), styles["TableBodyCell"]),
                    ],
                    [
                        _to_pdf_paragraph("Emotional Cost", styles["TableBodyCell"]),
                        _to_pdf_paragraph(_clip_pdf_cell_text(path_a.get("emotional_cost", "-")), styles["TableBodyCell"]),
                    ],
                    [
                        _to_pdf_paragraph(f"B: {path_b.get('label', '-')}", styles["TableBodyCell"]),
                        _to_pdf_paragraph(_clip_pdf_cell_text(path_b.get("trajectory", "-")), styles["TableBodyCell"]),
                    ],
                    [
                        _to_pdf_paragraph("Emotional Cost", styles["TableBodyCell"]),
                        _to_pdf_paragraph(_clip_pdf_cell_text(path_b.get("emotional_cost", "-")), styles["TableBodyCell"]),
                    ],
                ]
                choice_table = Table(table_rows, colWidths=[table_label_col, table_value_col])
                choice_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), choice_color),
                    ('BACKGROUND', (0, 1), (-1, -1), panel_bg),
                    ('FONTNAME', (0, 0), (-1, 0), PDF_FONT_BOLD),
                    ('FONTNAME', (0, 1), (-1, -1), PDF_FONT_REG),
                    ('GRID', (0, 0), (-1, -1), 0.5, separator_color),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                elements.append(choice_table)
            elements.append(Spacer(1, 10))

            predictive = fragment.get("predictive_compression")
            if isinstance(predictive, dict):
                predictive_rows = [
                    [_to_pdf_paragraph("Window", styles["TableHeaderCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("window", "-"), max_chars=120), styles["TableBodyCell"])],
                    [_to_pdf_paragraph("Dominant Theme", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("dominant_theme", "-")), styles["TableBodyCell"])],
                    [_to_pdf_paragraph("Probability", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("probability_strength", "-"), max_chars=120), styles["TableBodyCell"])],
                    [_to_pdf_paragraph("Warning", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("structural_warning", "-")), styles["TableBodyCell"])],
                    [_to_pdf_paragraph("Alignment", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("recommended_alignment", "-")), styles["TableBodyCell"])],
                ]
                predictive_table = Table(predictive_rows, colWidths=[table_label_col, table_value_col])
                predictive_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), predictive_color),
                    ('BACKGROUND', (0, 1), (-1, -1), table_alt),
                    ('FONTNAME', (0, 0), (-1, 0), PDF_FONT_BOLD),
                    ('FONTNAME', (0, 1), (-1, -1), PDF_FONT_REG),
                    ('GRID', (0, 0), (-1, -1), 0.5, separator_color),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                elements.append(predictive_table)
                elements.append(Spacer(1, 12))

        compact_forecasts = [line for line in dict.fromkeys(chapter_forecast_lines) if line]
        if compact_forecasts:
            elements.append(Paragraph("Forecast Snapshot", styles["Subtitle"]))
            for line in compact_forecasts[:3]:
                elements.append(_to_pdf_paragraph(f"• {line}", styles["Small"]))
            elements.append(Spacer(1, 10))

    return elements

def _build_semantic_highlight_block(tag: str, body: str, styles) -> Table:
    palette = {
        "KEY": colors.HexColor("#FFF6CC"),
        "WARNING": colors.HexColor("#FFE5DB"),
        "STRATEGY": colors.HexColor("#E5F0FF"),
        "FORECAST": colors.HexColor("#E8F6FF"),
    }
    bg_color = palette.get(tag, colors.HexColor("#F5F5F5"))
    paragraph_text = f"<b>[{tag}]</b> {convert_markdown_bold(body.strip())}"
    block = Table([[Paragraph(paragraph_text, styles["Body"])]])
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D9E6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return block


def _build_icon_led_row(icon: str, body: str, styles) -> Table:
    row = Table(
        [[Paragraph(icon.strip(), styles["Body"]), Paragraph(convert_markdown_bold(body.strip()), styles["Body"]) ]],
        colWidths=[0.6 * cm, None],
    )
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return row

def parse_markdown_to_flowables(text: str, styles):
    """Convert markdown-like text into ReportLab flowables."""
    flowables = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 0.2*cm))
            continue
        
        # Heading
        if line.startswith('### '):
            clean_line = line[4:].replace('**', '')  # Remove bold markers from headings
            flowables.append(Paragraph(clean_line, styles['Subtitle']))
        elif line.startswith('## '):
            clean_line = line[3:].replace('**', '')
            flowables.append(Paragraph(clean_line, styles['ChapterTitle']))
        elif line.startswith('# '):
            clean_line = line[2:].replace('**', '')
            flowables.append(Paragraph(clean_line, styles['ReportTitle']))
        # Semantic blocks
        elif semantic_match := re.match(r'^\*\*\[(!?[A-Z_]+)\]\*\*\s+(.*)$', line):
            raw_tag = semantic_match.group(1)
            tag = raw_tag.lstrip('!')
            body = semantic_match.group(2)
            if tag in {"KEY", "WARNING", "STRATEGY", "FORECAST"}:
                flowables.append(_build_semantic_highlight_block(tag, body, styles))
                flowables.append(Spacer(1, 0.15*cm))
            else:
                flowables.append(Paragraph(convert_markdown_bold(line), styles['Body']))
        elif semantic_match := re.match(r'^\[(!?[A-Z_]+)\]\s+(.*)$', line):
            raw_tag = semantic_match.group(1)
            tag = raw_tag.lstrip('!')
            body = semantic_match.group(2)
            if tag in {"KEY", "WARNING", "STRATEGY", "FORECAST"}:
                flowables.append(_build_semantic_highlight_block(tag, body, styles))
                flowables.append(Spacer(1, 0.15*cm))
            else:
                flowables.append(Paragraph(convert_markdown_bold(line), styles['Body']))
        elif icon_match := re.match(r'^ICON:\s*(\S+)\s+(.*)$', line):
            icon = icon_match.group(1)
            body = icon_match.group(2)
            flowables.append(_build_icon_led_row(icon, body, styles))
            flowables.append(Spacer(1, 0.1*cm))
        # Section markers
        elif line.startswith('[') and line.endswith(']'):
            flowables.append(Spacer(1, 0.3*cm))
            clean_line = line.replace('**', '')
            flowables.append(Paragraph(f"<b>{clean_line}</b>", styles['ChapterTitle']))
        # List
        elif line.startswith('- ') or line.startswith('* '):
            clean_line = convert_markdown_bold(line[2:])
            flowables.append(Paragraph('- ' + clean_line, styles['Body']))
        else:
            # Regular paragraph
            clean_line = convert_markdown_bold(line)
            flowables.append(Paragraph(clean_line, styles['Body']))
    
    return flowables


