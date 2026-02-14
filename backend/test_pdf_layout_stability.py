import os
import sys
import types
import unittest
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate


os.environ.setdefault("SWE_ENFORCE_EPHE", "0")
if "timezonefinder" not in sys.modules:
    tz_mod = types.ModuleType("timezonefinder")

    class TimezoneFinder:  # pragma: no cover - startup shim only
        def timezone_at(self, **kwargs):
            return "UTC"

    tz_mod.TimezoneFinder = TimezoneFinder
    sys.modules["timezonefinder"] = tz_mod

from backend.main import _clip_pdf_cell_text, create_pdf_styles, load_pdf_layout_config, render_report_payload_to_pdf


class TestPdfLayoutStability(unittest.TestCase):
    def test_clip_pdf_cell_text_truncates_long_content(self):
        text = "x" * 2000
        clipped = _clip_pdf_cell_text(text, max_chars=120)
        self.assertLessEqual(len(clipped), 140)
        self.assertIn("[truncated]", clipped)

    def test_render_report_payload_handles_very_long_table_cells(self):
        long_text = " ".join(["Long content for wrapping"] * 200)
        payload = {
            "chapter_blocks": {
                "Behavioral Risks": [
                    {
                        "title": "Stress Fork",
                        "summary": "High-pressure decision split",
                        "choice_fork": {
                            "path_a": {
                                "label": "Control",
                                "trajectory": long_text,
                                "emotional_cost": long_text,
                            },
                            "path_b": {
                                "label": "Avoidance",
                                "trajectory": long_text,
                                "emotional_cost": long_text,
                            },
                        },
                        "predictive_compression": {
                            "window": "2026-2028",
                            "dominant_theme": long_text,
                            "probability_strength": "0.79",
                            "structural_warning": long_text,
                            "recommended_alignment": long_text,
                        },
                    }
                ]
            }
        }
        styles = create_pdf_styles()
        config = load_pdf_layout_config()
        story = render_report_payload_to_pdf(payload, styles, config)
        self.assertGreater(len(story), 0)

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        doc.build(story)
        self.assertGreater(len(buf.getvalue()), 0)


if __name__ == "__main__":
    unittest.main()
