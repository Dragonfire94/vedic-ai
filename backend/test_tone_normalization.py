import unittest

from backend.report_engine import normalize_recommendation_tone


class TestToneNormalization(unittest.TestCase):
    def test_question_mark_removed(self):
        text = "- 기다려보기?\n- 이미 해보기"
        out = normalize_recommendation_tone(text, language="ko")
        self.assertEqual(out.splitlines()[0], "- 기다려보기")

    def test_haebolkka_replaced(self):
        text = "- 워킹 질문 작성해볼까?"
        out = normalize_recommendation_tone(text, language="ko")
        self.assertEqual(out.strip(), "- 워킹 질문 작성해보기")

    def test_non_bullet_unchanged(self):
        text = "질문해볼까?"
        out = normalize_recommendation_tone(text, language="ko")
        self.assertEqual(out, text)

    def test_language_guard(self):
        text = "- try it?"
        out = normalize_recommendation_tone(text, language="en")
        self.assertEqual(out, text)

    def test_chapter_scope(self):
        text = "## [Executive Summary]\n- 작성해볼까?\n## [Career & Success]\n- 작성해볼까?"
        out = normalize_recommendation_tone(text, language="ko", allowed_chapters={"Career & Success"})
        lines = out.splitlines()
        self.assertEqual(lines[1], "- 작성해볼까?")
        self.assertEqual(lines[3], "- 작성해보기")

    def test_gajyeobolkka_replaced(self):
        text = "- 한 번 시도 후 회고하는 습관 가져볼까?"
        out = normalize_recommendation_tone(text, language="ko")
        self.assertEqual(out.strip(), "- 한 번 시도 후 회고하는 습관 가져보기")
