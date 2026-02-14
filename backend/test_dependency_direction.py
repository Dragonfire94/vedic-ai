"""Dependency direction guardrails for backend modules.

This test prevents accidental circular dependency introduction by enforcing:
main -> btr_engine -> astro_engine
and never the reverse.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


BACKEND_DIR = Path(__file__).resolve().parent


def _imported_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
    return modules


class TestDependencyDirection(unittest.TestCase):
    def test_btr_engine_does_not_import_main(self) -> None:
        modules = _imported_modules(BACKEND_DIR / "btr_engine.py")
        self.assertNotIn("backend.main", modules)
        self.assertNotIn("main", modules)

    def test_astro_engine_does_not_import_main_or_btr_engine(self) -> None:
        modules = _imported_modules(BACKEND_DIR / "astro_engine.py")
        self.assertNotIn("backend.main", modules)
        self.assertNotIn("main", modules)
        self.assertNotIn("backend.btr_engine", modules)
        self.assertNotIn("btr_engine", modules)


if __name__ == "__main__":
    unittest.main()

