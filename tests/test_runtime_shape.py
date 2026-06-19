from pathlib import Path
import unittest


class RuntimeShapeTest(unittest.TestCase):
    def test_cli_entrypoint_has_no_ui_framework_dependency(self):
        forbidden = "gra" + "dio"
        app_source = Path("app.py").read_text(encoding="utf-8").lower()

        self.assertNotIn(f"import {forbidden}", app_source)
        self.assertNotIn("gr.blocks", app_source)


if __name__ == "__main__":
    unittest.main()
