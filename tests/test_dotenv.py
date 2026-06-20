"""Tests for neuralplane.dotenv."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src/neuralplane/ is on the path for direct imports.
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


class TestLoadDotenv(unittest.TestCase):
    """Test cases for the load_dotenv function."""

    def setUp(self):
        # Save and clear environ for test isolation
        self._saved = dict(os.environ)
        for k in list(os.environ):
            del os.environ[k]

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def test_loads_key_value(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("FOO=bar\n")
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["FOO"], "bar")
        finally:
            os.unlink(path)

    def test_multiple_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("A=1\nB=2\nC=3\n")
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["A"], "1")
            self.assertEqual(os.environ["B"], "2")
            self.assertEqual(os.environ["C"], "3")
        finally:
            os.unlink(path)

    def test_ignores_blank_and_comment(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("# comment\n\n  \n  # another comment\nD=value\n")
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["D"], "value")
            # No extra keys from blank/comment lines
            self.assertNotIn("", [k for k in os.environ if not k.startswith("_")])
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        from neuralplane.dotenv import load_dotenv

        # Should not raise
        load_dotenv("/tmp/nonexistent_dotenv_12345.env")

    def test_strips_quotes(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write('SINGLE="val"\nDOUBLE=\'other\'\n')
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["SINGLE"], "val")
            self.assertEqual(os.environ["DOUBLE"], "other")
        finally:
            os.unlink(path)

    def test_trims_whitespace(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("  KEY1  =  val1  \nKEY2=value2\n")
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["KEY1"], "val1")
            self.assertEqual(os.environ["KEY2"], "value2")
        finally:
            os.unlink(path)

    def test_does_not_override_existing(self):
        os.environ["EXISTING"] = "old"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("EXISTING=new\nOTHER=val\n")
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["EXISTING"], "old")
            self.assertEqual(os.environ["OTHER"], "val")
        finally:
            os.unlink(path)

    def test_value_contains_equals(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("EQUATION=foo=bar\n")
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["EQUATION"], "foo=bar")
        finally:
            os.unlink(path)

    def test_custom_path(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("CUSTOM=customval\n")
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["CUSTOM"], "customval")
        finally:
            os.unlink(path)

    def test_skips_lines_without_equals(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("INVALID\nVALID=ok\nJUSTWORD\n")
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertNotIn("INVALID", os.environ)
            self.assertNotIn("JUSTWORD", os.environ)
            self.assertEqual(os.environ["VALID"], "ok")
        finally:
            os.unlink(path)

    def test_strips_inline_comment(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write('KEY=value  # inline comment\nNUM=42#no space\nURL=https://example.com\n')
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["KEY"], "value")
            self.assertEqual(os.environ["NUM"], "42#no space")
            self.assertEqual(os.environ["URL"], "https://example.com")
        finally:
            os.unlink(path)

    def test_strips_inline_comment_no_space_before_hash(self):
        """KEY=value# 說明 → strips to value (hash+space, no space before hash)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write('SLUG=myworkspace# 此為說明\nID=abc-123# comment text\n')
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ["SLUG"], "myworkspace")
            self.assertEqual(os.environ["ID"], "abc-123")
        finally:
            os.unlink(path)

    def test_value_is_comment_only(self):
        """KEY=# comment with no space before # → treated as empty."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write('KEY=# just a comment\nOTHER=val\n')
            path = f.name
        try:
            from neuralplane.dotenv import load_dotenv

            load_dotenv(path)
            self.assertEqual(os.environ.get("KEY", ""), "")
            self.assertEqual(os.environ["OTHER"], "val")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
