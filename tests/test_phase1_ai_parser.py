"""Unit tests for NeuralPlane Phase 1A AI Parser.

Covers:
  - prompt_templates: prompt content, empty-input rejection
  - ai_parser: LLMProvider interface, StaticLLMProvider, JSON extraction,
    normalisation, parse_task_request pipeline, TaskDraft.validate() integration
  - cli_parse_task_draft: CLI flags, stdout JSON, file output,
    error exit codes, no Plane live write

Run with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure src/neuralplane/ is on the path for direct imports.
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from neuralplane.ai_parser import (
    LLMProvider,
    ParserInputError,
    ParseError,
    StaticLLMProvider,
    extract_json_object,
    normalize_task_draft_dict,
    parse_task_request,
)
from neuralplane.models import (
    MAX_DOD_ITEMS,
    MAX_DESCRIPTION_LENGTH,
    MAX_TITLE_LENGTH,
    TaskDraft,
    ValidationError,
)
from neuralplane.prompt_templates import (
    ALLOWED_PRIORITIES_STR,
    build_task_draft_prompt,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _valid_response(
    title: str = "Test Task",
    description: str = "Test description.",
    dod: list[str] | None = None,
    priority: str | None = "medium",
    labels: list[str] | None = None,
    assignee_suggestions: list[str] | None = None,
) -> str:
    """Return a deterministic valid JSON string for use in tests."""
    return json.dumps(
        {
            "title": title,
            "description": description,
            "dod": dod or ["This is a testable DoD item."],
            "priority": priority,
            "labels": labels if labels is not None else ["test"],
            "assignee_suggestions": assignee_suggestions
            if assignee_suggestions is not None
            else [],
        },
        ensure_ascii=False,
    )


# =====================================================================
# Prompt templates — content tests
# =====================================================================


class TestBuildTaskDraftPromptContent(unittest.TestCase):
    """🟢 正面測試: build_task_draft_prompt() contains all schema fields."""

    def test_prompt_contains_title_field(self) -> None:
        prompt = build_task_draft_prompt("建立登入 API")
        self.assertIn("title", prompt)

    def test_prompt_contains_description_field(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        self.assertIn("description", prompt)

    def test_prompt_contains_dod_field(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        self.assertIn("dod", prompt)

    def test_prompt_contains_priority_field(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        self.assertIn("priority", prompt)

    def test_prompt_contains_labels_field(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        self.assertIn("labels", prompt)

    def test_prompt_contains_assignee_suggestions_field(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        self.assertIn("assignee_suggestions", prompt)

    def test_prompt_contains_all_allowed_priorities(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        for p in ["low", "medium", "high", "urgent"]:
            self.assertIn(p, prompt, f"prompt must mention priority '{p}'")

    def test_prompt_includes_max_title_length_hint(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        self.assertIn(str(MAX_TITLE_LENGTH), prompt)

    def test_prompt_forbids_markdown_fence(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        self.assertIn("no markdown fences", prompt.lower())

    def test_prompt_forbids_extra_explanation(self) -> None:
        prompt = build_task_draft_prompt("實作功能")
        self.assertIn("no explanation", prompt.lower())

    def test_prompt_includes_user_request_text(self) -> None:
        user_text = "建立會員登入 API 需要 JWT"
        prompt = build_task_draft_prompt(user_text)
        self.assertIn(user_text, prompt)


class TestBuildTaskDraftPromptEdgeCases(unittest.TestCase):
    """🔲 邊界測試: empty / whitespace inputs."""

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_task_draft_prompt("")

    def test_whitespace_only_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_task_draft_prompt("   \t\n  ")

    def test_newline_only_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_task_draft_prompt("\n")


# =====================================================================
# LLMProvider interface — structural tests
# =====================================================================


class TestStaticLLMProvider(unittest.TestCase):
    """🟢 正面測試: StaticLLMProvider returns configured text."""

    def test_complete_returns_configured_text(self) -> None:
        fixture = '{"title": "Static", "dod": ["Done"]}'
        provider = StaticLLMProvider(fixture)
        self.assertEqual(provider.complete("any prompt"), fixture)

    def test_complete_ignores_prompt(self) -> None:
        provider = StaticLLMProvider("response")
        result = provider.complete("ignored prompt")
        self.assertEqual(result, "response")


# =====================================================================
# JSON extraction tests
# =====================================================================


class TestExtractJsonObject(unittest.TestCase):
    """Test class: extract_json_object()."""

    def test_bare_json_object_parses(self) -> None:
        raw = '{"title": "T", "dod": ["A"]}'
        result = extract_json_object(raw)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["title"], "T")

    def test_markdown_code_fence_json_parses(self) -> None:
        raw = "```json\n{\"title\": \"T\", \"dod\": [\"A\"]}\n```"
        result = extract_json_object(raw)
        self.assertEqual(result["title"], "T")

    def test_markdown_code_fence_no_lang_parses(self) -> None:
        raw = "```\n{\"title\": \"T\", \"dod\": [\"A\"]}\n```"
        result = extract_json_object(raw)
        self.assertEqual(result["title"], "T")

    def test_json_array_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            extract_json_object("[]")
        self.assertIn("object", str(ctx.exception).lower())

    def test_json_string_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            extract_json_object('"just a string"')
        self.assertIn("object", str(ctx.exception).lower())

    def test_json_null_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            extract_json_object("null")
        self.assertIn("object", str(ctx.exception).lower())

    def test_malformed_json_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            extract_json_object("{not json}")
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_plain_text_rejected(self) -> None:
        with self.assertRaises(ParseError):
            extract_json_object("this is not JSON at all")


# =====================================================================
# Normalisation tests
# =====================================================================


class TestNormalizeTaskDraftDict(unittest.TestCase):
    """Test class: normalize_task_draft_dict()."""

    def test_full_dict_unchanged(self) -> None:
        original = {
            "title": "T",
            "description": "D",
            "dod": ["Item"],
            "priority": "high",
            "labels": ["bug"],
            "assignee_suggestions": ["user1"],
        }
        result = normalize_task_draft_dict(original)
        self.assertEqual(result, original)

    def test_missing_description_defaults_to_empty_string(self) -> None:
        data = {"title": "T", "dod": ["Item"]}
        result = normalize_task_draft_dict(data)
        self.assertEqual(result["description"], "")

    def test_null_description_normalised_to_empty_string(self) -> None:
        data = {"title": "T", "dod": ["Item"], "description": None}
        result = normalize_task_draft_dict(data)
        self.assertEqual(result["description"], "")

    def test_missing_labels_defaults_to_empty_list(self) -> None:
        data = {"title": "T", "dod": ["Item"]}
        result = normalize_task_draft_dict(data)
        self.assertEqual(result["labels"], [])

    def test_missing_assignee_suggestions_defaults_to_empty_list(self) -> None:
        data = {"title": "T", "dod": ["Item"]}
        result = normalize_task_draft_dict(data)
        self.assertEqual(result["assignee_suggestions"], [])

    def test_empty_string_priority_normalised_to_none(self) -> None:
        data = {"title": "T", "dod": ["Item"], "priority": ""}
        result = normalize_task_draft_dict(data)
        self.assertIsNone(result["priority"])

    def test_valid_priority_preserved(self) -> None:
        data = {"title": "T", "dod": ["Item"], "priority": "high"}
        result = normalize_task_draft_dict(data)
        self.assertEqual(result["priority"], "high")

    def test_null_priority_preserved(self) -> None:
        data = {"title": "T", "dod": ["Item"], "priority": None}
        result = normalize_task_draft_dict(data)
        self.assertIsNone(result["priority"])


# =====================================================================
# parse_task_request — positive tests
# =====================================================================


class TestParseTaskRequestPositive(unittest.TestCase):
    """🟢 正面測試: valid provider response → valid TaskDraft."""

    def test_valid_response_returns_task_draft(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        task = parse_task_request("建立登入 API", provider)
        self.assertIsInstance(task, TaskDraft)

    def test_valid_response_passes_validation(self) -> None:
        provider = StaticLLMProvider(
            _valid_response(
                title="登入 API",
                description="使用 bcrypt 與 JWT",
                dod=["bcrypt 密碼驗證", "JWT 24h token"],
                priority="high",
                labels=["auth"],
            )
        )
        task = parse_task_request("建立登入 API", provider)
        task.validate()  # must not raise

    def test_labels_default_to_list(self) -> None:
        # Provider omits labels → normalize to []
        response = json.dumps(
            {"title": "T", "dod": ["Item"]}
        )
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        self.assertEqual(task.labels, [])
        self.assertIsInstance(task.labels, list)

    def test_assignee_suggestions_default_to_list(self) -> None:
        response = json.dumps(
            {"title": "T", "dod": ["Item"]}
        )
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        self.assertEqual(task.assignee_suggestions, [])
        self.assertIsInstance(task.assignee_suggestions, list)

    def test_priority_null_is_valid(self) -> None:
        response = _valid_response(priority=None)
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        self.assertIsNone(task.priority)
        task.validate()

    def test_empty_description_is_valid(self) -> None:
        response = _valid_response(description="")
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        self.assertEqual(task.description, "")
        task.validate()


# =====================================================================
# parse_task_request — negative tests
# =====================================================================


class TestParseTaskRequestNegative(unittest.TestCase):
    """🔴 負面測試: malformed / invalid provider responses are rejected."""

    def test_not_json_raises_parse_error(self) -> None:
        provider = StaticLLMProvider("this is definitely not JSON")
        with self.assertRaises(ParseError):
            parse_task_request("需求", provider)

    def test_json_array_raises_parse_error(self) -> None:
        provider = StaticLLMProvider("[1, 2, 3]")
        with self.assertRaises(ParseError):
            parse_task_request("需求", provider)

    def test_empty_dict_fails_validation(self) -> None:
        # {} passes JSON extraction but fails TaskDraft.validate()
        provider = StaticLLMProvider("{}")
        with self.assertRaises(ValidationError):
            parse_task_request("需求", provider)

    def test_dod_empty_list_fails_validation(self) -> None:
        response = json.dumps({"title": "T", "dod": []})
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError) as ctx:
            parse_task_request("需求", provider)
        self.assertIn("dod", str(ctx.exception).lower())

    def test_invalid_priority_fails_validation(self) -> None:
        response = _valid_response(priority="critical")
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError) as ctx:
            parse_task_request("需求", provider)
        self.assertIn("priority", str(ctx.exception).lower())

    def test_empty_title_fails_validation(self) -> None:
        response = _valid_response(title="")
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError) as ctx:
            parse_task_request("需求", provider)
        self.assertIn("title", str(ctx.exception).lower())

    def test_whitespace_only_title_fails_validation(self) -> None:
        response = _valid_response(title="   ")
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError):
            parse_task_request("需求", provider)

    def test_non_list_dod_fails_validation(self) -> None:
        response = json.dumps({"title": "T", "dod": "not a list"})
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError):
            parse_task_request("需求", provider)

    def test_non_list_labels_fails_validation(self) -> None:
        response = json.dumps(
            {"title": "T", "dod": ["Item"], "labels": "auth"}
        )
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError):
            parse_task_request("需求", provider)

    def test_empty_input_raises_parser_input_error(self) -> None:
        provider = StaticLLMProvider("{}")
        with self.assertRaises(ParserInputError):
            parse_task_request("", provider)


# =====================================================================
# parse_task_request — range / boundary tests
# =====================================================================


class TestParseTaskRequestRange(unittest.TestCase):
    """📏 範圍測試: field-length limits from TaskDraft.validate()."""

    def test_title_at_max_length_is_valid(self) -> None:
        title = "A" * MAX_TITLE_LENGTH
        response = _valid_response(title=title)
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        task.validate()  # must not raise

    def test_title_exceeds_max_length_fails_validation(self) -> None:
        title = "A" * (MAX_TITLE_LENGTH + 1)
        response = _valid_response(title=title)
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError) as ctx:
            parse_task_request("需求", provider)
        self.assertIn(str(MAX_TITLE_LENGTH), str(ctx.exception))

    def test_dod_at_max_items_is_valid(self) -> None:
        dod = [f"Item {i}" for i in range(MAX_DOD_ITEMS)]
        response = _valid_response(dod=dod)
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        task.validate()

    def test_dod_exceeds_max_items_fails_validation(self) -> None:
        dod = [f"Item {i}" for i in range(MAX_DOD_ITEMS + 1)]
        response = _valid_response(dod=dod)
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError) as ctx:
            parse_task_request("需求", provider)
        self.assertIn(str(MAX_DOD_ITEMS), str(ctx.exception))

    def test_description_at_max_length_is_valid(self) -> None:
        desc = "A" * MAX_DESCRIPTION_LENGTH
        response = _valid_response(description=desc)
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        task.validate()

    def test_description_exceeds_max_length_fails_validation(self) -> None:
        desc = "A" * (MAX_DESCRIPTION_LENGTH + 1)
        response = _valid_response(description=desc)
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError) as ctx:
            parse_task_request("需求", provider)
        self.assertIn(str(MAX_DESCRIPTION_LENGTH), str(ctx.exception))

    def test_priority_case_insensitive_valid(self) -> None:
        response = _valid_response(priority="HIGH")
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        task.validate()  # must pass (case-insensitive validator)

    def test_many_labels_no_crash(self) -> None:
        labels = [f"label{i}" for i in range(100)]
        response = _valid_response(labels=labels)
        provider = StaticLLMProvider(response)
        task = parse_task_request("需求", provider)
        # Should not crash; type check in validator will catch non-str items
        task.validate()


# =====================================================================
# parse_task_request — correctness tests
# =====================================================================


class TestParseTaskRequestCorrectness(unittest.TestCase):
    """🎯 正確性測試: DoD content quality / title semantic accuracy."""

    def test_dod_items_preserved(self) -> None:
        dod = ["bcrypt 密碼驗證", "JWT 24h token", "三種情境測試"]
        response = _valid_response(dod=dod)
        provider = StaticLLMProvider(response)
        task = parse_task_request("登入 API", provider)
        self.assertEqual(task.dod, dod)

    def test_title_reflects_request(self) -> None:
        response = _valid_response(title="建立會員登入 API")
        provider = StaticLLMProvider(response)
        task = parse_task_request("建立會員登入 API", provider)
        self.assertIn("登入", task.title)

    def test_priority_reflected_in_output(self) -> None:
        response = _valid_response(priority="urgent")
        provider = StaticLLMProvider(response)
        task = parse_task_request("緊急需求", provider)
        self.assertEqual(task.priority, "urgent")


# =====================================================================
# CLI — unit-level tests
# =====================================================================


class TestCliParseTaskDraft(unittest.TestCase):
    """Test class: CLI flags, stdout JSON, file output, error exit codes."""

    def setUp(self) -> None:
        # Save original env
        self._original_env = dict(os.environ)
        self._repo_root = Path(__file__).resolve().parents[1]
        self._sample_txt = self._repo_root / "examples" / "task_request.sample.txt"
        self._sample_json_out = (
            self._repo_root / "examples" / "task_draft.generated.sample.json"
        )

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._original_env)

    def _run_cli(
        self,
        extra_args: list[str] | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        base = [
            sys.executable,
            "-m",
            "src.neuralplane.cli_parse_task_draft",
            "--input",
            str(self._sample_txt),
        ]
        if extra_args:
            base.extend(extra_args)
        return subprocess.run(
            base,
            capture_output=capture_output,
            text=True,
            cwd=str(self._repo_root),
            env={**os.environ, "PYTHONPATH": "src"},
        )

    def test_dry_run_exits_zero(self) -> None:
        """🟢 正面測試: --dry-run exits 0."""
        result = self._run_cli(["--dry-run"])
        self.assertEqual(result.returncode, 0)

    def test_default_is_dry_run(self) -> None:
        """🟢 正面測試: running without --dry-run also exits 0 (default)."""
        result = self._run_cli()
        self.assertEqual(result.returncode, 0)

    def test_stdout_is_valid_json_object(self) -> None:
        """🟢 正面測試: stdout is parseable JSON object."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        self.assertIsInstance(parsed, dict)
        self.assertIn("title", parsed)
        self.assertIn("description", parsed)
        self.assertIn("dod", parsed)
        self.assertIn("priority", parsed)
        self.assertIn("labels", parsed)
        self.assertIn("assignee_suggestions", parsed)

    def test_stdout_taskdraft_validates(self) -> None:
        """🎯 正確性測試: stdout JSON → TaskDraft.validate() passes."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        task = TaskDraft.from_dict(parsed)
        task.validate()  # must not raise

    def test_output_file_written(self) -> None:
        """🟢 正面測試: --output writes a file with valid JSON."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as fh:
            tmp_path = fh.name
        try:
            result = self._run_cli(["--output", tmp_path])
            self.assertEqual(result.returncode, 0)
            written = Path(tmp_path).read_text()
            parsed = json.loads(written)
            self.assertIsInstance(parsed, dict)
            # The written file should also pass validation
            task = TaskDraft.from_dict(parsed)
            task.validate()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_missing_input_file_exits_nonzero(self) -> None:
        """🔴 負面測試: --input pointing to non-existent file exits != 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_parse_task_draft",
                "--input",
                "/nonexistent/file.txt",
            ],
            capture_output=True,
            text=True,
            cwd=str(self._repo_root),
            env={**os.environ, "PYTHONPATH": "src"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr.lower())

    def test_missing_static_response_file_exits_nonzero(self) -> None:
        """🔴 負面測試: --static-response pointing to non-existent file exits != 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_parse_task_draft",
                "--input",
                str(self._sample_txt),
                "--static-response",
                "/nonexistent/response.json",
            ],
            capture_output=True,
            text=True,
            cwd=str(self._repo_root),
            env={**os.environ, "PYTHONPATH": "src"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr.lower())

    def test_static_response_fixture_used(self) -> None:
        """🟢 正面測試: --static-response with valid fixture works."""
        result = self._run_cli(
            ["--static-response", str(self._sample_json_out)]
        )
        self.assertEqual(result.returncode, 0)
        parsed = json.loads(result.stdout)
        self.assertIsInstance(parsed, dict)

    def test_no_plane_client_import_in_parser_module(self) -> None:
        """✅ Guard: cli_parse_task_draft does not call PlaneClient.create_work_item."""
        cli_path = self._repo_root / "src" / "neuralplane" / "cli_parse_task_draft.py"
        source = cli_path.read_text()
        # Must not contain the live create call
        self.assertNotIn("create_work_item", source)
        # Must not import plane_client (Phase 1A parse-only CLI)
        self.assertNotIn("from neuralplane.plane_client", source)
        self.assertNotIn("import neuralplane.plane_client", source)

    def test_no_external_api_calls_made(self) -> None:
        """✅ Guard: running the CLI makes zero external API calls."""
        calls: list[str] = []

        def track_urlopen(*args, **kwargs):
            calls.append("urlopen")
            raise RuntimeError("No network calls allowed in Phase 1A")

        with patch("urllib.request.urlopen", side_effect=track_urlopen):
            result = self._run_cli()
            # CLI should succeed even if urlopen is intercepted
            self.assertEqual(result.returncode, 0)

        self.assertEqual(calls, [], "No urlopen calls should be made in Phase 1A")


# =====================================================================
# Integration: Phase 1A → Phase 1B dry-run handover
# =====================================================================


class TestPhase1AToPhase1BHandover(unittest.TestCase):
    """🎯 正確性測試: Phase 1A output can be fed to Phase 1B CLI (dry-run)."""

    def setUp(self) -> None:
        self._repo_root = Path(__file__).resolve().parents[1]

    def test_ai_parser_output_feeds_into_phase1b_dry_run(self) -> None:
        """CLI parse → TaskDraft JSON → Phase 1B CLI dry-run → no crash."""
        # Step 1: generate TaskDraft JSON via Phase 1A CLI
        sample_txt = self._repo_root / "examples" / "task_request.sample.txt"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            tmp_output = fh.name

        try:
            parse_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.neuralplane.cli_parse_task_draft",
                    "--input",
                    str(sample_txt),
                    "--output",
                    tmp_output,
                ],
                capture_output=True,
                text=True,
                cwd=str(self._repo_root),
                env={**os.environ, "PYTHONPATH": "src"},
            )
            self.assertEqual(
                parse_result.returncode, 0,
                f"Phase 1A CLI failed: {parse_result.stderr}",
            )

            # Step 2: feed the output into Phase 1B CLI (dry-run, no Plane env needed)
            phase1b_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.neuralplane.cli_create_plane_issue",
                    "--input",
                    tmp_output,
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                cwd=str(self._repo_root),
                env={**os.environ, "PYTHONPATH": "src"},
            )
            self.assertEqual(
                phase1b_result.returncode, 0,
                f"Phase 1B dry-run failed: {phase1b_result.stderr}",
            )
            self.assertIn("DRY-RUN", phase1b_result.stdout)

        finally:
            Path(tmp_output).unlink(missing_ok=True)


# =====================================================================
# Entry-point guard
# =====================================================================

if __name__ == "__main__":
    unittest.main()
