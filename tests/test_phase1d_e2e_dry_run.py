"""Unit tests for NeuralPlane Phase 1D E2E Dry-run Orchestrator.

Covers:
  - E2EDryRunReport dataclass / to_dict()
  - build_dry_run_plane_config() defaults
  - task_draft_to_dict() serialisation
  - run_e2e_dry_run() orchestration pipeline (Phase 1A → TaskDraft → Phase 1B)
  - CLI flags, stdout JSON, file output, error exit codes
  - No external API calls (source guard + patch)
  - No PlaneClient.create_work_item() invocation

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
    ParserInputError,
    ParseError,
    StaticLLMProvider,
)
from neuralplane.e2e_dry_run import (
    E2EDryRunReport,
    build_dry_run_plane_config,
    run_e2e_dry_run,
    task_draft_to_dict,
)
from neuralplane.models import TaskDraft, ValidationError


# ----------------------------------------------------------------------
# Fixtures / helpers
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
            "assignee_suggestions": (
                assignee_suggestions
                if assignee_suggestions is not None
                else []
            ),
        },
        ensure_ascii=False,
    )


# =====================================================================
# E2EDryRunReport dataclass
# =====================================================================


class TestE2EDryRunReportToDict(unittest.TestCase):
    """🟢 正面測試: E2EDryRunReport.to_dict() contains all required fields."""

    def test_to_dict_contains_dry_run(self) -> None:
        report = E2EDryRunReport(
            dry_run=True,
            external_api_calls=0,
            task_draft={},
            plane_endpoint="",
            plane_payload={},
        )
        self.assertIn("dry_run", report.to_dict())

    def test_to_dict_contains_external_api_calls(self) -> None:
        report = E2EDryRunReport(
            dry_run=True,
            external_api_calls=0,
            task_draft={},
            plane_endpoint="",
            plane_payload={},
        )
        self.assertIn("external_api_calls", report.to_dict())

    def test_to_dict_contains_task_draft(self) -> None:
        report = E2EDryRunReport(
            dry_run=True,
            external_api_calls=0,
            task_draft={"title": "T"},
            plane_endpoint="",
            plane_payload={},
        )
        self.assertIn("task_draft", report.to_dict())

    def test_to_dict_contains_plane_endpoint(self) -> None:
        report = E2EDryRunReport(
            dry_run=True,
            external_api_calls=0,
            task_draft={},
            plane_endpoint="https://api.plane.so/ws/projects/p/work-items/",
            plane_payload={},
        )
        self.assertIn("plane_endpoint", report.to_dict())

    def test_to_dict_contains_plane_payload(self) -> None:
        report = E2EDryRunReport(
            dry_run=True,
            external_api_calls=0,
            task_draft={},
            plane_endpoint="",
            plane_payload={"name": "T"},
        )
        self.assertIn("plane_payload", report.to_dict())

    def test_to_dict_contains_warnings(self) -> None:
        report = E2EDryRunReport(
            dry_run=True,
            external_api_calls=0,
            task_draft={},
            plane_endpoint="",
            plane_payload={},
            warnings=["some warning"],
        )
        d = report.to_dict()
        self.assertIn("warnings", d)
        self.assertEqual(d["warnings"], ["some warning"])

    def test_warnings_default_to_empty_list(self) -> None:
        report = E2EDryRunReport(
            dry_run=True,
            external_api_calls=0,
            task_draft={},
            plane_endpoint="",
            plane_payload={},
        )
        self.assertEqual(report.warnings, [])


# =====================================================================
# build_dry_run_plane_config
# =====================================================================


class TestBuildDryRunPlaneConfig(unittest.TestCase):
    """🟢 正面測試: build_dry_run_plane_config() produces a usable config."""

    def test_returns_plane_config(self) -> None:
        from neuralplane.plane_client import PlaneConfig
        cfg = build_dry_run_plane_config()
        self.assertIsInstance(cfg, PlaneConfig)

    def test_default_endpoint_contains_api_v1(self) -> None:
        cfg = build_dry_run_plane_config()
        url = cfg.build_work_items_url()
        self.assertIn("/api/v1/", url)

    def test_default_endpoint_contains_dry_run_workspace(self) -> None:
        cfg = build_dry_run_plane_config()
        url = cfg.build_work_items_url()
        self.assertIn("[dry-run-workspace]", url)

    def test_default_endpoint_contains_dry_run_project(self) -> None:
        cfg = build_dry_run_plane_config()
        url = cfg.build_work_items_url()
        self.assertIn("[dry-run-project]", url)

    def test_custom_base_url_used_in_endpoint(self) -> None:
        cfg = build_dry_run_plane_config(base_url="https://plane.example.com")
        url = cfg.build_work_items_url()
        self.assertTrue(url.startswith("https://plane.example.com/"))

    def test_custom_workspace_slug_used_in_endpoint(self) -> None:
        cfg = build_dry_run_plane_config(workspace_slug="my-workspace")
        url = cfg.build_work_items_url()
        self.assertIn("/workspaces/my-workspace/", url)

    def test_custom_project_id_used_in_endpoint(self) -> None:
        cfg = build_dry_run_plane_config(project_id="proj-abc-123")
        url = cfg.build_work_items_url()
        self.assertIn("/projects/proj-abc-123/", url)

    def test_state_id_is_dry_run_placeholder(self) -> None:
        cfg = build_dry_run_plane_config()
        self.assertEqual(cfg.default_state_id, "[dry-run-state]")

    def test_api_key_is_dry_run_placeholder(self) -> None:
        cfg = build_dry_run_plane_config()
        self.assertEqual(cfg.api_key, "[dry-run-no-key]")


# =====================================================================
# task_draft_to_dict
# =====================================================================


class TestTaskDraftToDict(unittest.TestCase):
    """🟢 正面測試: task_draft_to_dict() serialises all fields."""

    def test_all_fields_present(self) -> None:
        task = TaskDraft(
            title="Login API",
            description="Use bcrypt and JWT",
            dod=["bcrypt hashing"],
            priority="high",
            assignee_suggestions=["alice"],
            labels=["auth"],
        )
        d = task_draft_to_dict(task)
        self.assertEqual(d["title"], "Login API")
        self.assertEqual(d["description"], "Use bcrypt and JWT")
        self.assertEqual(d["dod"], ["bcrypt hashing"])
        self.assertEqual(d["priority"], "high")
        self.assertEqual(d["assignee_suggestions"], ["alice"])
        self.assertEqual(d["labels"], ["auth"])

    def test_optional_fields_defaults(self) -> None:
        task = TaskDraft(title="T", description="D", dod=["Done"])
        d = task_draft_to_dict(task)
        self.assertIsNone(d["priority"])
        self.assertEqual(d["assignee_suggestions"], [])
        self.assertEqual(d["labels"], [])


# =====================================================================
# run_e2e_dry_run — positive tests
# =====================================================================


class TestRunE2EDryRunPositive(unittest.TestCase):
    """🟢 正面測試: run_e2e_dry_run() with valid inputs."""

    def test_returns_e2e_dry_run_report(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("建立登入 API", provider)
        self.assertIsInstance(report, E2EDryRunReport)

    def test_dry_run_is_true(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        self.assertIs(report.dry_run, True)

    def test_external_api_calls_is_zero(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        self.assertEqual(report.external_api_calls, 0)

    def test_task_draft_is_valid_task_draft_dict(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        self.assertIsInstance(report.task_draft, dict)
        self.assertIn("title", report.task_draft)
        self.assertIn("dod", report.task_draft)

    def test_task_draft_round_trips_through_validate(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        task = TaskDraft.from_dict(report.task_draft)
        task.validate()  # must not raise

    def test_plane_payload_has_name(self) -> None:
        provider = StaticLLMProvider(_valid_response(title="Login API"))
        report = run_e2e_dry_run("需求", provider)
        self.assertIn("name", report.plane_payload)

    def test_plane_payload_has_description_html(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        self.assertIn("description_html", report.plane_payload)

    def test_plane_payload_has_state(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        self.assertIn("state", report.plane_payload)

    def test_plane_payload_description_html_contains_dod(self) -> None:
        provider = StaticLLMProvider(_valid_response(dod=["bcrypt hashing", "JWT token"]))
        report = run_e2e_dry_run("需求", provider)
        desc_html = report.plane_payload["description_html"]
        self.assertIn("[DoD]", desc_html)
        self.assertIn("bcrypt hashing", desc_html)
        self.assertIn("JWT token", desc_html)

    def test_plane_endpoint_is_non_empty_string(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        self.assertIsInstance(report.plane_endpoint, str)
        self.assertTrue(report.plane_endpoint.startswith("http"))

    def test_warnings_is_empty_list_in_happy_path(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        self.assertEqual(report.warnings, [])


class TestRunE2EDryRunNoExternalCalls(unittest.TestCase):
    """✅ Guard: run_e2e_dry_run() makes zero external API calls."""

    def test_no_urlopen_calls(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        calls: list[str] = []

        def track_urlopen(*args, **kwargs):
            calls.append("urlopen")
            raise RuntimeError("No network calls allowed in Phase 1D")

        with patch("urllib.request.urlopen", side_effect=track_urlopen):
            report = run_e2e_dry_run("需求", provider)
            # Should succeed even with urlopen patched
            self.assertIsInstance(report, E2EDryRunReport)

        self.assertEqual(calls, [], "No urlopen calls should be made")


# =====================================================================
# run_e2e_dry_run — negative tests
# =====================================================================


class TestRunE2EDryRunNegative(unittest.TestCase):
    """🔴 負面測試: malformed inputs are rejected with clear exceptions."""

    def test_empty_user_request_raises_parser_input_error(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        with self.assertRaises(ParserInputError):
            run_e2e_dry_run("", provider)

    def test_whitespace_only_user_request_raises_parser_input_error(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        with self.assertRaises(ParserInputError):
            run_e2e_dry_run("   \t\n  ", provider)

    def test_malformed_provider_response_raises_parse_error(self) -> None:
        provider = StaticLLMProvider("this is definitely not JSON")
        with self.assertRaises(ParseError):
            run_e2e_dry_run("需求", provider)

    def test_json_array_provider_response_raises_parse_error(self) -> None:
        provider = StaticLLMProvider("[1, 2, 3]")
        with self.assertRaises(ParseError):
            run_e2e_dry_run("需求", provider)

    def test_empty_dict_response_raises_validation_error(self) -> None:
        # {} passes JSON extraction but fails TaskDraft.validate()
        provider = StaticLLMProvider("{}")
        with self.assertRaises(ValidationError):
            run_e2e_dry_run("需求", provider)

    def test_empty_dod_list_raises_validation_error(self) -> None:
        response = json.dumps({"title": "T", "dod": []})
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError):
            run_e2e_dry_run("需求", provider)

    def test_empty_title_raises_validation_error(self) -> None:
        response = _valid_response(title="")
        provider = StaticLLMProvider(response)
        with self.assertRaises(ValidationError):
            run_e2e_dry_run("需求", provider)


# =====================================================================
# run_e2e_dry_run — range / boundary tests
# =====================================================================


class TestRunE2EDryRunRange(unittest.TestCase):
    """📏 範圍測試: field-length limits are respected."""

    def test_long_input_does_not_trigger_external_calls(self) -> None:
        long_text = "建立登入 API " * 500  # ~9500 chars
        provider = StaticLLMProvider(_valid_response())
        calls: list[str] = []

        def track_urlopen(*args, **kwargs):
            calls.append("urlopen")
            raise RuntimeError("No network calls allowed")

        with patch("urllib.request.urlopen", side_effect=track_urlopen):
            report = run_e2e_dry_run(long_text, provider)
            self.assertEqual(report.external_api_calls, 0)

        self.assertEqual(calls, [])


# =====================================================================
# run_e2e_dry_run — correctness tests
# =====================================================================


class TestRunE2EDryRunCorrectness(unittest.TestCase):
    """🎯 正確性測試: output matches Phase 1A/1B contract."""

    def test_task_draft_title_preserved(self) -> None:
        response = _valid_response(title="建立會員登入 API")
        provider = StaticLLMProvider(response)
        report = run_e2e_dry_run("需求", provider)
        self.assertEqual(report.task_draft["title"], "建立會員登入 API")

    def test_task_draft_dod_items_preserved(self) -> None:
        dod = ["bcrypt 密碼驗證", "JWT 24h token", "三種情境測試"]
        response = _valid_response(dod=dod)
        provider = StaticLLMProvider(response)
        report = run_e2e_dry_run("需求", provider)
        self.assertEqual(report.task_draft["dod"], dod)

    def test_task_draft_priority_preserved(self) -> None:
        response = _valid_response(priority="urgent")
        provider = StaticLLMProvider(response)
        report = run_e2e_dry_run("需求", provider)
        self.assertEqual(report.task_draft["priority"], "urgent")

    def test_task_draft_labels_preserved(self) -> None:
        response = _valid_response(labels=["auth", "api"])
        provider = StaticLLMProvider(response)
        report = run_e2e_dry_run("需求", provider)
        self.assertEqual(report.task_draft["labels"], ["auth", "api"])

    def test_plane_payload_name_matches_title(self) -> None:
        response = _valid_response(title="登入 API")
        provider = StaticLLMProvider(response)
        report = run_e2e_dry_run("需求", provider)
        self.assertEqual(report.plane_payload["name"], "登入 API")

    def test_plane_payload_state_from_config(self) -> None:
        provider = StaticLLMProvider(_valid_response())
        report = run_e2e_dry_run("需求", provider)
        # Default dry-run state_id is [dry-run-state]
        self.assertEqual(report.plane_payload["state"], "[dry-run-state]")

    def test_custom_state_id_in_plane_payload(self) -> None:
        from neuralplane.e2e_dry_run import build_dry_run_plane_config
        response = _valid_response()
        provider = StaticLLMProvider(response)
        config = build_dry_run_plane_config(state_id="custom-state-uuid")
        report = run_e2e_dry_run("需求", provider, config)
        self.assertEqual(report.plane_payload["state"], "custom-state-uuid")

    def test_plane_payload_description_html_contains_dod_items(self) -> None:
        dod = ["第一個 criterion", "第二個 criterion"]
        response = _valid_response(dod=dod)
        provider = StaticLLMProvider(response)
        report = run_e2e_dry_run("需求", provider)
        desc_html = report.plane_payload["description_html"]
        self.assertIn("第一個 criterion", desc_html)
        self.assertIn("第二個 criterion", desc_html)
        self.assertIn("[DoD]", desc_html)


# =====================================================================
# Source guard: PlaneClient.create_work_item not called
# =====================================================================


class TestNoLiveWriteSourceGuard(unittest.TestCase):
    """✅ Guard: e2e_dry_run.py does not call create_work_item()."""

    def test_e2e_dry_run_module_no_create_work_item_call(self) -> None:
        module_path = _SRC_ROOT / "neuralplane" / "e2e_dry_run.py"
        source = module_path.read_text()
        self.assertNotIn(
            "create_work_item",
            source,
            "e2e_dry_run.py must not call PlaneClient.create_work_item()",
        )

    def test_cli_e2e_dry_run_module_no_create_work_item_call(self) -> None:
        cli_path = _SRC_ROOT / "neuralplane" / "cli_e2e_dry_run.py"
        source = cli_path.read_text()
        self.assertNotIn(
            "create_work_item",
            source,
            "cli_e2e_dry_run.py must not call PlaneClient.create_work_item()",
        )


# =====================================================================
# CLI — integration tests
# =====================================================================


class TestCliE2EDryRun(unittest.TestCase):
    """Test class: CLI flags, stdout JSON, file output, error exit codes."""

    def setUp(self) -> None:
        self._original_env = dict(os.environ)
        self._repo_root = Path(__file__).resolve().parents[1]
        self._sample_txt = self._repo_root / "examples" / "task_request.sample.txt"

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
            "src.neuralplane.cli_e2e_dry_run",
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

    def test_cli_exits_zero_on_sample_input(self) -> None:
        """🟢 正面測試: CLI with sample input exits 0."""
        result = self._run_cli()
        self.assertEqual(
            result.returncode, 0,
            f"CLI failed: {result.stderr}",
        )

    def test_stdout_is_valid_json_object(self) -> None:
        """🟢 正面測試: stdout is parseable JSON object."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        self.assertIsInstance(parsed, dict)

    def test_stdout_contains_all_required_fields(self) -> None:
        """🟢 正面測試: stdout JSON has all required report fields."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        for field in [
            "dry_run",
            "external_api_calls",
            "task_draft",
            "plane_endpoint",
            "plane_payload",
            "warnings",
        ]:
            self.assertIn(field, parsed, f"Missing field: {field}")

    def test_stdout_dry_run_is_true(self) -> None:
        """🟢 正面測試: report dry_run is always true."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        self.assertIs(parsed["dry_run"], True)

    def test_stdout_external_api_calls_is_zero(self) -> None:
        """🟢 正面測試: report external_api_calls is always 0."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["external_api_calls"], 0)

    def test_stdout_task_draft_validates(self) -> None:
        """🎯 正確性測試: stdout task_draft round-trips through validate()."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        task = TaskDraft.from_dict(parsed["task_draft"])
        task.validate()  # must not raise

    def test_stdout_plane_payload_has_required_fields(self) -> None:
        """🟢 正面測試: plane_payload has name, description_html, state."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        payload = parsed["plane_payload"]
        self.assertIn("name", payload)
        self.assertIn("description_html", payload)
        self.assertIn("state", payload)

    def test_stdout_description_html_contains_dod(self) -> None:
        """🎯 正確性測試: description_html contains [DoD]."""
        result = self._run_cli()
        parsed = json.loads(result.stdout)
        desc_html = parsed["plane_payload"]["description_html"]
        self.assertIn("[DoD]", desc_html)

    def test_output_file_written_with_valid_json(self) -> None:
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
            self.assertIn("dry_run", parsed)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_output_file_and_stdout_are_consistent(self) -> None:
        """🟢 正面測試: output file and stdout contain the same report."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as fh:
            tmp_path = fh.name
        try:
            result = self._run_cli(["--output", tmp_path])
            self.assertEqual(result.returncode, 0)
            stdout_parsed = json.loads(result.stdout)
            file_parsed = json.loads(Path(tmp_path).read_text())
            self.assertEqual(stdout_parsed, file_parsed)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_missing_input_file_exits_nonzero(self) -> None:
        """🔴 負面測試: --input pointing to non-existent file exits != 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_e2e_dry_run",
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
        """🔴 負面測試: --static-response with non-existent file exits != 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_e2e_dry_run",
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

    def test_custom_plane_base_url_reflected_in_endpoint(self) -> None:
        """🟢 正面測試: --plane-base-url appears in plane_endpoint."""
        result = self._run_cli(
            ["--plane-base-url", "https://plane.example.com"]
        )
        self.assertEqual(result.returncode, 0)
        parsed = json.loads(result.stdout)
        self.assertTrue(
            parsed["plane_endpoint"].startswith(
                "https://plane.example.com/"
            ),
            f"Expected prefix 'https://plane.example.com/' "
            f"in '{parsed['plane_endpoint']}'",
        )

    def test_custom_workspace_slug_reflected_in_endpoint(self) -> None:
        """🟢 正面測試: --workspace-slug appears in plane_endpoint."""
        result = self._run_cli(
            ["--workspace-slug", "my-test-workspace"]
        )
        self.assertEqual(result.returncode, 0)
        parsed = json.loads(result.stdout)
        self.assertIn(
            "/workspaces/my-test-workspace/",
            parsed["plane_endpoint"],
        )

    def test_custom_project_id_reflected_in_endpoint(self) -> None:
        """🟢 正面測試: --project-id appears in plane_endpoint."""
        result = self._run_cli(
            ["--project-id", "proj-test-abc"]
        )
        self.assertEqual(result.returncode, 0)
        parsed = json.loads(result.stdout)
        self.assertIn(
            "/projects/proj-test-abc/",
            parsed["plane_endpoint"],
        )

    def test_custom_state_id_reflected_in_plane_payload(self) -> None:
        """🟢 正面測試: --state-id appears in plane_payload state."""
        result = self._run_cli(
            ["--state-id", "custom-state-uuid"]
        )
        self.assertEqual(result.returncode, 0)
        parsed = json.loads(result.stdout)
        self.assertEqual(
            parsed["plane_payload"]["state"],
            "custom-state-uuid",
        )

    def test_cli_succeeds_without_any_plane_env(self) -> None:
        """🔲 邊界測試: CLI succeeds with no Plane env vars set."""
        # Unset all Plane env vars
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("PLANE_")}
        env["PYTHONPATH"] = "src"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_e2e_dry_run",
                "--input",
                str(self._sample_txt),
            ],
            capture_output=True,
            text=True,
            cwd=str(self._repo_root),
            env=env,
        )
        self.assertEqual(
            result.returncode, 0,
            f"CLI failed without Plane env: {result.stderr}",
        )

    def test_cli_malformed_static_response_exits_nonzero(self) -> None:
        """🔴 負面測試: --static-response with malformed JSON exits != 0."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            fh.write("this is definitely not JSON")
            tmp_path = fh.name
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.neuralplane.cli_e2e_dry_run",
                    "--input",
                    str(self._sample_txt),
                    "--static-response",
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                cwd=str(self._repo_root),
                env={**os.environ, "PYTHONPATH": "src"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("parse", result.stderr.lower())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_cli_no_external_api_calls_made(self) -> None:
        """✅ Guard: running the CLI makes zero external API calls."""
        calls: list[str] = []

        def track_urlopen(*args, **kwargs):
            calls.append("urlopen")
            raise RuntimeError("No network calls allowed in Phase 1D")

        with patch("urllib.request.urlopen", side_effect=track_urlopen):
            result = self._run_cli()
            # CLI should succeed even if urlopen is intercepted
            self.assertEqual(
                result.returncode, 0,
                f"CLI failed: {result.stderr}",
            )

        self.assertEqual(
            calls, [],
            "No urlopen calls should be made in Phase 1D",
        )


# =====================================================================
# CLI integration: end-to-end with sample files
# =====================================================================


class TestCliE2EDryRunWithSampleFiles(unittest.TestCase):
    """🟢 正面測試: CLI works with existing sample files."""

    def setUp(self) -> None:
        self._repo_root = Path(__file__).resolve().parents[1]

    def test_cli_with_task_draft_generated_sample_as_static_response(self) -> None:
        """🟢 正面測試: --static-response with task_draft.generated.sample.json."""
        sample_json = self._repo_root / "examples" / "task_draft.generated.sample.json"
        sample_txt = self._repo_root / "examples" / "task_request.sample.txt"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_e2e_dry_run",
                "--input",
                str(sample_txt),
                "--static-response",
                str(sample_json),
            ],
            capture_output=True,
            text=True,
            cwd=str(self._repo_root),
            env={**os.environ, "PYTHONPATH": "src"},
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        parsed = json.loads(result.stdout)
        self.assertIs(parsed["dry_run"], True)
        self.assertEqual(parsed["external_api_calls"], 0)
        self.assertEqual(parsed["task_draft"]["title"], "建立會員登入 API")


# =====================================================================
# Entry-point guard
# =====================================================================

if __name__ == "__main__":
    unittest.main()
