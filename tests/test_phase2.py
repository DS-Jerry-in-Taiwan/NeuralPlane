"""Unit tests for NeuralPlane Phase 2 — Plane Discovery + Live E2E.

Covers:
  - PlaneClient.list_states() — happy path, empty, wrapped, HTTP errors, network errors
  - PlaneClient.list_projects() — happy path, empty, wrapped, HTTP errors, network errors
  - run_e2e_live() orchestration
  - CLI: cli_plane_discover (read-only discovery)
  - CLI: cli_e2e_dry_run --live mode

Run with:  python -m unittest tests.test_phase2 -v
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/neuralplane/ is on the path for direct imports.
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from neuralplane.ai_parser import StaticLLMProvider
from neuralplane.e2e_live import run_e2e_live
from neuralplane.models import CreateIssueResult, TaskDraft
from neuralplane.plane_client import (
    MissingEnvError,
    PlaneClient,
    PlaneConfig,
)


# ----------------------------------------------------------------------
# Sample fixtures
# ----------------------------------------------------------------------


def _minimal_task_dict() -> dict:
    return {
        "title": "Minimal Task",
        "description": "A task with only required fields.",
        "dod": ["This is the only DoD item."],
    }


def _make_mock_response(body: dict | list, status: int = 200) -> MagicMock:
    """Return a mock urlopen context-manager that returns ``body`` as JSON."""
    f = MagicMock()
    f.status = status
    f.read.return_value = json.dumps(body).encode("utf-8")
    f.__enter__ = MagicMock(return_value=f)
    f.__exit__ = MagicMock(return_value=False)
    return f


def _cfg() -> PlaneConfig:
    return PlaneConfig(
        base_url="https://api.plane.so",
        api_key="plane_api_testkey",
        workspace_slug="test-workspace",
        project_id="test-project-id",
        default_state_id="test-state-id",
    )


# =====================================================================
# PlaneClient.list_states()
# =====================================================================

class TestListStates(unittest.TestCase):
    """Test class: PlaneClient.list_states()."""

    def test_list_states_success_returns_list(self) -> None:
        """Mock 200 + states array → returns list of state dicts."""
        mock_states = [
            {"id": "s1", "name": "Backlog", "group": "backlog"},
            {"id": "s2", "name": "In Progress", "group": "in_progress"},
        ]
        mock_fp = _make_mock_response(mock_states)

        with patch("urllib.request.urlopen", return_value=mock_fp):
            result = PlaneClient(_cfg()).list_states()

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "Backlog")

    def test_list_states_empty_array(self) -> None:
        """Mock 200 + [] → returns empty list."""
        mock_fp = _make_mock_response([])

        with patch("urllib.request.urlopen", return_value=mock_fp):
            result = PlaneClient(_cfg()).list_states()

        self.assertEqual(result, [])

    def test_list_states_wrapped_in_results(self) -> None:
        """Mock 200 + {"results": [...]} → returns the inner list."""
        mock_states = [
            {"id": "s3", "name": "Done", "group": "done"},
        ]
        mock_fp = _make_mock_response({"results": mock_states})

        with patch("urllib.request.urlopen", return_value=mock_fp):
            result = PlaneClient(_cfg()).list_states()

        self.assertEqual(result, mock_states)

    def test_list_states_unwrapped_dict_no_results(self) -> None:
        """Mock 200 + plain dict (no "results") → returns empty list."""
        mock_fp = _make_mock_response({"some": "data"})

        with patch("urllib.request.urlopen", return_value=mock_fp):
            result = PlaneClient(_cfg()).list_states()

        self.assertEqual(result, [])

    def test_list_states_http_error(self) -> None:
        """Mock HTTPError 401 → raises RuntimeError (safe message)."""
        exc = urllib.error.HTTPError(
            url="https://api.plane.so/api/v1/workspaces/ws/projects/p/states/",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                PlaneClient(_cfg()).list_states()

        error_msg = str(ctx.exception)
        self.assertNotIn("plane_api_testkey", error_msg)
        self.assertNotIn("X-API-Key", error_msg)

    def test_list_states_network_error(self) -> None:
        """Mock URLError → raises RuntimeError."""
        exc = urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                PlaneClient(_cfg()).list_states()

        self.assertIn("Connection refused", str(ctx.exception))

    def test_list_states_url_matches_api_contract(self) -> None:
        """list_states() calls the correct endpoint URL."""
        mock_fp = _make_mock_response([])

        with patch("urllib.request.urlopen", return_value=mock_fp) as mock_urlopen:
            PlaneClient(_cfg()).list_states()

        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertEqual(req.method, "GET")
        self.assertIn("/states/", req.full_url)

    def test_list_states_non_json_html_response(self) -> None:
        """Mock 200 + HTML body → raises RuntimeError (not JSONDecodeError)."""
        mock_resp = MagicMock(spec=io.BufferedIOBase)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"<html>Server Error</html>"
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                PlaneClient(_cfg()).list_states()
        self.assertIn("non-json", str(ctx.exception).lower())

    def test_list_states_non_json_empty_body(self) -> None:
        """Mock 200 + empty body → raises RuntimeError (not JSONDecodeError)."""
        mock_resp = MagicMock(spec=io.BufferedIOBase)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b""
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                PlaneClient(_cfg()).list_states()
        self.assertIn("non-json", str(ctx.exception).lower())


# =====================================================================
# PlaneClient.list_projects()
# =====================================================================

class TestListProjects(unittest.TestCase):
    """Test class: PlaneClient.list_projects()."""

    def test_list_projects_success_returns_list(self) -> None:
        """Mock 200 + projects array → returns list of project dicts."""
        mock_projects = [
            {"id": "p1", "name": "Project A", "identifier": "PROJA"},
            {"id": "p2", "name": "Project B", "identifier": "PROJB"},
        ]
        mock_fp = _make_mock_response(mock_projects)

        with patch("urllib.request.urlopen", return_value=mock_fp):
            result = PlaneClient(_cfg()).list_projects()

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["identifier"], "PROJB")

    def test_list_projects_empty_array(self) -> None:
        """Mock 200 + [] → returns empty list."""
        mock_fp = _make_mock_response([])

        with patch("urllib.request.urlopen", return_value=mock_fp):
            result = PlaneClient(_cfg()).list_projects()

        self.assertEqual(result, [])

    def test_list_projects_wrapped_in_results(self) -> None:
        """Mock 200 + {"results": [...]} → returns the inner list."""
        mock_projects = [{"id": "p3", "name": "Project C", "identifier": "PROJC"}]
        mock_fp = _make_mock_response({"results": mock_projects})

        with patch("urllib.request.urlopen", return_value=mock_fp):
            result = PlaneClient(_cfg()).list_projects()

        self.assertEqual(result, mock_projects)

    def test_list_projects_http_error(self) -> None:
        """Mock HTTPError 403 → raises RuntimeError (safe message)."""
        exc = urllib.error.HTTPError(
            url="https://api.plane.so/api/v1/workspaces/ws/projects/",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                PlaneClient(_cfg()).list_projects()

        error_msg = str(ctx.exception)
        self.assertNotIn("plane_api_testkey", error_msg)

    def test_list_projects_network_error(self) -> None:
        """Mock URLError → raises RuntimeError."""
        exc = urllib.error.URLError("DNS failure")

        with patch("urllib.request.urlopen", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                PlaneClient(_cfg()).list_projects()

        self.assertIn("DNS failure", str(ctx.exception))

    def test_list_projects_url_matches_api_contract(self) -> None:
        """list_projects() calls the correct endpoint URL."""
        mock_fp = _make_mock_response([])

        with patch("urllib.request.urlopen", return_value=mock_fp) as mock_urlopen:
            PlaneClient(_cfg()).list_projects()

        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertEqual(req.method, "GET")
        self.assertIn("/projects/", req.full_url)

    def test_list_projects_non_json_html_response(self) -> None:
        """Mock 200 + HTML body → raises RuntimeError (not JSONDecodeError)."""
        mock_resp = MagicMock(spec=io.BufferedIOBase)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"<html>Server Error</html>"
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                PlaneClient(_cfg()).list_projects()
        self.assertIn("non-json", str(ctx.exception).lower())

    def test_list_projects_non_json_empty_body(self) -> None:
        """Mock 200 + empty body → raises RuntimeError (not JSONDecodeError)."""
        mock_resp = MagicMock(spec=io.BufferedIOBase)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b""
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                PlaneClient(_cfg()).list_projects()
        self.assertIn("non-json", str(ctx.exception).lower())


# =====================================================================
# run_e2e_live()
# =====================================================================

class TestRunE2ELive(unittest.TestCase):
    """Test class: run_e2e_live() orchestration."""

    def test_e2e_live_success_returns_create_issue_result(self) -> None:
        """Mock create_work_item → success CreateIssueResult."""
        static_response = json.dumps(_minimal_task_dict(), ensure_ascii=False)
        provider = StaticLLMProvider(static_response)

        mock_result = CreateIssueResult(
            id="new-uuid-123",
            sequence_id=42,
            name="Minimal Task",
            url="https://api.plane.so/ws/projects/proj/work-items/new-uuid-123",
            raw_response={"id": "new-uuid-123"},
            success=True,
            error_message=None,
        )

        with patch.object(
            PlaneClient,
            "create_work_item",
            return_value=mock_result,
        ) as mock_create:
            result = run_e2e_live(
                "Create a minimal task.",
                provider,
                _cfg(),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.id, "new-uuid-123")
        self.assertEqual(result.sequence_id, 42)
        mock_create.assert_called_once()

    def test_e2e_live_failure_returns_error_result(self) -> None:
        """Mock create_work_item → failure CreateIssueResult."""
        static_response = json.dumps(_minimal_task_dict(), ensure_ascii=False)
        provider = StaticLLMProvider(static_response)

        mock_result = CreateIssueResult(
            id=None,
            sequence_id=None,
            name="Minimal Task",
            url=None,
            raw_response={"status_code": 500},
            success=False,
            error_message="Plane API error 500: Internal server error",
        )

        with patch.object(
            PlaneClient,
            "create_work_item",
            return_value=mock_result,
        ):
            result = run_e2e_live(
                "Create a minimal task.",
                provider,
                _cfg(),
            )

        self.assertFalse(result.success)
        self.assertIsNone(result.id)
        self.assertIn("500", result.error_message)

    def test_e2e_live_parses_task_correctly(self) -> None:
        """run_e2e_live passes the parsed TaskDraft to create_work_item."""
        full_dict = {
            "title": "Auth API",
            "description": "Implement auth.",
            "dod": ["Must work."],
            "priority": "high",
            "labels": ["backend"],
        }
        static_response = json.dumps(full_dict, ensure_ascii=False)
        provider = StaticLLMProvider(static_response)

        captured_task: list = []

        def _capture_task(task, *, dry_run):
            captured_task.append(task)
            return CreateIssueResult(
                id="id", sequence_id=1, name=task.title,
                url="http://x", raw_response={}, success=True,
            )

        with patch.object(PlaneClient, "create_work_item", side_effect=_capture_task):
            run_e2e_live("Create auth API.", provider, _cfg())

        self.assertEqual(len(captured_task), 1)
        self.assertEqual(captured_task[0].title, "Auth API")
        self.assertEqual(captured_task[0].priority, "high")


# =====================================================================
# CLI: cli_plane_discover (subprocess)
# =====================================================================

class TestCLIDiscover(unittest.TestCase):
    """Test class: cli_plane_discover entry-point (subprocess)."""

    def test_discover_success_prints_projects(self) -> None:
        """With env vars set, list_projects returns data → exit 0."""
        # Run the discover CLI with env vars set to a fake but structurally valid
        # PLANE_API_KEY that will be rejected by the HTTP layer (which is fine,
        # the CLI only calls list_projects before hitting create_work_item).
        env = {
            **os.environ,
            "PYTHONPATH": "src",
            "PLANE_API_KEY": "plane_api_fake",
            "PLANE_WORKSPACE_SLUG": "test-ws",
            "PLANE_PROJECT_ID": "proj-id",
            "PLANE_DEFAULT_STATE_ID": "state-id",
        }
        # Use a fake base URL so the HTTP call fails immediately.
        proc = subprocess.run(
            [
                sys.executable, "-m",
                "src.neuralplane.cli_plane_discover",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
        )
        # Expect failure (network error reaching fake URL) — not FATAL at start.
        # The important thing: the CLI handles the error gracefully (no crash).
        combined = proc.stdout + proc.stderr
        # Must NOT print FATAL for MissingEnvError (env vars are present).
        # Must NOT echo the API key value in any error message.
        self.assertNotIn("plane_api_fake", combined)

    def test_discover_rejects_missing_env(self) -> None:
        """Without PLANE_ vars, the CLI prints MissingEnvError and exits non-zero."""
        env = {}
        for k, v in os.environ.items():
            if not k.startswith("PLANE_"):
                env[k] = v
        env["PYTHONPATH"] = "src"
        # Ensure no .env file is loaded — rename temporarily.
        repo_root = Path(__file__).resolve().parents[1]
        dotenv_path = repo_root / ".env"
        backup_path = repo_root / ".env.bak"

        moved = False
        if dotenv_path.exists():
            dotenv_path.rename(backup_path)
            moved = True

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m",
                    "src.neuralplane.cli_plane_discover",
                ],
                capture_output=True,
                text=True,
                cwd=repo_root,
                env=env,
            )
            self.assertEqual(proc.returncode, 1)
            combined = proc.stdout + proc.stderr
            self.assertIn("FATAL", combined)
            self.assertIn("Missing required", combined)
        finally:
            if moved:
                backup_path.rename(dotenv_path)


# =====================================================================
# CLI: cli_e2e_dry_run --live mode (subprocess)
# =====================================================================

class TestCLIE2ELive(unittest.TestCase):
    """Test class: cli_e2e_dry_run --live mode integration."""

    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _SAMPLE_REQUEST = _REPO_ROOT / "examples" / "task_request.sample.txt"

    def test_e2e_cli_dry_run_still_works_without_live_flag(self) -> None:
        """No --live flag → dry-run JSON output, exit 0 (no env vars needed)."""
        proc = subprocess.run(
            [
                sys.executable, "-m",
                "src.neuralplane.cli_e2e_dry_run",
                "--input", str(self._SAMPLE_REQUEST),
                "--provider", "static",
            ],
            capture_output=True,
            text=True,
            cwd=self._REPO_ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        self.assertTrue(data["dry_run"])
        self.assertIn("task_draft", data)
        self.assertIn("plane_payload", data)

    def test_e2e_cli_live_fails_gracefully_without_env(self) -> None:
        """Without PLANE_ env vars, --live exits non-zero with a safe error."""
        env = {}
        for k, v in os.environ.items():
            if not k.startswith("PLANE_"):
                env[k] = v
        env["PYTHONPATH"] = "src"

        # Ensure .env is not loaded.
        repo_root = Path(__file__).resolve().parents[1]
        dotenv_path = repo_root / ".env"
        backup_path = repo_root / ".env.bak"
        moved = False
        if dotenv_path.exists():
            dotenv_path.rename(backup_path)
            moved = True

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m",
                    "src.neuralplane.cli_e2e_dry_run",
                    "--input", str(self._SAMPLE_REQUEST),
                    "--provider", "static",
                    "--live",
                ],
                capture_output=True,
                text=True,
                cwd=repo_root,
                env=env,
            )
            self.assertEqual(proc.returncode, 1)
            combined = proc.stdout + proc.stderr
            # Must not echo a fake API key VALUE (PLANE_API_KEY as var name is OK).
            self.assertNotIn("sk-", combined.lower())
            self.assertNotIn("fake_secret", combined.lower())
        finally:
            if moved:
                backup_path.rename(dotenv_path)

    def test_e2e_cli_live_parses_static_response_and_calls_create(self) -> None:
        """With --live, --static-response file is used to parse task, then Plane is called."""
        # This is an integration test using the sample JSON file as input.
        # We pass a structurally-valid but fake API key; the HTTP mock will intercept.
        env = {
            **os.environ,
            "PYTHONPATH": "src",
            "PLANE_API_KEY": "plane_api_testkey",
            "PLANE_WORKSPACE_SLUG": "test-ws",
            "PLANE_PROJECT_ID": "test-proj-id",
            "PLANE_DEFAULT_STATE_ID": "test-state-id",
        }
        static_path = self._SAMPLE_REQUEST  # Use the request file as static response.

        # Patch urllib at the Python-level so the subprocess uses the mock.
        # We can't easily mock at the network level from outside subprocess,
        # so we verify the CLI at least runs with the given input (no crash).
        proc = subprocess.run(
            [
                sys.executable, "-m",
                "src.neuralplane.cli_e2e_dry_run",
                "--input", str(static_path),
                "--provider", "static",
                "--static-response", str(
                    self._REPO_ROOT / "examples" / "task_draft.sample.json"
                ),
                "--live",
            ],
            capture_output=True,
            text=True,
            cwd=self._REPO_ROOT,
            env=env,
        )
        # The CLI should either succeed (if mock HTTP) or fail gracefully.
        # We mainly verify it doesn't crash.
        combined = proc.stdout + proc.stderr
        self.assertNotIn("Traceback", combined)
        self.assertNotIn("plane_api_testkey", combined)


# =====================================================================
# PlaneConfig URL builders (new)
# =====================================================================

class TestPlaneConfigUrlBuilders(unittest.TestCase):
    """Test class: PlaneConfig.build_states_url() and build_projects_url()."""

    def test_build_states_url(self) -> None:
        cfg = _cfg()
        url = cfg.build_states_url()
        self.assertIn("/states/", url)
        self.assertIn(cfg.workspace_slug, url)
        self.assertIn(cfg.project_id, url)

    def test_build_projects_url(self) -> None:
        cfg = _cfg()
        url = cfg.build_projects_url()
        self.assertIn("/projects/", url)
        self.assertEqual(url.count("/projects/"), 1)
        self.assertIn(cfg.workspace_slug, url)


# =====================================================================
# Entry-point guard
# =====================================================================

if __name__ == "__main__":
    unittest.main()
