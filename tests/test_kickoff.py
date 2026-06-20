"""Unit tests for NeuralPlane Development Kickoff feature.

Covers:
  - generate_branch_name (positive, negative, boundary)
  - WorkContextResult dataclass (positive, correctness)
  - GitHubConfig.from_env (positive, missing env)
  - GitHubClient mock success paths
  - GitHubClient error paths (401/403, 422, network)
  - run_kickoff dry-run
  - run_kickoff full success (mock all APIs)
  - run_kickoff no-branch mode
  - run_kickoff Plane failure (GitHub not called)
  - overall_success logic

Run with:  python -m unittest discover -s tests
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
from neuralplane.github_client import (
    GitHubClient,
    GitHubConfig,
    MissingEnvError,
)
from neuralplane.kickoff import _build_issue_key, run_kickoff
from neuralplane.models import (
    CreateIssueResult,
    WorkContextResult,
    generate_branch_name,
)
from neuralplane.plane_client import PlaneConfig


# ----------------------------------------------------------------------
# Sample fixtures
# ----------------------------------------------------------------------

_DEFAULT_LLM_RESPONSE = json.dumps(
    {
        "title": "建立會員登入 API",
        "description": "建立會員登入端點，支援 bcrypt 密碼驗證與 JWT access token 簽發。",
        "dod": [
            "密碼驗證必須使用 bcrypt hash，不可明文比對。",
            "登入成功後必須簽發有效期限為 24 小時的 JWT access token。",
            "必須包含登入成功、密碼錯誤、使用者不存在三種情境的測試。",
        ],
        "priority": "high",
        "labels": ["auth", "api"],
        "assignee_suggestions": [],
    },
    ensure_ascii=False,
)


def _dry_run_plane_config() -> PlaneConfig:
    return PlaneConfig(
        base_url="https://api.plane.so",
        api_key="[dry-run-no-key]",
        workspace_slug="[dry-run-workspace]",
        project_id="[dry-run-project]",
        default_state_id="[dry-run-state]",
    )


def _dry_run_github_config() -> GitHubConfig:
    return GitHubConfig(
        token="[dry-run-no-token]",
        repo="owner/repo",
    )


def _mock_sha_response() -> dict:
    return {
        "ref": "refs/heads/main",
        "node_id": "RC_kwDOBHe7",
        "url": "https://api.github.com/repos/owner/repo/git/refs/heads/main",
        "object": {
            "sha": "abc123def456",
            "type": "commit",
            "url": "https://api.github.com/repos/owner/repo/git/commit/abc123def456",
        },
    }


def _mock_branch_created_response() -> dict:
    return {
        "ref": "refs/heads/np-123-login-api",
        "node_id": "RC_kwDOBHe8",
        "url": "https://api.github.com/repos/owner/repo/git/refs/heads/np-123-login-api",
        "object": {
            "sha": "abc123def456",
            "type": "commit",
            "url": "https://api.github.com/repos/owner/repo/git/commit/abc123def456",
        },
    }


# =====================================================================
# generate_branch_name — Positive Tests
# =====================================================================


class TestGenerateBranchNamePositive(unittest.TestCase):
    """🟢 正面測試: basic functionality."""

    def test_simple_english_title(self) -> None:
        result = generate_branch_name("NP-123", "Implement login API")
        self.assertEqual(result, "np-123-implement-login-api")

    def test_chinese_title(self) -> None:
        # Chinese characters are each matched by [^a-z0-9]+ as a group,
        # so "建立會員登入 API" → "api" (each non-alphanumeric run → single hyphen)
        result = generate_branch_name("NP-123", "建立會員登入 API")
        self.assertEqual(result, "np-123-api")

    def test_mixed_alphanumeric(self) -> None:
        result = generate_branch_name("NP-456", "Fix bug 1234 and enhance UX")
        self.assertEqual(result, "np-456-fix-bug-1234-and-enhance-ux")

    def test_issue_key_lowercased(self) -> None:
        result = generate_branch_name("NP-789", "Add feature X")
        self.assertEqual(result, "np-789-add-feature-x")


class TestGenerateBranchNameEdge(unittest.TestCase):
    """🔲 邊界測試: slug truncation and special characters."""

    def test_slug_truncated_to_30_chars(self) -> None:
        # The slug (without the issue key prefix) should be at most 30 chars.
        # For issue_key "NP-1", the branch name is "np-1-{slug}" where
        # slug itself is truncated to 30 chars.
        long_title = "This is a very long title that exceeds thirty characters"
        result = generate_branch_name("NP-1", long_title)
        # Verify it starts with the issue key prefix
        self.assertTrue(result.startswith("np-1-"))
        # The slug part (everything after "np-1-") should be <= 30 chars
        slug = result[len("np-1-"):]
        self.assertLessEqual(len(slug), 30)

    def test_special_chars_converted_to_hyphens(self) -> None:
        result = generate_branch_name("NP-1", "Login API (v2) [beta]")
        # Parentheses and brackets become hyphens
        parts = result.split("-")
        for part in parts:
            self.assertTrue(
                part.isalnum(),
                f"Expected alphanumeric, got '{part}'",
            )


class TestGenerateBranchNameNegative(unittest.TestCase):
    """🔴 負面測試: empty string handling."""

    def test_empty_title_uses_only_issue_key(self) -> None:
        result = generate_branch_name("NP-123", "")
        # Empty slug stripped → just issue key with trailing dash
        self.assertTrue(result.startswith("np-123-"))
        self.assertTrue(result.endswith("-"))


# =====================================================================
# WorkContextResult — Positive Tests
# =====================================================================


class TestWorkContextResultPositive(unittest.TestCase):
    """🟢 正面測試: WorkContextResult field correctness."""

    def test_all_fields_set(self) -> None:
        result = WorkContextResult(
            plane_success=True,
            plane_issue_key="NP-123",
            plane_issue_url="https://plane.so/project/work-items/abc",
            plane_issue_id="abc",
            plane_error=None,
            github_success=True,
            github_branch_name="np-123-login-api",
            github_repo="owner/repo",
            github_error=None,
            overall_success=True,
            checkout_instruction="git checkout np-123-login-api",
        )
        self.assertTrue(result.plane_success)
        self.assertEqual(result.plane_issue_key, "NP-123")
        self.assertEqual(result.github_branch_name, "np-123-login-api")
        self.assertTrue(result.overall_success)
        self.assertEqual(result.checkout_instruction, "git checkout np-123-login-api")

    def test_defaults_plane_failure(self) -> None:
        result = WorkContextResult(plane_success=False)
        self.assertFalse(result.plane_success)
        self.assertFalse(result.github_success)
        self.assertFalse(result.overall_success)
        self.assertIsNone(result.plane_issue_key)
        self.assertIsNone(result.github_branch_name)
        self.assertIsNone(result.checkout_instruction)

    def test_github_failure_partial_success(self) -> None:
        result = WorkContextResult(
            plane_success=True,
            plane_issue_key="NP-456",
            github_success=False,
            github_error="Branch already exists",
            overall_success=False,
        )
        self.assertTrue(result.plane_success)
        self.assertFalse(result.github_success)
        self.assertFalse(result.overall_success)
        self.assertEqual(result.github_error, "Branch already exists")


# =====================================================================
# overall_success — Correctness Test
# =====================================================================


class TestOverallSuccessLogic(unittest.TestCase):
    """🎯 正確性測試: overall_success is True only when both succeed."""

    def test_true_when_both_succeed(self) -> None:
        result = WorkContextResult(
            plane_success=True,
            github_success=True,
            overall_success=True,
        )
        self.assertTrue(result.overall_success)

    def test_false_when_plane_fails(self) -> None:
        result = WorkContextResult(
            plane_success=False,
            github_success=True,
            overall_success=False,
        )
        self.assertFalse(result.overall_success)

    def test_false_when_github_fails(self) -> None:
        result = WorkContextResult(
            plane_success=True,
            github_success=False,
            overall_success=False,
        )
        self.assertFalse(result.overall_success)

    def test_false_when_both_fail(self) -> None:
        result = WorkContextResult(
            plane_success=False,
            github_success=False,
            overall_success=False,
        )
        self.assertFalse(result.overall_success)


# =====================================================================
# GitHubConfig — Positive Tests
# =====================================================================


class TestGitHubConfigFromEnv(unittest.TestCase):
    """🟢 正面測試: GitHubConfig from_env."""

    def setUp(self) -> None:
        # Ensure clean slate
        self._saved = {}
        for key in ("GITHUB_TOKEN", "GITHUB_REPO"):
            self._saved[key] = os.environ.get(key)
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, val in self._saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_happy_path(self) -> None:
        os.environ["GITHUB_TOKEN"] = "ghp_testtoken123"
        os.environ["GITHUB_REPO"] = "owner/repo"
        config = GitHubConfig.from_env()
        self.assertEqual(config.token, "ghp_testtoken123")
        self.assertEqual(config.repo, "owner/repo")
        self.assertEqual(config.api_base, "https://api.github.com")

    def test_custom_api_base_via_direct_init(self) -> None:
        # GitHubConfig does not read GITHUB_API_BASE from env (only GITHUB_TOKEN
        # and GITHUB_REPO). Test construction via direct init instead.
        config = GitHubConfig(
            token="token",
            repo="owner/repo",
            api_base="https://github.mycompany.com/api/v3",
        )
        self.assertEqual(config.api_base, "https://github.mycompany.com/api/v3")


# =====================================================================
# GitHubConfig — Missing Env Tests
# =====================================================================


class TestGitHubConfigMissingEnv(unittest.TestCase):
    """🔲 邊界測試: Missing environment variables."""

    def setUp(self) -> None:
        # Ensure clean slate: save and clear all relevant env vars
        self._saved = {}
        for key in ("GITHUB_TOKEN", "GITHUB_REPO"):
            self._saved[key] = os.environ.get(key)
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        # Restore
        for key, val in self._saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_missing_token(self) -> None:
        os.environ["GITHUB_REPO"] = "owner/repo"
        with self.assertRaises(MissingEnvError) as ctx:
            GitHubConfig.from_env()
        self.assertIn("GITHUB_TOKEN", ctx.exception.missing)

    def test_missing_repo(self) -> None:
        os.environ["GITHUB_TOKEN"] = "token"
        with self.assertRaises(MissingEnvError) as ctx:
            GitHubConfig.from_env()
        self.assertIn("GITHUB_REPO", ctx.exception.missing)

    def test_missing_both(self) -> None:
        with self.assertRaises(MissingEnvError) as ctx:
            GitHubConfig.from_env()
        self.assertIn("GITHUB_TOKEN", ctx.exception.missing)
        self.assertIn("GITHUB_REPO", ctx.exception.missing)

    def test_whitespace_only_token(self) -> None:
        os.environ["GITHUB_TOKEN"] = "   "
        os.environ["GITHUB_REPO"] = "owner/repo"
        with self.assertRaises(MissingEnvError) as ctx:
            GitHubConfig.from_env()
        self.assertIn("GITHUB_TOKEN", ctx.exception.missing)


# =====================================================================
# GitHubClient — Positive Tests (mocked)
# =====================================================================


class TestGitHubClientSuccess(unittest.TestCase):
    """🟢 正面測試: GitHubClient mock success paths."""

    def setUp(self) -> None:
        self._orig_environ = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig_environ)

    @patch("urllib.request.urlopen")
    def test_get_branch_head_sha(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                status=200,
                read=MagicMock(
                    return_value=json.dumps(_mock_sha_response()).encode()
                ),
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = GitHubConfig(token="tok", repo="owner/repo")
        client = GitHubClient(config)
        sha = client.get_branch_head_sha("main")
        self.assertEqual(sha, "abc123def456")

    @patch("urllib.request.urlopen")
    def test_create_branch_success(self, mock_urlopen: MagicMock) -> None:
        # First call: get_sha; second call: create_branch
        mock_responses = [
            json.dumps(_mock_sha_response()).encode(),
            json.dumps(_mock_branch_created_response()).encode(),
        ]
        mock_read = iter(mock_responses)

        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                status=200,
                read=MagicMock(side_effect=lambda: next(mock_read)),
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = GitHubConfig(token="tok", repo="owner/repo")
        client = GitHubClient(config)
        result = client.create_branch("np-123-login-api", "main")
        self.assertEqual(result, "np-123-login-api")


# =====================================================================
# GitHubClient — Error Tests
# =====================================================================


class TestGitHubClientErrors(unittest.TestCase):
    """🔴 負面測試: GitHubClient error paths."""

    def setUp(self) -> None:
        self._orig_environ = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig_environ)

    @patch.object(GitHubClient, "get_branch_head_sha")
    @patch("urllib.request.urlopen")
    def test_create_branch_422_already_exists(
        self, mock_urlopen: MagicMock, mock_get_sha: MagicMock
    ) -> None:
        # get_sha succeeds; create_branch POST raises 422
        mock_get_sha.return_value = "abc123def456"

        error_body = json.dumps(
            {"message": "Branch 'np-123-login-api' already exists."}
        ).encode()

        def raise_422(*args, **kwargs):
            exc = urllib.error.HTTPError(
                url="http://test",
                code=422,
                msg="Unprocessable Entity",
                hdrs={},
                fp=io.BytesIO(error_body),
            )
            raise exc

        mock_urlopen.return_value.__enter__ = MagicMock(side_effect=raise_422)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = GitHubConfig(token="tok", repo="owner/repo")
        client = GitHubClient(config)
        with self.assertRaises(RuntimeError) as ctx:
            client.create_branch("np-123-login-api", "main")
        self.assertIn("already exists", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_create_branch_401_unauthorized(self, mock_urlopen: MagicMock) -> None:
        error_body = json.dumps(
            {"message": "Bad credentials"}
        ).encode()

        def raise_http_error(*args, **kwargs):
            exc = urllib.error.HTTPError(
                url="http://test",
                code=401,
                msg="Unauthorized",
                hdrs={},
                fp=None,
            )
            exc.read = MagicMock(return_value=error_body)
            raise exc

        mock_urlopen.side_effect = raise_http_error

        config = GitHubConfig(token="bad-token", repo="owner/repo")
        client = GitHubClient(config)
        with self.assertRaises(RuntimeError) as ctx:
            client.create_branch("new-branch", "main")
        # Must not leak the token
        self.assertNotIn("bad-token", str(ctx.exception))
        self.assertNotIn("tok", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_create_branch_network_error(self, mock_urlopen: MagicMock) -> None:
        def raise_url_error(*args, **kwargs):
            raise urllib.error.URLError("Connection refused")

        mock_urlopen.side_effect = raise_url_error

        config = GitHubConfig(token="tok", repo="owner/repo")
        client = GitHubClient(config)
        with self.assertRaises(RuntimeError) as ctx:
            client.create_branch("new-branch", "main")
        self.assertIn("Network error", str(ctx.exception))


# =====================================================================
# run_kickoff — Dry-run Tests
# =====================================================================


class TestRunKickoffDryRun(unittest.TestCase):
    """🟢 正面測試: run_kickoff dry-run mode."""

    def test_dry_run_with_github_config(self) -> None:
        provider = StaticLLMProvider(_DEFAULT_LLM_RESPONSE)
        result = run_kickoff(
            user_request="Create login API",
            llm_provider=provider,
            plane_config=_dry_run_plane_config(),
            github_config=_dry_run_github_config(),
            dry_run=True,
        )
        self.assertTrue(result.plane_success)
        self.assertTrue(result.github_success)
        self.assertEqual(result.plane_issue_key, "[dry-run]")
        self.assertEqual(result.github_branch_name, "[dry-run-branch]")
        self.assertTrue(result.overall_success)

    def test_dry_run_without_github_config(self) -> None:
        provider = StaticLLMProvider(_DEFAULT_LLM_RESPONSE)
        result = run_kickoff(
            user_request="Create login API",
            llm_provider=provider,
            plane_config=_dry_run_plane_config(),
            github_config=None,
            dry_run=True,
        )
        self.assertTrue(result.plane_success)
        self.assertFalse(result.github_success)
        self.assertIsNone(result.github_branch_name)
        self.assertIsNone(result.checkout_instruction)


# =====================================================================
# run_kickoff — Full Success Tests (mocked)
# =====================================================================


class TestRunKickoffFullSuccess(unittest.TestCase):
    """🟢 正面測試: run_kickoff full success with mocked APIs."""

    def setUp(self) -> None:
        self._orig_environ = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig_environ)

    @patch("urllib.request.urlopen")
    def test_full_success_mock_all(self, mock_urlopen: MagicMock) -> None:
        # Mock Plane API response
        plane_response = {
            "id": "plane-uuid-123",
            "name": "建立會員登入 API",
            "sequence_id": 456,
            "state": "backlog",
        }
        # Mock GitHub get_sha and create_branch
        mock_responses = [
            json.dumps(plane_response).encode(),  # Plane POST
            json.dumps(_mock_sha_response()).encode(),  # GitHub get_sha
            json.dumps(_mock_branch_created_response()).encode(),  # GitHub create
        ]
        mock_read = iter(mock_responses)

        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                status=200,
                read=MagicMock(side_effect=lambda: next(mock_read)),
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        provider = StaticLLMProvider(_DEFAULT_LLM_RESPONSE)
        result = run_kickoff(
            user_request="Create login API",
            llm_provider=provider,
            plane_config=_dry_run_plane_config(),
            github_config=_dry_run_github_config(),
            dry_run=False,
        )
        self.assertTrue(result.plane_success)
        self.assertTrue(result.github_success)
        self.assertTrue(result.overall_success)
        self.assertIsNotNone(result.checkout_instruction)
        self.assertIn("git checkout", result.checkout_instruction)


# =====================================================================
# run_kickoff — No-branch Mode
# =====================================================================


class TestRunKickoffNoBranch(unittest.TestCase):
    """🟢 正面測試: run_kickoff with --no-branch mode."""

    def setUp(self) -> None:
        self._orig_environ = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig_environ)

    @patch("urllib.request.urlopen")
    def test_no_branch_skips_github(self, mock_urlopen: MagicMock) -> None:
        # Only Plane POST should be called
        plane_response = {
            "id": "plane-uuid-123",
            "name": "Test Task",
            "sequence_id": 789,
        }
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                status=200,
                read=MagicMock(
                    return_value=json.dumps(plane_response).encode()
                ),
            )
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        provider = StaticLLMProvider(_DEFAULT_LLM_RESPONSE)
        result = run_kickoff(
            user_request="Test task",
            llm_provider=provider,
            plane_config=_dry_run_plane_config(),
            github_config=None,  # no branch
            dry_run=False,
        )
        self.assertTrue(result.plane_success)
        self.assertFalse(result.github_success)
        self.assertIsNone(result.github_branch_name)
        self.assertFalse(result.overall_success)  # GitHub was not attempted


# =====================================================================
# run_kickoff — Plane Failure (GitHub not called)
# =====================================================================


class TestRunKickoffPlaneFailure(unittest.TestCase):
    """🔴 負面測試: run_kickoff when Plane fails — GitHub should not be called."""

    def setUp(self) -> None:
        self._orig_environ = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig_environ)

    @patch("urllib.request.urlopen")
    def test_plane_fails_github_not_called(self, mock_urlopen: MagicMock) -> None:
        def raise_http_error(*args, **kwargs):
            exc = urllib.error.HTTPError(
                url="http://test",
                code=400,
                msg="Bad Request",
                hdrs={},
                fp=None,
            )
            exc.read = MagicMock(
                return_value=json.dumps(
                    {"message": "Invalid payload"}
                ).encode()
            )
            raise exc

        mock_urlopen.side_effect = raise_http_error

        provider = StaticLLMProvider(_DEFAULT_LLM_RESPONSE)
        result = run_kickoff(
            user_request="Create login API",
            llm_provider=provider,
            plane_config=_dry_run_plane_config(),
            github_config=_dry_run_github_config(),
            dry_run=False,
        )
        self.assertFalse(result.plane_success)
        self.assertFalse(result.github_success)
        self.assertFalse(result.overall_success)
        self.assertIsNotNone(result.plane_error)
        # GitHub should not have been attempted
        self.assertIsNone(result.github_error)


# =====================================================================
# _build_issue_key — Tests
# =====================================================================


class TestBuildIssueKey(unittest.TestCase):
    """🟢 正面測試: _build_issue_key."""

    def test_with_sequence_id(self) -> None:
        result = _build_issue_key(sequence_id=123, issue_id="abc-uuid")
        self.assertEqual(result, "NP-123")

    def test_with_zero_sequence_id(self) -> None:
        result = _build_issue_key(sequence_id=0, issue_id="abc-uuid")
        self.assertEqual(result, "NP-0")

    def test_no_sequence_id(self) -> None:
        result = _build_issue_key(sequence_id=None, issue_id="abc-uuid")
        self.assertIsNone(result)


# =====================================================================
# CLI Integration Tests
# =====================================================================


class TestCliKickoffDryRun(unittest.TestCase):
    """🟢 正面測試: CLI dry-run via subprocess."""

    def test_cli_dry_run_basic(self) -> None:
        # Use the existing sample task request
        sample_path = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "task_request.sample.txt"
        )
        if not sample_path.exists():
            self.skipTest(f"Sample file not found: {sample_path}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_kickoff",
                "--input",
                str(sample_path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        # Should be valid JSON
        output = json.loads(result.stdout)
        self.assertTrue(output.get("plane_success"))
        self.assertTrue(output.get("overall_success"))

    def test_cli_missing_input_file(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_kickoff",
                "--input",
                "/nonexistent/file.txt",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr.lower())

    def test_cli_live_and_dry_run_conflict(self) -> None:
        sample_path = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "task_request.sample.txt"
        )
        if not sample_path.exists():
            self.skipTest(f"Sample file not found: {sample_path}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.neuralplane.cli_kickoff",
                "--input",
                str(sample_path),
                "--live",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FATAL", result.stderr)


# =====================================================================
# Secret Hygiene Tests
# =====================================================================


class TestSecretHygiene(unittest.TestCase):
    """🎯 正確性測試: GitHub token never appears in error messages."""

    def setUp(self) -> None:
        self._orig_environ = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig_environ)

    @patch("urllib.request.urlopen")
    def test_error_message_never_contains_token(self, mock_urlopen: MagicMock) -> None:
        # Use a token format that matches the "Bearer" pattern so redaction kicks in
        secret_token = "Bearer ghp_super_secret_token_12345"

        def raise_http_error(*args, **kwargs):
            exc = urllib.error.HTTPError(
                url="http://test",
                code=403,
                msg="Forbidden",
                hdrs={},
                fp=io.BytesIO(
                    json.dumps({"message": f"Bad credentials {secret_token}"}).encode()
                ),
            )
            raise exc

        mock_urlopen.side_effect = raise_http_error

        config = GitHubConfig(token="ghp_super_secret_token_12345", repo="owner/repo")
        client = GitHubClient(config)
        with self.assertRaises(RuntimeError) as ctx:
            client.create_branch("new-branch", "main")
        error_str = str(ctx.exception)
        # The token should not appear directly (Bearer is in API_KEY_PATTERNS → redaction)
        self.assertNotIn("ghp_super_secret_token_12345", error_str)
        self.assertIn("[GITHUB_TOKEN redacted]", error_str)
