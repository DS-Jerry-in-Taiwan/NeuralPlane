"""Unit tests for NeuralPlane Phase 1B System-first Plane Issue Create.

Covers:
  - TaskDraft schema (positive, negative, range, correctness, boundary)
  - description_html renderer (HTML escaping, [DoD] anchor)
  - Plane payload builder (required fields, API contract alignment)
  - PlaneClient dry-run (no network, no env required)
  - PlaneClient live mode (env validation, secret hygiene)
  - CreateIssueResult parsing (positive, negative, boundary)

Run with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/neuralplane/ is on the path for direct imports.
# parents[1] = tests/ → NeuralPlane/ → parents[1] / "src" = NeuralPlane/src/
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from neuralplane.models import (
    ALLOWED_PRIORITIES,
    MAX_DESCRIPTION_LENGTH,
    MAX_DOD_ITEMS,
    MAX_TITLE_LENGTH,
    CreateIssueResult,
    TaskDraft,
    ValidationError,
)
from neuralplane.plane_client import (
    MissingEnvError,
    PlaneClient,
    PlaneConfig,
)
from neuralplane.rendering import render_description_html


# ----------------------------------------------------------------------
# Sample fixture helpers
# ----------------------------------------------------------------------


def _sample_dict() -> dict:
    """Return a valid TaskDraft dict."""
    return {
        "title": "實作會員登入 API",
        "description": "建立會員登入端點，支援 bcrypt 密碼驗證與 JWT token 簽發。",
        "dod": [
            "密碼驗證必須使用 bcrypt hash，不可明文比對。",
            "JWT access token 有效期限必須為 24 小時。",
            "登入成功、密碼錯誤、使用者不存在三種情境必須有測試。",
        ],
        "priority": "high",
        "assignee_suggestions": [],
        "labels": ["550e8400-e29b-41d4-a716-446655440000", "660e8400-e29b-41d4-a716-446655440001"],
    }


def _minimal_dict() -> dict:
    return {
        "title": "Minimal Task",
        "description": "A task with only required fields.",
        "dod": ["This is the only DoD item."],
    }


# =====================================================================
# TaskDraft — Construction
# =====================================================================


class TestTaskDraftFromDict(unittest.TestCase):
    """🟢 正面測試: valid JSON → TaskDraft."""

    def test_full_dict_round_trip(self) -> None:
        d = _sample_dict()
        task = TaskDraft.from_dict(d)
        self.assertEqual(task.title, d["title"])
        self.assertEqual(task.description, d["description"])
        self.assertEqual(task.dod, d["dod"])
        self.assertEqual(task.priority, d["priority"])
        self.assertEqual(task.assignee_suggestions, d["assignee_suggestions"])
        self.assertEqual(task.labels, d["labels"])

    def test_minimal_dict(self) -> None:
        d = _minimal_dict()
        task = TaskDraft.from_dict(d)
        self.assertEqual(task.title, d["title"])
        self.assertEqual(task.dod, d["dod"])
        self.assertIsNone(task.priority)
        self.assertEqual(task.assignee_suggestions, [])
        self.assertEqual(task.labels, [])

    def test_unknown_keys_ignored(self) -> None:
        d = {**_sample_dict(), "unknown_field": 123, "another": "ignored"}
        task = TaskDraft.from_dict(d)
        # Should not raise; unknown fields are ignored.
        self.assertEqual(task.title, d["title"])

    def test_missing_optional_fields_default_to_empty(self) -> None:
        d = {"title": "T", "dod": ["Item"]}
        task = TaskDraft.from_dict(d)
        self.assertEqual(task.description, "")
        self.assertIsNone(task.priority)
        self.assertEqual(task.assignee_suggestions, [])
        self.assertEqual(task.labels, [])


# =====================================================================
# TaskDraft — Validation: title
# =====================================================================


class TestTaskDraftTitleValidation(unittest.TestCase):
    """Test class: TaskDraft.title validation."""

    def test_empty_title_raises(self) -> None:
        """🔴 負面測試: empty string title must be rejected."""
        task = TaskDraft(title="", description="D", dod=["D"])
        with self.assertRaises(ValidationError) as ctx:
            task.validate()
        self.assertIn("title", str(ctx.exception).lower())

    def test_whitespace_only_title_raises(self) -> None:
        """🔴 負面測試: whitespace-only title must be rejected."""
        task = TaskDraft(title="   \t", description="D", dod=["D"])
        with self.assertRaises(ValidationError):
            task.validate()

    def test_title_too_long_raises(self) -> None:
        """📏 範圍測試: title exceeding MAX_TITLE_LENGTH raises."""
        task = TaskDraft(title="A" * (MAX_TITLE_LENGTH + 1), description="D", dod=["D"])
        with self.assertRaises(ValidationError) as ctx:
            task.validate()
        self.assertIn(str(MAX_TITLE_LENGTH), str(ctx.exception))

    def test_title_at_limit_is_valid(self) -> None:
        """📏 範圍測試: title at exactly MAX_TITLE_LENGTH is accepted."""
        task = TaskDraft(title="A" * MAX_TITLE_LENGTH, description="D", dod=["D"])
        task.validate()  # Should not raise

    def test_trimmed_title_used_in_payload(self) -> None:
        """🎯 正確性測試: trimmed_title() strips whitespace."""
        task = TaskDraft(title="  登入 API  ", description="D", dod=["D"])
        self.assertEqual(task.trimmed_title(), "登入 API")


# =====================================================================
# TaskDraft — Validation: dod
# =====================================================================


class TestTaskDraftDodValidation(unittest.TestCase):
    """Test class: TaskDraft.dod validation."""

    def test_empty_dod_list_raises(self) -> None:
        """🔴 負面測試: empty dod list [] must be rejected."""
        task = TaskDraft(title="T", description="D", dod=[])
        with self.assertRaises(ValidationError) as ctx:
            task.validate()
        self.assertIn("dod", str(ctx.exception).lower())

    def test_blank_dod_item_raises(self) -> None:
        """🔲 邊界測試: a blank string in dod list must be rejected."""
        task = TaskDraft(title="T", description="D", dod=["Valid item", "  \t", ""])
        with self.assertRaises(ValidationError) as ctx:
            task.validate()
        self.assertIn("dod", str(ctx.exception).lower())

    def test_too_many_dod_items_raises(self) -> None:
        """📏 範圍測試: more than MAX_DOD_ITEMS raises."""
        task = TaskDraft(
            title="T",
            description="D",
            dod=[f"Item {i}" for i in range(MAX_DOD_ITEMS + 1)],
        )
        with self.assertRaises(ValidationError) as ctx:
            task.validate()
        self.assertIn(str(MAX_DOD_ITEMS), str(ctx.exception))

    def test_dod_at_limit_is_valid(self) -> None:
        """📏 範圍測試: exactly MAX_DOD_ITEMS is accepted."""
        task = TaskDraft(
            title="T",
            description="D",
            dod=[f"Item {i}" for i in range(MAX_DOD_ITEMS)],
        )
        task.validate()  # Should not raise

    def test_single_dod_item_valid(self) -> None:
        """🟢 正面測試: one-item dod list is valid."""
        task = TaskDraft(title="T", description="D", dod=["Just one."])
        task.validate()

    def test_dod_all_items_preserved(self) -> None:
        """🎯 正確性測試: all dod items are stored as-is (before trimming in renderer)."""
        items = ["Item A", "Item B", "Item C"]
        task = TaskDraft(title="T", description="D", dod=items)
        self.assertEqual(task.dod, items)


# =====================================================================
# TaskDraft — Validation: priority
# =====================================================================


class TestTaskDraftPriorityValidation(unittest.TestCase):
    """Test class: TaskDraft.priority validation."""

    def test_valid_priorities(self) -> None:
        """🟢 正面測試: all allowed priority values are accepted."""
        for p in sorted(ALLOWED_PRIORITIES):
            task = TaskDraft(title="T", description="D", dod=["D"], priority=p)
            task.validate()  # Should not raise

    def test_invalid_priority_raises(self) -> None:
        """🔴 負面測試: unknown priority value raises."""
        task = TaskDraft(title="T", description="D", dod=["D"], priority="critical")
        with self.assertRaises(ValidationError) as ctx:
            task.validate()
        self.assertIn("priority", str(ctx.exception).lower())

    def test_priority_case_insensitive(self) -> None:
        """🔲 邊界測試: priority is case-insensitive."""
        task = TaskDraft(title="T", description="D", dod=["D"], priority="HIGH")
        task.validate()  # should pass (lowercased in validator)

    def test_none_priority_valid(self) -> None:
        """🟢 正面測試: priority=None is valid."""
        task = TaskDraft(title="T", description="D", dod=["D"], priority=None)
        task.validate()

    def test_priority_or_none_lowercases(self) -> None:
        """🎯 正確性測試: priority_or_none() returns lowercased value."""
        task = TaskDraft(title="T", description="D", dod=["D"], priority="HIGH")
        self.assertEqual(task.priority_or_none(), "high")


# =====================================================================
# TaskDraft — Validation: description
# =====================================================================


class TestTaskDraftDescriptionValidation(unittest.TestCase):
    """Test class: TaskDraft.description validation."""

    def test_empty_description_is_valid(self) -> None:
        """🟢 正面測試: empty description string is valid."""
        task = TaskDraft(title="T", description="", dod=["D"])
        task.validate()  # should not raise

    def test_description_too_long_raises(self) -> None:
        """📏 範圍測試: description exceeding MAX_DESCRIPTION_LENGTH raises."""
        task = TaskDraft(
            title="T",
            description="A" * (MAX_DESCRIPTION_LENGTH + 1),
            dod=["D"],
        )
        with self.assertRaises(ValidationError) as ctx:
            task.validate()
        self.assertIn(str(MAX_DESCRIPTION_LENGTH), str(ctx.exception))

    def test_description_at_limit_valid(self) -> None:
        """📏 範圍測試: description at exactly the limit is accepted."""
        task = TaskDraft(
            title="T",
            description="A" * MAX_DESCRIPTION_LENGTH,
            dod=["D"],
        )
        task.validate()


# =====================================================================
# Rendering — description_html
# =====================================================================


class TestRenderDescriptionHtml(unittest.TestCase):
    """Test class: render_description_html output quality."""

    def test_dod_items_all_present(self) -> None:
        """🎯 正確性測試: every DoD item appears in the HTML output."""
        task = TaskDraft.from_dict(_sample_dict())
        html = render_description_html(task)
        for item in task.dod:
            self.assertIn(item, html)

    def test_dod_anchor_present(self) -> None:
        """🎯 正確性測試: output contains the [DoD] anchor string."""
        task = TaskDraft.from_dict(_sample_dict())
        html = render_description_html(task)
        self.assertIn("[DoD]", html)

    def test_description_included(self) -> None:
        """🟢 正面測試: description text is present in the HTML."""
        task = TaskDraft.from_dict(_sample_dict())
        html = render_description_html(task)
        self.assertIn(task.description, html)

    def test_html_escaping_script_tag(self) -> None:
        """🔴 負面測試: a <script> tag in any user field is escaped."""
        malicious = "Text with <script>alert(1)</script> inside"
        task = TaskDraft(
            title="T",
            description=malicious,
            dod=[malicious],
        )
        html = render_description_html(task)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_html_escaping_double_quote(self) -> None:
        """🔴 負面測試: double quotes in user text are escaped."""
        task = TaskDraft(
            title='Title with "double quotes"',
            description="",
            dod=['Item with "quotes" and <special>'],
        )
        html = render_description_html(task)
        # Raw (unescaped) double-quote characters must NOT appear in the HTML.
        self.assertNotIn('"quotes"', html)
        # The escaped version (&quot;) must appear in place of raw quotes.
        self.assertIn("&quot;", html)

    def test_empty_description_still_has_dod(self) -> None:
        """🔲 邊界測試: empty description but DoD present → DoD still renders."""
        task = TaskDraft(title="T", description="", dod=["Only DoD item."])
        html = render_description_html(task)
        self.assertIn("[DoD]", html)
        self.assertIn("Only DoD item.", html)

    def test_long_description_no_crash(self) -> None:
        """📏 範圍測試: a 5000-char description does not crash."""
        task = TaskDraft(
            title="T",
            description="A" * 5000,
            dod=["D"],
        )
        html = render_description_html(task)  # should not raise
        self.assertIn("A" * 5000, html)

    def test_output_contains_ul_li_tags(self) -> None:
        """🟢 正面測試: DoD items are wrapped in <ul>/<li> tags."""
        task = TaskDraft.from_dict(_minimal_dict())
        html = render_description_html(task)
        self.assertIn("<ul>", html)
        self.assertIn("<li>", html)

    def test_output_contains_h2_anchor(self) -> None:
        """🟢 正面測試: [DoD] is rendered as an <h2> tag."""
        task = TaskDraft.from_dict(_minimal_dict())
        html = render_description_html(task)
        self.assertIn("<h2>[DoD]</h2>", html)


# =====================================================================
# Plane Config
# =====================================================================


class TestPlaneConfigFromEnv(unittest.TestCase):
    """Test class: PlaneConfig environment variable loading."""

    def setUp(self) -> None:
        # Save original env to restore in tearDown.
        self._original_env = dict(os.environ)

    def tearDown(self) -> None:
        # Restore original env.
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_all_required_vars_present(self) -> None:
        """🟢 正面測試: with all env vars set, from_env() succeeds."""
        os.environ.update({
            "PLANE_API_KEY": "plane_api_testkey",
            "PLANE_WORKSPACE_SLUG": "my-workspace",
            "PLANE_PROJECT_ID": "proj-uuid-123",
            "PLANE_DEFAULT_STATE_ID": "state-uuid-456",
        })
        cfg = PlaneConfig.from_env()
        self.assertEqual(cfg.api_key, "plane_api_testkey")
        self.assertEqual(cfg.workspace_slug, "my-workspace")
        self.assertEqual(cfg.project_id, "proj-uuid-123")
        self.assertEqual(cfg.default_state_id, "state-uuid-456")
        self.assertEqual(cfg.base_url, "https://api.plane.so")  # default

    def test_base_url_overridden(self) -> None:
        """🟢 正面測試: PLANE_BASE_URL overrides the default."""
        os.environ.update({
            "PLANE_BASE_URL": "https://selfhosted.plane.example.com",
            "PLANE_API_KEY": "key",
            "PLANE_WORKSPACE_SLUG": "ws",
            "PLANE_PROJECT_ID": "p",
            "PLANE_DEFAULT_STATE_ID": "s",
        })
        cfg = PlaneConfig.from_env()
        self.assertEqual(cfg.base_url, "https://selfhosted.plane.example.com")

    def test_missing_any_required_var_raises(self) -> None:
        """🔴 負面測試: missing a single required var raises MissingEnvError."""
        # Set only some vars.
        os.environ.update({
            "PLANE_API_KEY": "key",
            # PLANE_WORKSPACE_SLUG missing
            "PLANE_PROJECT_ID": "p",
            "PLANE_DEFAULT_STATE_ID": "s",
        })
        with self.assertRaises(MissingEnvError) as ctx:
            PlaneConfig.from_env()
        self.assertIn("PLANE_WORKSPACE_SLUG", str(ctx.exception))
        # Must NOT contain the key value.
        self.assertNotIn("key", str(ctx.exception))

    def test_error_message_does_not_leak_api_key(self) -> None:
        """🟢 正面測試: MissingEnvError message does not contain the API key value."""
        os.environ.update({
            "PLANE_API_KEY": "plane_api_SUPER_SECRET_KEY_12345",
            "PLANE_WORKSPACE_SLUG": "ws",
            "PLANE_PROJECT_ID": "p",
            # PLANE_DEFAULT_STATE_ID missing
        })
        with self.assertRaises(MissingEnvError) as ctx:
            PlaneConfig.from_env()
        error_str = str(ctx.exception)
        self.assertNotIn("SUPER_SECRET", error_str)
        self.assertNotIn("plane_api_", error_str)


# =====================================================================
# PlaneClient — Payload Builder
# =====================================================================


class TestPlaneClientPayloadBuilder(unittest.TestCase):
    """Test class: PlaneClient.build_create_payload()."""

    def _cfg(self) -> PlaneConfig:
        return PlaneConfig(
            base_url="https://api.plane.so",
            api_key="plane_api_test",
            workspace_slug="test-workspace",
            project_id="proj-test-123",
            default_state_id="state-test-456",
        )

    def test_required_fields_present(self) -> None:
        """🟢 正面測試: payload contains name, description_html, state."""
        task = TaskDraft.from_dict(_sample_dict())
        client = PlaneClient(self._cfg())
        payload = client.build_create_payload(task)
        self.assertIn("name", payload)
        self.assertIn("description_html", payload)
        self.assertIn("state", payload)

    def test_name_matches_title(self) -> None:
        """🎯 正確性測試: payload name equals trimmed title."""
        task = TaskDraft.from_dict(_sample_dict())
        client = PlaneClient(self._cfg())
        payload = client.build_create_payload(task)
        self.assertEqual(payload["name"], task.trimmed_title())

    def test_state_equals_config_default(self) -> None:
        """🎯 正確性測試: payload state comes from config."""
        cfg = self._cfg()
        task = TaskDraft.from_dict(_minimal_dict())
        client = PlaneClient(cfg)
        payload = client.build_create_payload(task)
        self.assertEqual(payload["state"], cfg.default_state_id)

    def test_priority_included_when_present(self) -> None:
        """🟢 正面測試: priority is included in payload when set."""
        task = TaskDraft.from_dict(_sample_dict())  # priority="high"
        payload = PlaneClient(self._cfg()).build_create_payload(task)
        self.assertIn("priority", payload)
        self.assertEqual(payload["priority"], "high")

    def test_priority_absent_when_none(self) -> None:
        """🟢 正面測試: priority key is absent when task.priority is None."""
        task = TaskDraft(title="T", description="D", dod=["D"], priority=None)
        payload = PlaneClient(self._cfg()).build_create_payload(task)
        self.assertNotIn("priority", payload)

    def test_assignees_included(self) -> None:
        """🟢 正面測試: assignees field is included when suggestions present."""
        d = _sample_dict()
        d["assignee_suggestions"] = [
            "550e8400-e29b-41d4-a716-446655440000",
            "660e8400-e29b-41d4-a716-446655440001",
        ]
        task = TaskDraft.from_dict(d)
        payload = PlaneClient(self._cfg()).build_create_payload(task)
        self.assertIn("assignees", payload)
        self.assertEqual(
            payload["assignees"],
            ["550e8400-e29b-41d4-a716-446655440000",
             "660e8400-e29b-41d4-a716-446655440001"],
        )

    def test_labels_included(self) -> None:
        """🟢 正面測試: labels field is included when present."""
        task = TaskDraft.from_dict(_sample_dict())  # labels=UUIDs
        payload = PlaneClient(self._cfg()).build_create_payload(task)
        self.assertIn("labels", payload)
        self.assertEqual(
            payload["labels"],
            ["550e8400-e29b-41d4-a716-446655440000",
             "660e8400-e29b-41d4-a716-446655440001"],
        )

    def test_description_html_contains_dod(self) -> None:
        """🎯 正確性測試: description_html in payload contains DoD items."""
        task = TaskDraft.from_dict(_sample_dict())
        payload = PlaneClient(self._cfg()).build_create_payload(task)
        desc_html = payload["description_html"]
        for item in task.dod:
            self.assertIn(item, desc_html)
        self.assertIn("[DoD]", desc_html)

    def test_endpoint_path_matches_api_contract(self) -> None:
        """🎯 正確性測試: build URL follows Phase 0 API contract path pattern."""
        cfg = self._cfg()
        expected_url = (
            "https://api.plane.so/api/v1/workspaces/test-workspace/"
            "projects/proj-test-123/work-items/"
        )
        self.assertEqual(cfg.build_work_items_url(), expected_url)


# =====================================================================
# PlaneClient — Dry-run
# =====================================================================


class TestPlaneClientDryRun(unittest.TestCase):
    """Test class: dry-run mode (no network, no env required)."""

    def _cfg(self) -> PlaneConfig:
        return PlaneConfig(
            base_url="https://api.plane.so",
            api_key="plane_api_fake",
            workspace_slug="dry-workspace",
            project_id="dry-proj",
            default_state_id="dry-state",
        )

    def test_dry_run_returns_success(self) -> None:
        """🟢 正面測試: dry_run=True returns success=True with payload."""
        task = TaskDraft.from_dict(_sample_dict())
        client = PlaneClient(self._cfg())
        result = client.create_work_item(task, dry_run=True)
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)

    def test_dry_run_includes_payload(self) -> None:
        """🟢 正面測試: dry_run result raw_response contains the payload."""
        task = TaskDraft.from_dict(_sample_dict())
        client = PlaneClient(self._cfg())
        result = client.create_work_item(task, dry_run=True)
        self.assertIn("payload", result.raw_response)
        self.assertIn("dry_run", result.raw_response)
        self.assertTrue(result.raw_response["dry_run"])

    def test_dry_run_no_api_key_in_result(self) -> None:
        """🟢 正面測試: result does not echo the API key."""
        task = TaskDraft.from_dict(_minimal_dict())
        result = PlaneClient(self._cfg()).create_work_item(task, dry_run=True)
        result_str = json.dumps(result.raw_response)
        self.assertNotIn("plane_api_fake", result_str)

    def test_build_create_payload_no_network(self) -> None:
        """🔲 邊界測試 / dry-run: build_create_payload never makes HTTP calls."""
        task = TaskDraft.from_dict(_minimal_dict())
        client = PlaneClient(self._cfg())
        with patch("urllib.request.urlopen") as mock_urlopen:
            payload = client.build_create_payload(task)
            mock_urlopen.assert_not_called()
        self.assertIsInstance(payload, dict)


# =====================================================================
# PlaneClient — Live mode: env validation
# =====================================================================


class TestPlaneClientLiveEnvValidation(unittest.TestCase):
    """Test class: live mode env validation / fail-fast."""

    def setUp(self) -> None:
        self._original_env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_missing_api_key_raises_missing_env_error(self) -> None:
        """🔴 負面測試: missing PLANE_API_KEY raises MissingEnvError at from_env."""
        # Unset all required vars.
        for key in ["PLANE_API_KEY", "PLANE_WORKSPACE_SLUG",
                    "PLANE_PROJECT_ID", "PLANE_DEFAULT_STATE_ID"]:
            os.environ.pop(key, None)
        with self.assertRaises(MissingEnvError) as ctx:
            PlaneConfig.from_env()
        self.assertIn("PLANE_API_KEY", str(ctx.exception))


# =====================================================================
# PlaneClient — Live mode: HTTP error handling
# =====================================================================


class TestPlaneClientHttpErrors(unittest.TestCase):
    """Test class: HTTP error handling (401, 403, 429, network)."""

    def _cfg(self) -> PlaneConfig:
        return PlaneConfig(
            base_url="https://api.plane.so",
            api_key="plane_api_fake",
            workspace_slug="ws",
            project_id="proj",
            default_state_id="state",
        )

    def _mock_http_error(self, code: int, body: str = "{}") -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            url="https://api.plane.so/api/v1/workspaces/ws/projects/proj/work-items/",
            code=code,
            msg=f"HTTP {code}",
            hdrs={},
            fp=None,
        )

    def test_401_error_no_key_leak(self) -> None:
        """🔴 負面測試: 401 error message does NOT contain the API key."""
        task = TaskDraft.from_dict(_minimal_dict())
        client = PlaneClient(self._cfg())

        exc = self._mock_http_error(401)

        with patch("urllib.request.urlopen", side_effect=exc):
            result = client.create_work_item(task, dry_run=False)

        self.assertFalse(result.success)
        error_str = result.error_message or ""
        self.assertNotIn("plane_api_fake", error_str)
        self.assertNotIn("X-API-Key", error_str)

    def test_403_error_message_is_safe(self) -> None:
        """🔴 負面測試: 403 error message does NOT contain the API key."""
        task = TaskDraft.from_dict(_minimal_dict())
        client = PlaneClient(self._cfg())

        exc = self._mock_http_error(403)

        with patch("urllib.request.urlopen", side_effect=exc):
            result = client.create_work_item(task, dry_run=False)

        self.assertFalse(result.success)
        error_str = result.error_message or ""
        self.assertNotIn("plane_api_fake", error_str)

    def test_429_includes_rate_limit_hint(self) -> None:
        """📏 範圍測試: 429 response includes a rate-limit hint."""
        task = TaskDraft.from_dict(_minimal_dict())
        client = PlaneClient(self._cfg())

        exc = self._mock_http_error(429)

        with patch("urllib.request.urlopen", side_effect=exc):
            result = client.create_work_item(task, dry_run=False)

        self.assertFalse(result.success)
        self.assertIn("rate limit", (result.error_message or "").lower())

    def test_network_error_includes_reason(self) -> None:
        """📏 範圍測試: URLError includes the reason string."""
        task = TaskDraft.from_dict(_minimal_dict())
        client = PlaneClient(self._cfg())

        exc = urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=exc):
            result = client.create_work_item(task, dry_run=False)

        self.assertFalse(result.success)
        self.assertIn("Connection refused", (result.error_message or ""))


# =====================================================================
# CreateIssueResult — Parsing
# =====================================================================


class TestCreateIssueResultParsing(unittest.TestCase):
    """Test class: CreateIssueResult response parsing."""

    def test_all_fields_parsed_from_mock_201_response(self) -> None:
        """🟢 正面測試: successful 201 response parses all fields."""
        mock_resp = {
            "id": "uuid-abc123",
            "name": "實作會員登入 API",
            "sequence_id": 421,
            "state": "state-uuid",
            "project": "proj-uuid",
        }

        def fake_urlopen(req, timeout=None):
            f = MagicMock()
            f.status = 201
            f.read.return_value = json.dumps(mock_resp).encode()
            f.__enter__ = MagicMock(return_value=f)
            f.__exit__ = MagicMock(return_value=False)
            return f

        cfg = PlaneConfig(
            base_url="https://api.plane.so",
            api_key="plane_api_test",
            workspace_slug="ws",
            project_id="proj",
            default_state_id="state",
        )
        task = TaskDraft.from_dict(_sample_dict())
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = PlaneClient(cfg).create_work_item(task, dry_run=False)

        self.assertTrue(result.success)
        self.assertEqual(result.id, "uuid-abc123")
        self.assertEqual(result.name, "實作會員登入 API")
        self.assertEqual(result.sequence_id, 421)

    def test_missing_sequence_id_does_not_crash(self) -> None:
        """🔲 邊界測試: response without sequence_id yields None."""
        mock_resp = {
            "id": "uuid-abc123",
            "name": "Task",
            # no sequence_id
        }

        def fake_urlopen(req, timeout=None):
            f = MagicMock()
            f.status = 201
            f.read.return_value = json.dumps(mock_resp).encode()
            f.__enter__ = MagicMock(return_value=f)
            f.__exit__ = MagicMock(return_value=False)
            return f

        cfg = PlaneConfig(
            base_url="https://api.plane.so",
            api_key="plane_api_test",
            workspace_slug="ws",
            project_id="proj",
            default_state_id="state",
        )
        task = TaskDraft(title="T", description="D", dod=["D"])

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = PlaneClient(cfg).create_work_item(task, dry_run=False)

        self.assertTrue(result.success)
        self.assertIsNone(result.sequence_id)

    def test_result_name_equals_task_title(self) -> None:
        """🎯 正確性測試: result.name mirrors the task's trimmed title."""
        mock_resp = {
            "id": "uuid-xyz",
            "name": "Task With Title",
            "sequence_id": 1,
        }

        def fake_urlopen(req, timeout=None):
            f = MagicMock()
            f.status = 201
            f.read.return_value = json.dumps(mock_resp).encode()
            f.__enter__ = MagicMock(return_value=f)
            f.__exit__ = MagicMock(return_value=False)
            return f

        cfg = PlaneConfig(
            base_url="https://api.plane.so",
            api_key="plane_api_test",
            workspace_slug="ws",
            project_id="proj",
            default_state_id="state",
        )
        task = TaskDraft(title="  Task With Title  ", description="D", dod=["D"])

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = PlaneClient(cfg).create_work_item(task, dry_run=False)

        self.assertEqual(result.name, "Task With Title")


# =====================================================================
# Secret hygiene
# =====================================================================


class TestSecretHygiene(unittest.TestCase):
    """Test class: API key / secrets never appear in user-facing output."""

    def test_plain_api_key_not_in_missing_env_error(self) -> None:
        """🟢 正面測試 / secret hygiene: MissingEnvError excludes key values."""
        import os
        original = dict(os.environ)
        try:
            os.environ.update({
                "PLANE_API_KEY": "plane_api_TOTALLY_REAL_KEY_DO_NOT_LEAK",
                "PLANE_WORKSPACE_SLUG": "ws",
                "PLANE_PROJECT_ID": "p",
                # PLANE_DEFAULT_STATE_ID intentionally missing
            })
            with self.assertRaises(MissingEnvError) as ctx:
                PlaneConfig.from_env()
            msg = str(ctx.exception)
            self.assertNotIn("TOTALLY_REAL_KEY", msg)
            self.assertNotIn("plane_api_", msg)
        finally:
            os.environ.clear()
            os.environ.update(original)


# =====================================================================
# CLI integration tests (dry-run, no network)
# =====================================================================


class TestCliDryRunIntegration(unittest.TestCase):
    """Test class: CLI dry-run flow with the sample JSON."""

    def setUp(self) -> None:
        self._original_cwd = os.getcwd()
        # repo root = NeuralPlane/ (parent of tests/)
        repo_root = Path(__file__).resolve().parents[1]
        self._sample_json = repo_root / "examples" / "task_draft.sample.json"

    def test_conflicting_flags_fail_fast(self) -> None:
        """🔴 負面測試: --dry-run and --live together → fatal error, no token leak."""
        # Use the minimal dict as input so we don't need validation to pass.
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(_minimal_dict(), fh)
            tmp_path = fh.name

        try:
            import subprocess
            proc = subprocess.run(
                [
                    sys.executable, "-m",
                    "src.neuralplane.cli_create_plane_issue",
                    "--input", tmp_path,
                    "--dry-run", "--live",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "src"},
            )
            # Should exit with error.
            self.assertNotEqual(proc.returncode, 0)
            # Error message should mention conflicting flags.
            combined = proc.stdout + proc.stderr
            self.assertIn("--live", combined)
            self.assertIn("--dry-run", combined)
            # Must NOT mention API key or token value.
            self.assertNotIn("plane_api", combined.lower())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_sample_json_loads_and_validates(self) -> None:
        """🟢 正面測試: sample JSON is a valid TaskDraft."""
        raw = json.loads(self._sample_json.read_text())
        task = TaskDraft.from_dict(raw)
        task.validate()  # must not raise

    def test_sample_json_produces_description_html(self) -> None:
        """🟢 正面測試: sample JSON renders description_html with [DoD]."""
        raw = json.loads(self._sample_json.read_text())
        task = TaskDraft.from_dict(raw)
        html = render_description_html(task)
        self.assertIn("[DoD]", html)
        self.assertIn("<ul>", html)
        for item in raw["dod"]:
            self.assertIn(item, html)

    def test_sample_json_builds_plane_payload(self) -> None:
        """🟢 正面測試: sample JSON builds a complete Plane create payload."""
        raw = json.loads(self._sample_json.read_text())
        task = TaskDraft.from_dict(raw)

        cfg = PlaneConfig(
            base_url="https://api.plane.so",
            api_key="plane_api_test",
            workspace_slug="test-ws",
            project_id="test-proj",
            default_state_id="test-state",
        )
        payload = PlaneClient(cfg).build_create_payload(task)

        self.assertEqual(payload["name"], "實作會員登入 API")
        self.assertIn("description_html", payload)
        self.assertIn("state", payload)
        self.assertEqual(payload["priority"], "high")


# =====================================================================
# Entry-point guard
# =====================================================================

if __name__ == "__main__":
    unittest.main()
