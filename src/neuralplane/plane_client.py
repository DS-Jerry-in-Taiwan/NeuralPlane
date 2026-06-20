"""Plane API client for NeuralPlane Phase 1B.

Provides:
  - ``PlaneConfig`` — holds all connection settings (read from env vars).
  - ``PlaneClient`` — builds the create-payload and optionally calls the Plane API.

API contract (from Phase 0 api_capability_matrix.md):

    POST {base_url}/api/v1/workspaces/{workspace_slug}/projects/{project_id}/work-items/
    Header: X-API-Key: plane_api_<token>
    Body: name, description_html, state

Secrets (PLANE_API_KEY) are NEVER printed in logs or error messages.
"""

from __future__ import annotations

import json
import os
import re as _re
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from neuralplane.models import CreateIssueResult, TaskDraft
from neuralplane.rendering import render_description_html


# ----------------------------------------------------------------------
# Default / env key names
# ----------------------------------------------------------------------

_ENV_BASE_URL = "PLANE_BASE_URL"
_ENV_WEB_BASE_URL = "PLANE_WEB_BASE_URL"
_ENV_API_KEY = "PLANE_API_KEY"
_ENV_WORKSPACE_SLUG = "PLANE_WORKSPACE_SLUG"
_ENV_PROJECT_ID = "PLANE_PROJECT_ID"
_ENV_DEFAULT_STATE_ID = "PLANE_DEFAULT_STATE_ID"

_DEFAULT_BASE_URL = "https://api.plane.so"

# Default User-Agent to avoid Cloudflare bot blocking (Error 1010).
_USER_AGENT = (
    "NeuralPlane/1.0 "
    "(https://github.com/neuralplane; +https://plane.so)"
)

_REQUIRED_ENV_KEYS = [
    _ENV_API_KEY,
    _ENV_WORKSPACE_SLUG,
    _ENV_PROJECT_ID,
    _ENV_DEFAULT_STATE_ID,
]


# ----------------------------------------------------------------------
# MissingEnvError
# ----------------------------------------------------------------------


class MissingEnvError(Exception):
    """Raised when one or more required environment variables are not set.

    The message lists the missing variable names but never their values.
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        formatted = ", ".join(sorted(missing))
        super().__init__(
            f"Missing required environment variables: {formatted}. "
            "Set them before calling live methods."
        )


# ----------------------------------------------------------------------
# PlaneConfig
# ----------------------------------------------------------------------


@dataclass
class PlaneConfig:
    """Configuration for connecting to a Plane instance.

    Attributes
    ----------
    base_url:
        Base of the Plane API (default https://api.plane.so for cloud).
        May be overridden via ``PLANE_BASE_URL`` for self-hosted instances.
    api_key:
        Full API key (``plane_api_<token>``). Read from ``PLANE_API_KEY``.
        Must NOT be printed in logs.
    workspace_slug:
        Workspace identifier used in the URL path (e.g. ``my-team``).
    project_id:
        UUID of the target Plane project.
    default_state_id:
        UUID of the initial state for created work items.
    web_base_url:
        Base of the Plane Web UI (default auto-detected from ``base_url``).
        For Plane Cloud: ``https://api.plane.so`` → ``https://app.plane.so``.
        May be overridden via ``PLANE_WEB_BASE_URL`` for self-hosted instances.
    """

    base_url: str
    api_key: str
    workspace_slug: str
    project_id: str
    default_state_id: str
    web_base_url: str = ""

    @classmethod
    def from_env(cls) -> PlaneConfig:
        """Build a PlaneConfig from environment variables.

        Raises
        ------
        MissingEnvError
            If any required variable is missing or empty.
        """
        missing: list[str] = []
        values: dict[str, str] = {}

        for key in _REQUIRED_ENV_KEYS:
            value = os.environ.get(key, "").strip()
            if not value:
                missing.append(key)
            values[key] = value

        if missing:
            raise MissingEnvError(missing)

        base_url = os.environ.get(_ENV_BASE_URL, "").strip()
        if not base_url:
            base_url = _DEFAULT_BASE_URL

        # --- web_base_url (Plane Cloud Web UI) --------------------------------
        web_base = os.environ.get(_ENV_WEB_BASE_URL, "").strip()
        if not web_base:
            # Auto-detect: replace "api.plane.so" with "app.plane.so"
            web_base = base_url.replace("api.plane.so", "app.plane.so")

        return cls(
            base_url=base_url,
            web_base_url=web_base,
            api_key=values[_ENV_API_KEY],
            workspace_slug=values[_ENV_WORKSPACE_SLUG],
            project_id=values[_ENV_PROJECT_ID],
            default_state_id=values[_ENV_DEFAULT_STATE_ID],
        )

    def build_work_items_url(self) -> str:
        """Return the full URL for the work-items create endpoint."""
        return (
            f"{self.base_url}/api/v1/workspaces/"
            f"{self.workspace_slug}/projects/{self.project_id}/work-items/"
        )

    def build_states_url(self) -> str:
        """Return the full URL for the project-states endpoint."""
        return (
            f"{self.base_url}/api/v1/workspaces/"
            f"{self.workspace_slug}/projects/{self.project_id}/states/"
        )

    def build_projects_url(self) -> str:
        """Return the full URL for the workspace projects list endpoint."""
        return (
            f"{self.base_url}/api/v1/workspaces/"
            f"{self.workspace_slug}/projects/"
        )


# ----------------------------------------------------------------------
# PlaneClient
# ----------------------------------------------------------------------


class PlaneClient:
    """Low-level Plane API client for creating work items.

    Supports two modes:
      - **dry-run** (``create_work_item(..., dry_run=True)``): builds and
        returns the payload without making any HTTP request.
      - **live** (``create_work_item(..., dry_run=False)``): calls the Plane API.

    Parameters
    ----------
    config:
        A populated ``PlaneConfig`` instance.
    """

    def __init__(self, config: PlaneConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Payload builder (always available, no network)
    # ------------------------------------------------------------------

    def build_create_payload(self, task: TaskDraft) -> dict[str, Any]:
        """Build the JSON body that will be sent to the Plane create endpoint.

        Parameters
        ----------
        task:
            A validated TaskDraft.

        Returns
        -------
        dict
            The payload dict suitable for ``json.dumps`` and ``urllib``.
        """
        description_html = render_description_html(task)

        payload: dict[str, Any] = {
            "name": task.trimmed_title(),
            "description_html": description_html,
            "state": self._config.default_state_id,
        }

        priority = task.priority_or_none()
        if priority:
            payload["priority"] = priority

        if task.assignee_suggestions and _all_uuids(task.assignee_suggestions):
            # Plane requires assignee values to be user UUIDs.
            payload["assignees"] = task.assignee_suggestions

        if task.labels and _all_uuids(task.labels):
            # Plane requires label values to be existing label UUIDs.
            # Plain-text labels are dropped to avoid a 400 error.
            payload["labels"] = task.labels

        return payload

    # ------------------------------------------------------------------
    # Live API call
    # ------------------------------------------------------------------

    def create_work_item(
        self,
        task: TaskDraft,
        *,
        dry_run: bool = False,
    ) -> CreateIssueResult:
        """Create a Plane work item from a TaskDraft.

        Parameters
        ----------
        task:
            A validated TaskDraft.
        dry_run:
            If True, only build and return the payload — do not make any HTTP
            request.  Default False (live call).

        Returns
        -------
        CreateIssueResult
            Populated with server data on success; on failure ``success=False``
            and ``error_message`` does not contain the API key.
        """
        payload = self.build_create_payload(task)

        if dry_run:
            # Return a synthetic success result with the payload attached.
            return CreateIssueResult(
                id=None,
                sequence_id=None,
                name=task.trimmed_title(),
                url=None,
                raw_response={"dry_run": True, "payload": payload},
                success=True,
                error_message=None,
            )

        # Live call — build the request
        url = self._config.build_work_items_url()
        headers = {
            "Content-Type": "application/json",
            # The key may start with "plane_api_" or be a generic API key.
            # We pass it exactly as configured.
            "X-API-Key": self._config.api_key,
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }

        body_bytes = json.dumps(payload).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                raw: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            # Attempt to read the server's error body (may be empty).
            body: dict[str, Any] = {}
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                pass

            # Surface a safe error message — never echo the API key.
            safe_message = _safe_error_message(exc, body)

            if exc.code == 401 or exc.code == 403:
                safe_message = (
                    "Authentication/authorisation failed. "
                    "Check that your PLANE_API_KEY is valid and has write "
                    "permission on the target project."
                )
            elif exc.code == 429:
                safe_message = (
                    "Plane API rate limit exceeded (60 req/min). "
                    "Retry after a short delay."
                )

            return CreateIssueResult(
                id=None,
                sequence_id=None,
                name=task.trimmed_title(),
                url=None,
                raw_response={"status_code": exc.code, "body": body},
                success=False,
                error_message=safe_message,
            )

        except urllib.error.URLError as exc:
            return CreateIssueResult(
                id=None,
                sequence_id=None,
                name=task.trimmed_title(),
                url=None,
                raw_response={},
                success=False,
                error_message=f"Network error reaching Plane: {exc.reason}",
            )

        except Exception as exc:
            return CreateIssueResult(
                id=None,
                sequence_id=None,
                name=task.trimmed_title(),
                url=None,
                raw_response={},
                success=False,
                error_message=f"Unexpected error during Plane API call: {exc}",
            )

        # Parse a successful response.
        # Plane work-item responses typically contain:
        #   id, name, sequence_id, state, project, ...
        item_id = raw.get("id")
        seq_id: int | None = None
        seq_val = raw.get("sequence_id")
        if isinstance(seq_val, int):
            seq_id = seq_val
        elif isinstance(seq_val, str):
            # Try to extract an integer from strings like "NP-421"
            m = _SEQUENCE_NUMERIC_RE.search(seq_val)
            if m:
                seq_id = int(m.group(1))

        # Try to build a URL if we have enough info.
        work_item_url: str | None = None
        if item_id:
            work_item_url = (
                f"{self._config.web_base_url}/"
                f"{self._config.workspace_slug}/"
                f"projects/{self._config.project_id}/"
                f"work-items/{item_id}"
            )

        return CreateIssueResult(
            id=item_id,
            sequence_id=seq_id,
            name=raw.get("name", task.trimmed_title()),
            url=work_item_url,
            raw_response=raw,
            success=True,
            error_message=None,
        )

    # ------------------------------------------------------------------
    # Discovery: list project states
    # ------------------------------------------------------------------

    def list_states(self) -> list[dict[str, Any]]:
        """Fetch all workflow states for the configured project.

        Calls ``GET /api/v1/workspaces/{slug}/projects/{id}/states/``.

        Returns
        -------
        list[dict[str, Any]]
            Each dict contains at least ``id``, ``name``, and ``group``.
            Returns an empty list if the API returns an empty array.
            Plane API may wrap results in a ``"results"`` key or return
            a top-level array — both forms are handled.

        Raises
        ------
        RuntimeError
            On HTTP errors (401/403/429/etc.) or network failures.
            The error message never contains the API key.
        """
        url = self._config.build_states_url()
        try:
            result = self._get_json(url)
        except urllib.error.HTTPError as exc:
            body: dict[str, Any] = {}
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                pass
            safe_message = _safe_error_message(exc, body)
            raise RuntimeError(safe_message) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Network error reaching Plane states endpoint: {exc.reason}"
            ) from exc

        # Handle both wrapped and unwrapped response shapes.
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        if isinstance(result, list):
            return result
        return []

    # ------------------------------------------------------------------
    # Discovery: list workspace projects
    # ------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        """Fetch all projects in the configured workspace.

        Calls ``GET /api/v1/workspaces/{slug}/projects/``.

        Returns
        -------
        list[dict[str, Any]]
            Each dict contains at least ``id``, ``name``, and ``identifier``.
            Returns an empty list if the API returns an empty array.
            Plane API may wrap results in a ``"results"`` key or return
            a top-level array — both forms are handled.

        Raises
        ------
        RuntimeError
            On HTTP errors (401/403/429/etc.) or network failures.
            The error message never contains the API key.
        """
        url = self._config.build_projects_url()
        try:
            result = self._get_json(url)
        except urllib.error.HTTPError as exc:
            body: dict[str, Any] = {}
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                pass
            safe_message = _safe_error_message(exc, body)
            raise RuntimeError(safe_message) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Network error reaching Plane projects endpoint: {exc.reason}"
            ) from exc

        # Handle both wrapped and unwrapped response shapes.
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        if isinstance(result, list):
            return result
        return []

    # ------------------------------------------------------------------
    # Internal GET helper
    # ------------------------------------------------------------------

    def _get_json(self, url: str) -> dict[str, Any] | list[Any]:
        """Perform a GET request and return parsed JSON.

        Parameters
        ----------
        url:
            Full URL to GET.

        Returns
        -------
        Parsed JSON (dict or list).

        Raises
        ------
        urllib.error.HTTPError
        urllib.error.URLError
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
            "X-API-Key": self._config.api_key,
        }
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Plane API returned non-JSON response: {exc}"
            ) from exc


# ----------------------------------------------------------------------
# Internal utilities
# ----------------------------------------------------------------------

_SEQUENCE_NUMERIC_RE = _re.compile(r"-(\d+)$")

_API_KEY_PATTERNS = ("PLANE_API_KEY", "plane_api_", "X-API-Key")


def _all_uuids(items: list[str]) -> bool:
    """Return True if every item in ``items`` is a valid UUID."""
    for item in items:
        try:
            uuid.UUID(item)
        except (ValueError, AttributeError):
            return False
    return True


def _safe_error_message(exc: urllib.error.HTTPError, body: dict[str, Any]) -> str:
    """Build a human-readable error message that never contains the API key."""
    server_msg = _strip_api_key_from_string(
        body.get("message") or str(exc)
    )
    return f"Plane API error {exc.code}: {server_msg}"


def _strip_api_key_from_string(text: str) -> str:
    """Remove API key substrings from ``text`` to prevent accidental leakage."""
    lower = text.lower()
    for pattern in _API_KEY_PATTERNS:
        if pattern.lower() in lower:
            return f"[{pattern} redacted]"
    return text
