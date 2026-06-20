"""NeuralPlane Development Kickoff — GitHub API client.

Provides:
  - ``GitHubConfig`` — holds connection settings (read from env vars).
  - ``GitHubClient`` — creates branches via the GitHub Git Data API.

Secrets (GITHUB_TOKEN) are NEVER printed in logs or error messages.

API reference: https://docs.github.com/en/rest/git/refs
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# ----------------------------------------------------------------------
# Default / env key names
# ----------------------------------------------------------------------

_ENV_TOKEN = "GITHUB_TOKEN"
_ENV_REPO = "GITHUB_REPO"

_REQUIRED_ENV_KEYS = [_ENV_TOKEN, _ENV_REPO]

_TOKEN_REDACTED = "[GITHUB_TOKEN redacted]"


# ----------------------------------------------------------------------
# MissingEnvError
# ----------------------------------------------------------------------


class MissingEnvError(Exception):
    """Raised when one or more required environment variables are not set."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        formatted = ", ".join(sorted(missing))
        super().__init__(
            f"Missing required environment variables: {formatted}. "
            "Set them before calling live methods."
        )


# ----------------------------------------------------------------------
# GitHubConfig
# ----------------------------------------------------------------------


@dataclass
class GitHubConfig:
    """Configuration for connecting to the GitHub API.

    Attributes
    ----------
    token:
        GitHub personal access token (read from ``GITHUB_TOKEN``).
    repo:
        Repository in the format ``owner/repo`` (read from ``GITHUB_REPO``).
    api_base:
        GitHub API base URL (default ``https://api.github.com``).
    """

    token: str
    repo: str
    api_base: str = "https://api.github.com"

    @classmethod
    def from_env(cls) -> GitHubConfig:
        """Build a GitHubConfig from environment variables.

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

        return cls(
            token=values[_ENV_TOKEN],
            repo=values[_ENV_REPO],
        )

    @property
    def _api_url(self) -> str:
        return f"{self.api_base}/repos/{self.repo}"


# ----------------------------------------------------------------------
# GitHubClient
# ----------------------------------------------------------------------


class GitHubClient:
    """Low-level GitHub API client for creating branches.

    Parameters
    ----------
    config:
        A populated ``GitHubConfig`` instance.
    """

    def __init__(self, config: GitHubConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_branch_head_sha(self, branch: str = "main") -> str:
        """Get the SHA of the latest commit on a branch.

        Parameters
        ----------
        branch:
            Branch name (default ``"main"``).

        Returns
        -------
        str
            The full SHA of the branch HEAD commit.

        Raises
        ------
        RuntimeError
            On HTTP errors or network failures. Never contains the token.
        """
        url = f"{self._config._api_url}/git/refs/heads/{branch}"
        data = self._get_json(url)
        # Response: {"ref": "...", "node_id": "...", "url": "...", "object": {"sha": "...", ...}}
        obj = data.get("object", {})
        sha = obj.get("sha")
        if not sha:
            raise RuntimeError(
                f"GitHub API response for refs/heads/{branch} "
                f"does not contain object.sha"
            )
        return sha

    def create_branch(
        self, new_branch: str, base_branch: str = "main"
    ) -> str:
        """Create a new branch from an existing base branch.

        Parameters
        ----------
        new_branch:
            Name of the new branch to create.
        base_branch:
            Name of the existing branch to branch from (default ``"main"``).

        Returns
        -------
        str
            The name of the newly created branch.

        Raises
        ------
        RuntimeError
            On HTTP errors or network failures. Never contains the token.
        """
        base_sha = self.get_branch_head_sha(base_branch)
        url = f"{self._config._api_url}/git/refs"

        payload = {
            "ref": f"refs/heads/{new_branch}",
            "sha": base_sha,
        }

        body_bytes = json.dumps(payload).encode("utf-8")
        headers = self._headers()

        try:
            req = urllib.request.Request(
                url, data=body_bytes, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                raw: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body: dict[str, Any] = {}
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                pass
            safe_msg = _safe_error_message(exc, body)

            if exc.code == 422:
                # 422 can mean branch already exists
                err_detail = body.get("message", "")
                if "already exists" in err_detail.lower():
                    raise RuntimeError(
                        f"Branch '{new_branch}' already exists in "
                        f"{self._config.repo}. Choose a different name."
                    ) from exc
                raise RuntimeError(safe_msg) from exc
            elif exc.code in (401, 403):
                raise RuntimeError(
                    "Authentication/authorisation failed. "
                    "Check that your GITHUB_TOKEN is valid and has write "
                    "permission on the repository."
                ) from exc
            raise RuntimeError(safe_msg) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Network error reaching GitHub API: {exc.reason}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error during GitHub API call: {exc}"
            ) from exc

        return new_branch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "NeuralPlane/1.0",
        }

    def _get_json(self, url: str) -> dict[str, Any] | list[Any]:
        headers = self._headers()
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body: dict[str, Any] = {}
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                pass
            safe_msg = _safe_error_message(exc, body)
            raise RuntimeError(safe_msg) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Network error reaching GitHub API: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"GitHub API returned non-JSON response: {exc}"
            ) from exc


# ----------------------------------------------------------------------
# Internal utilities
# ----------------------------------------------------------------------

_API_KEY_PATTERNS = ("GITHUB_TOKEN", "Authorization", "Bearer")


def _safe_error_message(
    exc: urllib.error.HTTPError, body: dict[str, Any]
) -> str:
    """Build a human-readable error message that never contains the token."""
    server_msg = _strip_token_from_string(
        body.get("message") or str(exc)
    )
    return f"GitHub API error {exc.code}: {server_msg}"


def _strip_token_from_string(text: str) -> str:
    """Remove token substrings from ``text`` to prevent accidental leakage."""
    lower = text.lower()
    for pattern in _API_KEY_PATTERNS:
        if pattern.lower() in lower:
            return _TOKEN_REDACTED
    return text
