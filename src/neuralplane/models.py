"""Data models for NeuralPlane Phase 1B System-first Plane Issue Create.

Defines TaskDraft, DoDItem, and CreateIssueResult dataclasses with validation.
Uses stdlib dataclasses only — no external dependencies.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------------------------
# Validation error
# ----------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when a TaskDraft fails validation.

    Does NOT include secrets or sensitive values in the message.
    """

    pass


# ----------------------------------------------------------------------
# Allowed values (defined here so tests can reference them)
# ----------------------------------------------------------------------

ALLOWED_PRIORITIES: frozenset[str] = frozenset({"low", "medium", "high", "urgent"})

MAX_TITLE_LENGTH: int = 255
MAX_DOD_ITEMS: int = 50
MAX_DESCRIPTION_LENGTH: int = 10_000


# ----------------------------------------------------------------------
# TaskDraft
# ----------------------------------------------------------------------


@dataclass
class TaskDraft:
    """A manually-defined task draft ready to be turned into a Plane work item.

    Attributes
    ----------
    title:
        Short, concise task title (required). Will be trimmed and used as
        the Plane work-item name.
    description:
        Detailed description of the task (required, but may be empty string).
    dod:
        List of Definition-of-Done criteria (required, at least 1 item).
        Each item must be non-empty after trimming.
    priority:
        One of "low", "medium", "high", "urgent". Optional — may be None.
    assignee_suggestions:
        List of suggested assignee identifiers (e.g. GitHub usernames or
        Plane member IDs). Optional;Plane may or may not use them.
    labels:
        Optional list of label strings to attach to the work item.
    """

    title: str
    description: str
    dod: list[str]
    priority: str | None = None
    assignee_suggestions: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskDraft:
        """Build a TaskDraft from a dict (e.g. parsed JSON).

        Unknown keys are ignored so the schema can evolve without breaking
        older consumers.
        """
        return cls(
            title=data.get("title", ""),
            description=data.get("description", "") or "",
            dod=data.get("dod") or [],
            priority=data.get("priority"),
            assignee_suggestions=data.get("assignee_suggestions") or [],
            labels=data.get("labels") or [],
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate this TaskDraft.

        Raises
        ------
        ValidationError
            If title is empty/blank, dod is empty or contains blank items,
            priority is invalid, or any field exceeds its defined limit.
        """
        # --- title --------------------------------------------------------
        title = self.title
        if not isinstance(title, str):
            raise ValidationError("title must be a string")

        title = title.strip()
        if not title:
            raise ValidationError("title is required and cannot be empty or whitespace")

        if len(title) > MAX_TITLE_LENGTH:
            raise ValidationError(
                f"title exceeds maximum length of {MAX_TITLE_LENGTH} characters "
                f"(got {len(title)})"
            )

        # --- description --------------------------------------------------
        if not isinstance(self.description, str):
            raise ValidationError("description must be a string")

        if len(self.description) > MAX_DESCRIPTION_LENGTH:
            raise ValidationError(
                f"description exceeds maximum length of {MAX_DESCRIPTION_LENGTH} "
                f"characters (got {len(self.description)})"
            )

        # --- dod ----------------------------------------------------------
        if not isinstance(self.dod, list):
            raise ValidationError("dod must be a list")

        if len(self.dod) == 0:
            raise ValidationError("dod must contain at least one item")

        if len(self.dod) > MAX_DOD_ITEMS:
            raise ValidationError(
                f"dod exceeds maximum of {MAX_DOD_ITEMS} items (got {len(self.dod)})"
            )

        for i, item in enumerate(self.dod):
            if not isinstance(item, str):
                raise ValidationError(f"dod item #{i+1} must be a string")
            if not item.strip():
                raise ValidationError(
                    f"dod item #{i+1} cannot be empty or whitespace-only"
                )

        # --- priority -----------------------------------------------------
        if self.priority is not None:
            if not isinstance(self.priority, str):
                raise ValidationError("priority must be a string or null")
            if self.priority.lower() not in ALLOWED_PRIORITIES:
                raise ValidationError(
                    f"priority must be one of {sorted(ALLOWED_PRIORITIES)}; "
                    f"got '{self.priority}'"
                )

        # --- assignee_suggestions -----------------------------------------
        if not isinstance(self.assignee_suggestions, list):
            raise ValidationError("assignee_suggestions must be a list")
        for item in self.assignee_suggestions:
            if not isinstance(item, str):
                raise ValidationError(
                    "all assignee_suggestions items must be strings"
                )

        # --- labels --------------------------------------------------------
        if not isinstance(self.labels, list):
            raise ValidationError("labels must be a list")
        for item in self.labels:
            if not isinstance(item, str):
                raise ValidationError("all labels must be strings")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def trimmed_title(self) -> str:
        """Return title with leading/trailing whitespace removed."""
        return self.title.strip()

    def priority_or_none(self) -> str | None:
        """Return lowercased priority or None."""
        if self.priority is None:
            return None
        return self.priority.lower()


# ----------------------------------------------------------------------
# CreateIssueResult
# ----------------------------------------------------------------------


@dataclass
class CreateIssueResult:
    """Result of a Plane work-item create operation.

    Attributes
    ----------
    id:
        The Plane work-item UUID, if the server returned one.
    sequence_id:
        The human-readable sequence ID (e.g. "NP-421"), if returned.
    name:
        The name/title of the created work item (mirrors the input).
    url:
        Full URL to the work item in Plane, if constructible.
    raw_response:
        The full decoded JSON response from the Plane API (for debugging /
        traceability). In production logs only non-sensitive headers and
        status codes are printed; this field is retained for test assertions.
    success:
        True if the HTTP response status was 2xx.
    error_message:
        Human-readable error description, if success is False.
        Does NOT contain the API key.
    """

    id: str | None
    sequence_id: int | None
    name: str | None
    url: str | None
    raw_response: dict[str, Any]
    success: bool
    error_message: str | None = None


# ----------------------------------------------------------------------
# WorkContextResult (Development Kickoff)
# ----------------------------------------------------------------------


import re as _re


def generate_branch_name(issue_key: str, title: str) -> str:
    """Generate a git branch name from a Plane issue key and title.

    Parameters
    ----------
    issue_key:
        Plane issue key e.g. "NP-123".
    title:
        Task title e.g. "建立會員登入 API".

    Returns
    -------
    str
        Lowercase branch name e.g. "np-123-login-api".
    """
    slug = title.strip().lower()
    slug = _re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    slug = slug[:30]
    return f"{issue_key.lower()}-{slug}"


@dataclass
class WorkContextResult:
    """Result of the Development Kickoff orchestration.

    Attributes
    ----------
    plane_success:
        Whether the Plane issue was created successfully.
    plane_issue_key:
        Human-readable issue key e.g. "NP-123".
    plane_issue_url:
        Full URL to the issue in Plane.
    plane_issue_id:
        The Plane work-item UUID.
    plane_error:
        Error message if Plane creation failed (no secrets).
    github_success:
        Whether the GitHub branch was created successfully.
    github_branch_name:
        Branch name e.g. "np-123-login-api".
    github_repo:
        Repository name e.g. "owner/repo".
    github_error:
        Error message if GitHub branch creation failed (no secrets).
    overall_success:
        True only if both Plane and GitHub succeeded.
    checkout_instruction:
        Human-readable checkout command e.g. "git checkout np-123-login-api".
    """

    plane_success: bool
    plane_issue_key: str | None = None
    plane_issue_url: str | None = None
    plane_issue_id: str | None = None
    plane_error: str | None = None

    github_success: bool = False
    github_branch_name: str | None = None
    github_repo: str | None = None
    github_error: str | None = None

    overall_success: bool = False
    checkout_instruction: str | None = None
