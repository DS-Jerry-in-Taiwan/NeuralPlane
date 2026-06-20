"""NeuralPlane Phase 1D — E2E Dry-run Orchestration Library.

Ties together Phase 1A (parse_task_request) and Phase 1B (PlaneClient) into a
single end-to-end dry-run flow that produces a structured report without making
any external API calls.

Flow::

    user_request (str)
      → parse_task_request(user_request, provider)   # Phase 1A
      → TaskDraft.validate()
      → PlaneClient.build_create_payload(task)       # Phase 1B
      → E2EDryRunReport (dataclass → JSON)

No external API calls are made; no Plane API key is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuralplane.ai_parser import (
    LLMProvider,
    ParserInputError,
    ParseError,
    parse_task_request,
)
from neuralplane.models import TaskDraft, ValidationError
from neuralplane.plane_client import PlaneClient, PlaneConfig


# ----------------------------------------------------------------------
# E2EDryRunReport
# ----------------------------------------------------------------------


@dataclass
class E2EDryRunReport:
    """Result of a full E2E dry-run orchestration.

    Attributes
    ----------
    dry_run:
        Always ``True`` for this implementation.
    external_api_calls:
        Always ``0`` — no network calls are made.
    task_draft:
        The parsed and validated TaskDraft as a plain dict.
    plane_endpoint:
        The URL that would be targeted (uses dry-run placeholder values).
    plane_payload:
        The JSON body that would be sent to the Plane API.
    warnings:
        Any non-fatal issues encountered (empty in the happy path).
    """

    dry_run: bool
    external_api_calls: int
    task_draft: dict[str, Any]
    plane_endpoint: str
    plane_payload: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dict for JSON output."""
        return {
            "dry_run": self.dry_run,
            "external_api_calls": self.external_api_calls,
            "task_draft": self.task_draft,
            "plane_endpoint": self.plane_endpoint,
            "plane_payload": self.plane_payload,
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------
# Configuration helpers
# ----------------------------------------------------------------------


def build_dry_run_plane_config(
    *,
    base_url: str = "https://api.plane.so",
    workspace_slug: str = "[dry-run-workspace]",
    project_id: str = "[dry-run-project]",
    state_id: str = "[dry-run-state]",
) -> PlaneConfig:
    """Build a placeholder PlaneConfig for dry-run use.

    Parameters
    ----------
    base_url:
        Base of the Plane API. Defaults to ``https://api.plane.so``.
    workspace_slug:
        Workspace identifier for the URL path.
        Defaults to the placeholder ``[dry-run-workspace]``.
    project_id:
        Target project UUID.  Defaults to ``[dry-run-project]``.
    state_id:
        State UUID for new work items.  Defaults to ``[dry-run-state]``.

    Returns
    -------
    PlaneConfig
        A config that can be used with ``PlaneClient`` but will not
        authenticate against a real Plane instance.
    """
    return PlaneConfig(
        base_url=base_url,
        api_key="[dry-run-no-key]",
        workspace_slug=workspace_slug,
        project_id=project_id,
        default_state_id=state_id,
    )


# ----------------------------------------------------------------------
# TaskDraft serialiser
# ----------------------------------------------------------------------


def task_draft_to_dict(task: TaskDraft) -> dict[str, Any]:
    """Serialise a ``TaskDraft`` to a plain dict.

    Parameters
    ----------
    task:
        A validated ``TaskDraft`` instance.

    Returns
    -------
    dict[str, Any]
        Plain-dict representation suitable for JSON serialisation.
    """
    return {
        "title": task.title,
        "description": task.description,
        "dod": task.dod,
        "priority": task.priority,
        "assignee_suggestions": task.assignee_suggestions,
        "labels": task.labels,
    }


# ----------------------------------------------------------------------
# Orchestration entry-point
# ----------------------------------------------------------------------


def run_e2e_dry_run(
    user_request: str,
    provider: LLMProvider,
    config: PlaneConfig | None = None,
) -> E2EDryRunReport:
    """Run the full E2E dry-run pipeline.

    Orchestrates Phase 1A (``parse_task_request``) → TaskDraft →
    Phase 1B (``PlaneClient.build_create_payload``) and returns a
    structured ``E2EDryRunReport`` without making any network calls.

    Parameters
    ----------
    user_request:
        Natural-language task description.
    provider:
        An ``LLMProvider`` instance (e.g. ``StaticLLMProvider``).
    config:
        Optional ``PlaneConfig``.  If omitted, a dry-run placeholder config
        is used so the function runs without any Plane environment variables.

    Returns
    -------
    E2EDryRunReport
        A report containing the parsed TaskDraft, Plane endpoint, and
        the payload that would be sent.

    Raises
    ------
    ParserInputError
        If ``user_request`` is empty or whitespace-only.
    ParseError
        If the provider's response cannot be parsed as a JSON object.
    ValidationError
        If the parsed TaskDraft fails validation.
    """
    # Use the provided config or fall back to a dry-run placeholder.
    if config is None:
        config = build_dry_run_plane_config()

    # Phase 1A: parse the natural-language request into a TaskDraft.
    task = parse_task_request(user_request, provider)

    # Phase 1B: build the Plane payload (no network call).
    client = PlaneClient(config)
    plane_payload = client.build_create_payload(task)
    plane_endpoint = config.build_work_items_url()

    return E2EDryRunReport(
        dry_run=True,
        external_api_calls=0,
        task_draft=task_draft_to_dict(task),
        plane_endpoint=plane_endpoint,
        plane_payload=plane_payload,
        warnings=[],
    )
