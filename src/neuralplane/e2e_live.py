"""NeuralPlane Phase 2 — E2E Live Orchestration Library.

Orchestrates Phase 1A (parse_task_request) → TaskDraft → validate →
Phase 1B (PlaneClient.create_work_item with dry_run=False) to create
a real Plane work item from a natural-language request.

Flow::

    user_request (str)
      → parse_task_request(user_request, provider)   # Phase 1A
      → TaskDraft.validate()
      → PlaneClient.create_work_item(task, dry_run=False)  # Phase 1B
      → CreateIssueResult

Requires PLANE_API_KEY and related env vars to be set.
"""

from __future__ import annotations

from neuralplane.ai_parser import (
    LLMProvider,
    ParserInputError,
    ParseError,
    parse_task_request,
)
from neuralplane.models import CreateIssueResult, TaskDraft, ValidationError
from neuralplane.plane_client import PlaneClient, PlaneConfig


def run_e2e_live(
    user_request: str,
    provider: LLMProvider,
    config: PlaneConfig,
) -> CreateIssueResult:
    """Run the full E2E pipeline and create a real Plane work item.

    Orchestrates Phase 1A (``parse_task_request``) → TaskDraft → validate →
    Phase 1B (``PlaneClient.create_work_item`` with ``dry_run=False``).

    Parameters
    ----------
    user_request:
        Natural-language task description.
    provider:
        An ``LLMProvider`` instance (e.g. ``OpenAIProvider``).
    config:
        A populated ``PlaneConfig`` (must come from ``PlaneConfig.from_env()``;
        missing env vars raise ``MissingEnvError`` before any network call).

    Returns
    -------
    CreateIssueResult
        Server response from the Plane create endpoint.

    Raises
    ------
    ParserInputError
        If ``user_request`` is empty or whitespace-only.
    ParseError
        If the provider's response cannot be parsed as a JSON object.
    ValidationError
        If the parsed TaskDraft fails validation.
    RuntimeError
        If the Plane API returns an error HTTP status.
    """
    # Phase 1A: parse the natural-language request into a TaskDraft.
    task = parse_task_request(user_request, provider)

    # Phase 1B: validate + create (live — calls Plane API).
    client = PlaneClient(config)
    return client.create_work_item(task, dry_run=False)
