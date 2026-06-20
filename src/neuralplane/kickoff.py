"""NeuralPlane Development Kickoff — Orchestration.

Orchestrates:
  1. Phase 1A: ``parse_task_request`` → ``TaskDraft``
  2. Phase 1B: ``PlaneClient.create_work_item`` → Plane issue
  3. GitHub: ``GitHubClient.create_branch`` → branch

Flow::

    user_request (str)
      → parse_task_request(user_request, provider)
      → PlaneClient.create_work_item(task, dry_run=False)
      → GitHubClient.create_branch(new_branch, base_branch)
      → WorkContextResult
"""

from __future__ import annotations

from neuralplane.ai_parser import (
    LLMProvider,
    ParserInputError,
    ParseError,
    parse_task_request,
)
from neuralplane.github_client import GitHubClient, GitHubConfig
from neuralplane.models import (
    CreateIssueResult,
    TaskDraft,
    ValidationError,
    WorkContextResult,
    generate_branch_name,
)
from neuralplane.plane_client import PlaneClient, PlaneConfig


def run_kickoff(
    user_request: str,
    llm_provider: LLMProvider,
    plane_config: PlaneConfig,
    github_config: GitHubConfig | None = None,
    *,
    dry_run: bool = False,
    base_branch: str = "main",
) -> WorkContextResult:
    """Run the full Development Kickoff pipeline.

    Parameters
    ----------
    user_request:
        Natural-language task description.
    llm_provider:
        An ``LLMProvider`` instance.
    plane_config:
        A populated ``PlaneConfig``.
    github_config:
        A ``GitHubConfig`` or ``None`` (skip branch creation).
    dry_run:
        If True, skip all live API calls and return a dry-run result.
        Default False.
    base_branch:
        Base branch name for GitHub branch creation (default ``"main"``).

    Returns
    -------
    WorkContextResult
        Result with partial-success reporting.

    Raises
    ------
    ParserInputError
        If ``user_request`` is empty.
    ParseError
        If the provider response cannot be parsed.
    ValidationError
        If the parsed ``TaskDraft`` fails validation.
    """
    # --- Step 1: Parse -------------------------------------------------
    task: TaskDraft = parse_task_request(user_request, llm_provider)

    # --- Step 2: Dry-run shortcut -------------------------------------
    if dry_run:
        return WorkContextResult(
            plane_success=True,
            plane_issue_key="[dry-run]",
            plane_issue_url=None,
            plane_issue_id=None,
            plane_error=None,
            github_success=True if github_config else False,
            github_branch_name="[dry-run-branch]" if github_config else None,
            github_repo=github_config.repo if github_config else None,
            github_error=None,
            overall_success=True,
            checkout_instruction="git checkout [dry-run-branch]"
            if github_config
            else None,
        )

    # --- Step 3: Create Plane issue -----------------------------------
    client = PlaneClient(plane_config)
    plane_result: CreateIssueResult = client.create_work_item(task, dry_run=False)

    if not plane_result.success:
        return WorkContextResult(
            plane_success=False,
            plane_issue_key=None,
            plane_issue_url=None,
            plane_issue_id=None,
            plane_error=plane_result.error_message,
            github_success=False,
            github_branch_name=None,
            github_repo=None,
            github_error=None,
            overall_success=False,
            checkout_instruction=None,
        )

    # --- Step 4: Build issue key and branch name ----------------------
    issue_key = _build_issue_key(
        plane_result.sequence_id, plane_result.id
    )
    branch_name: str | None = None
    if issue_key and github_config:
        branch_name = generate_branch_name(issue_key, task.trimmed_title())

    # --- Step 5: Create GitHub branch (optional) ----------------------
    github_success = False
    github_error: str | None = None

    if branch_name and github_config:
        gh_client = GitHubClient(github_config)
        try:
            gh_client.create_branch(branch_name, base_branch=base_branch)
            github_success = True
        except RuntimeError as exc:
            github_error = str(exc)

    # --- Step 6: Assemble result --------------------------------------
    overall = plane_result.success and github_success
    checkout: str | None = None
    if github_success and branch_name:
        checkout = f"git checkout {branch_name}"

    return WorkContextResult(
        plane_success=plane_result.success,
        plane_issue_key=issue_key,
        plane_issue_url=plane_result.url,
        plane_issue_id=plane_result.id,
        plane_error=None,
        github_success=github_success,
        github_branch_name=branch_name,
        github_repo=github_config.repo if github_config else None,
        github_error=github_error,
        overall_success=overall,
        checkout_instruction=checkout,
    )


def _build_issue_key(
    sequence_id: int | None,
    issue_id: str | None,
) -> str | None:
    """Build a human-readable issue key from the Plane response.

    If ``sequence_id`` is available, returns ``NP-{sequence_id}``.
    Otherwise returns ``None``.
    """
    if sequence_id is not None:
        return f"NP-{sequence_id}"
    return None
