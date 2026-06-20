"""NeuralPlane Phase 2 — Plane Workspace Discovery CLI.

Reads Plane environment variables, calls list_projects() and list_states()
for each project, and prints a human-readable inventory of available projects
and their workflow states.

This is a read-only tool — no work items are created.

Usage
-----
    # Requires PLANE_API_KEY, PLANE_WORKSPACE_SLUG, PLANE_PROJECT_ID, PLANE_DEFAULT_STATE_ID
    python -m src.neuralplane.cli_plane_discover

    # Sample output format:
    # === Workspace: my-team ===
    # Projects:
    #   [proj-uuid-1] my-project (NP)
    #     States:
    #       [state-uuid-1] Backlog     (backlog)
    #       [state-uuid-2] In Progress (in_progress)
    #       [state-uuid-3] Done         (done)

Environment variables (all required)
--------------------------------------
PLANE_BASE_URL          Base URL for the Plane API (default: https://api.plane.so)
PLANE_API_KEY           API key (X-API-Key header value)
PLANE_WORKSPACE_SLUG    Workspace slug used in the URL path
PLANE_PROJECT_ID        Target project UUID (used for listing states)
PLANE_DEFAULT_STATE_ID  Default state UUID (not directly used here, but required)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to the path so the package can be imported when running as a module.
# parents[0] = src/neuralplane/, parents[1] = src/ → NeuralPlane/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralplane.dotenv import load_dotenv
from neuralplane.plane_client import (
    MissingEnvError,
    PlaneClient,
    PlaneConfig,
)


def _fatal(message: str) -> None:
    """Print an error to stderr and exit with code 1."""
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def _indent(text: str, spaces: int = 2) -> str:
    """Return text indented by ``spaces`` spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def _format_states(states: list[dict]) -> str:
    """Format a list of state dicts into a readable string."""
    if not states:
        return "  (no states returned)"
    lines = []
    for s in states:
        sid = s.get("id", "?")
        name = s.get("name", "?")
        group = s.get("group", "?")
        lines.append(f"    [{sid}] {name} ({group})")
    return "\n".join(lines)


def main() -> int:
    load_dotenv()

    # Load configuration (all env vars required for live calls).
    try:
        config = PlaneConfig.from_env()
    except MissingEnvError as exc:
        _fatal(str(exc))

    client = PlaneClient(config)

    # Fetch all projects in the workspace.
    try:
        projects = client.list_projects()
    except RuntimeError as exc:
        _fatal(f"Failed to list projects: {exc}")

    print(f"=== Workspace: {config.workspace_slug} ===")
    print(f"Base URL  : {config.base_url}")
    print(f"Projects  : {len(projects)}")
    print()

    if not projects:
        print("  (no projects returned)")
        return 0

    for proj in projects:
        proj_id = proj.get("id", "?")
        proj_name = proj.get("name", "?")
        proj_identifier = proj.get("identifier", "?")
        print(f"  [{proj_id}] {proj_name} ({proj_identifier})")

        # Only fetch states for the configured project (avoids extra API calls).
        if proj_id == config.project_id:
            try:
                states = client.list_states()
            except RuntimeError as exc:
                print(f"    (failed to list states: {exc})")
                continue
            print("    States:")
            print(_indent(_format_states(states), spaces=4))
        else:
            print("    (states not fetched — not the target project)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
