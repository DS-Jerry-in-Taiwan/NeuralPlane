"""CLI entry-point for NeuralPlane Phase 1B — Plane Issue Creator.

Usage
-----
    # Dry-run (no token / no network required)
    python -m src.neuralplane.cli_create_plane_issue \\
        --input examples/task_draft.sample.json \\
        --dry-run

    # Live create (requires env vars set; will NOT run without them)
    export PLANE_BASE_URL="https://api.plane.so"
    export PLANE_API_KEY="plane_api_..."
    export PLANE_WORKSPACE_SLUG="my-workspace"
    export PLANE_PROJECT_ID="<uuid>"
    export PLANE_DEFAULT_STATE_ID="<uuid>"

    python -m src.neuralplane.cli_create_plane_issue \\
        --input examples/task_draft.sample.json \\
        --live

Environment variables
--------------------
PLANE_BASE_URL          Base URL for the Plane API (default: https://api.plane.so)
PLANE_API_KEY           API key (X-API-Key header value)
PLANE_WORKSPACE_SLUG    Workspace slug used in the URL path
PLANE_PROJECT_ID        Target project UUID
PLANE_DEFAULT_STATE_ID  State UUID to assign to newly created work items
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src/ to the path so the package can be imported when running as a module.
# parents[0] = src/neuralplane/, parents[1] = src/ → "from neuralplane.models" resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralplane.models import TaskDraft, ValidationError
from neuralplane.plane_client import (
    MissingEnvError,
    PlaneClient,
    PlaneConfig,
)


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralplane-create-plane-issue",
        description=(
            "Create a Plane work item from a TaskDraft JSON file. "
            "Run with --dry-run to preview the payload without calling the Plane API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to a TaskDraft JSON file.",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        default=False,
        dest="dry_run",
        help=(
            "Build the payload and print it without calling Plane. "
            "This is the default when not specified."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Actually call the Plane API to create the work item. "
            "Requires PLANE_API_KEY, PLANE_WORKSPACE_SLUG, PLANE_PROJECT_ID, "
            "and PLANE_DEFAULT_STATE_ID to be set in the environment."
        ),
    )
    return parser


# ----------------------------------------------------------------------
# Load & validate JSON
# ----------------------------------------------------------------------


def _load_task_draft(path: Path) -> TaskDraft:
    """Load and validate a TaskDraft from a JSON file.

    Raises
    ------
    SystemExit
        If the file cannot be read or parsed, or if validation fails.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fatal(f"Input file not found: {path}")
    except json.JSONDecodeError as exc:
        _fatal(f"Invalid JSON in {path}: {exc}")

    try:
        task = TaskDraft.from_dict(raw)
        task.validate()
    except ValidationError as exc:
        _fatal(f"TaskDraft validation failed: {exc}")

    return task


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------


def _print_payload_summary(
    payload: dict,
    endpoint: str,
    dry_run: bool,
) -> None:
    indent = 2
    payload_json = json.dumps(payload, ensure_ascii=False, indent=indent)
    mode_label = "DRY-RUN" if dry_run else "LIVE"
    print(f"=== NeuralPlane Phase 1B [{mode_label}] ===")
    print(f"Endpoint : {endpoint}")
    print(f"Payload   :\n{payload_json}")
    print("=== End ===")


def _print_result(result) -> None:
    """Print a CreateIssueResult in a human-readable form."""
    if result.success:
        print("=== Plane Create — SUCCESS ===")
        if result.id:
            print(f"  ID           : {result.id}")
        if result.sequence_id is not None:
            print(f"  Sequence ID  : {result.sequence_id}")
        if result.name:
            print(f"  Name         : {result.name}")
        if result.url:
            print(f"  URL          : {result.url}")
    else:
        print("=== Plane Create — FAILED ===")
        if result.error_message:
            print(f"  Error        : {result.error_message}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Determine run mode with clear precedence:
    # 1. If --live is specified alone → live
    # 2. If --dry-run is specified alone → dry-run
    # 3. If neither is specified → dry-run (safe default)
    # 4. If both are specified → fail fast (ambiguous)
    if args.live and args.dry_run:
        _fatal(
            "Conflicting flags: --dry-run and --live cannot be used together. "
            "Use one or the other."
        )

    live_mode = bool(args.live)
    dry_run = not live_mode

    # Load & validate the input JSON (needed for both modes).
    input_path = Path(args.input).expanduser().resolve()
    task = _load_task_draft(input_path)

    if args.live:
        # Live mode — require all env vars
        try:
            config = PlaneConfig.from_env()
        except MissingEnvError as exc:
            _fatal(str(exc))
        client = PlaneClient(config)
        result = client.create_work_item(task, dry_run=False)
        _print_result(result)
        return 0 if result.success else 1

    else:
        # Dry-run mode — no env vars needed
        # We still need a config for the endpoint URL, but we can build
        # a minimal one from defaults / any env vars present (non-fatal).
        try:
            config = PlaneConfig.from_env()
        except MissingEnvError:
            # Fall back to a bare-minimum config so dry-run still works.
            config = PlaneConfig(
                base_url="https://api.plane.so",
                api_key="[dry-run-no-key]",
                workspace_slug="[dry-run-workspace]",
                project_id="[dry-run-project]",
                default_state_id="[dry-run-state]",
            )

        client = PlaneClient(config)
        payload = client.build_create_payload(task)
        endpoint = config.build_work_items_url()

        _print_payload_summary(payload, endpoint, dry_run=True)
        return 0


def _fatal(message: str) -> None:
    """Print an error message and exit with status 1."""
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
