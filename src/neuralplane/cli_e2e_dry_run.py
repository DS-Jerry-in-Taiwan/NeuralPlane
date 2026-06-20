"""NeuralPlane Phase 1D — Single-command E2E Dry-run CLI.

Reads a natural-language task description from a text file, runs it through
the full E2E dry-run pipeline (Phase 1A parse + Phase 1B payload build),
and prints a structured JSON report to stdout.

Usage
-----
    # Static provider (default — no network, no API keys required)
    python -m src.neuralplane.cli_e2e_dry_run \\
        --input examples/task_request.sample.txt

    # OpenAI provider (requires LLM_API_KEY env var)
    LLM_API_KEY=sk-... python -m src.neuralplane.cli_e2e_dry_run \\
        --input examples/task_request.sample.txt \\
        --provider openai

    # Write report to a file
    python -m src.neuralplane.cli_e2e_dry_run \\
        --input examples/task_request.sample.txt \\
        --output /tmp/e2e_dry_run_report.json

    # Use a custom static-response fixture
    python -m src.neuralplane.cli_e2e_dry_run \\
        --input examples/task_request.sample.txt \\
        --static-response examples/task_draft.generated.sample.json

    # Override Plane endpoint components
    python -m src.neuralplane.cli_e2e_dry_run \\
        --input examples/task_request.sample.txt \\
        --plane-base-url https://plane.example.com \\
        --workspace-slug my-workspace \\
        --project-id abc-123 \\
        --state-id def-456

No Plane API key is required for static provider.
No LLM API key is required for static provider.  No external API calls are made
in static mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src/ to the path so the package can be imported when running as a module.
# parents[0] = src/neuralplane/, parents[1] = src/ → NeuralPlane/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralplane.ai_parser import (
    LLMProvider,
    ParserInputError,
    ParseError,
    StaticLLMProvider,
)
from neuralplane.dotenv import load_dotenv
from neuralplane.e2e_dry_run import (
    E2EDryRunReport,
    build_dry_run_plane_config,
    run_e2e_dry_run,
)
from neuralplane.e2e_live import run_e2e_live
from neuralplane.plane_client import MissingEnvError, PlaneConfig
from neuralplane.models import ValidationError
from neuralplane.providers.openai_provider import OpenAIProvider

# ----------------------------------------------------------------------
# Default static response (used when --static-response is not provided).
# Matches the deterministic sample in cli_parse_task_draft.py.
# ----------------------------------------------------------------------

_DEFAULT_STATIC_RESPONSE = json.dumps(
    {
        "title": "建立會員登入 API",
        "description": "建立會員登入端點，支援 bcrypt 密碼驗證、JWT access token 簽發與基本 rate limiting。",
        "dod": [
            "密碼驗證必須使用 bcrypt hash，不可明文比對。",
            "登入成功後必須簽發有效期限為 24 小時的 JWT access token。",
            "必須包含登入成功、密碼錯誤、使用者不存在三種情境的測試。",
            "API 必須實作基本 rate limiting 以防止暴力破解。",
        ],
        "priority": "high",
        "labels": ["auth", "api"],
        "assignee_suggestions": [],
    },
    ensure_ascii=False,
)


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralplane-e2e-dry-run",
        description=(
            "Run the full E2E dry-run pipeline: parse a natural-language "
            "task description into a TaskDraft, build the Plane create payload, "
            "and emit a JSON report.  "
            "Use --provider static (the default) for no-network mode.  "
            "Use --provider openai with LLM_API_KEY set for real LLM calls."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to a text file containing the natural-language task description.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Path to write the JSON report. "
        "If omitted, output is written to stdout only.",
    )
    parser.add_argument(
        "--provider",
        default="static",
        choices=["static", "openai"],
        help="LLM provider to use: 'static' (no network) or 'openai' (OpenAI-compatible API).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Model name for the OpenAI provider "
        "(default: gpt-4o, or LLM_MODEL env var).",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="Base URL for the OpenAI provider "
        "(default: https://api.openai.com/v1, or LLM_BASE_URL env var).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        metavar="T",
        help="Sampling temperature for the OpenAI provider (0.0–2.0, default: 0.3, "
        "or LLM_TEMPERATURE env var).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        help="Maximum tokens to generate for the OpenAI provider "
        "(default: 4096, or LLM_MAX_TOKENS env var).",
    )
    parser.add_argument(
        "--static-response",
        metavar="PATH",
        help="Path to a JSON file containing the static provider response. "
        "If omitted, a built-in deterministic sample response is used. "
        "(static provider only; ignored when --provider openai)",
    )
    parser.add_argument(
        "--plane-base-url",
        default="https://api.plane.so",
        help="Plane API base URL (default: https://api.plane.so).",
    )
    parser.add_argument(
        "--workspace-slug",
        default="[dry-run-workspace]",
        help="Workspace slug for the Plane endpoint URL path "
        "(default: [dry-run-workspace]).",
    )
    parser.add_argument(
        "--project-id",
        default="[dry-run-project]",
        help="Project ID (UUID) for the Plane endpoint URL path "
        "(default: [dry-run-project]).",
    )
    parser.add_argument(
        "--state-id",
        default="[dry-run-state]",
        help="State ID (UUID) used as the default state in the Plane payload "
        "(default: [dry-run-state]).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Actually call the Plane API to create the work item. "
            "Requires PLANE_API_KEY, PLANE_WORKSPACE_SLUG, PLANE_PROJECT_ID, "
            "and PLANE_DEFAULT_STATE_ID to be set in the environment.\n\n"
            "This will CREATE a REAL work item in your Plane project."
        ),
    )
    return parser


# ----------------------------------------------------------------------
# Provider factory
# ----------------------------------------------------------------------


def _build_provider(args: argparse.Namespace) -> LLMProvider:
    """Build an LLMProvider from CLI arguments.

    For the ``openai`` provider, a warning is logged if ``--static-response``
    is also supplied (it is ignored in that mode).
    """
    if args.provider == "openai":
        if args.static_response:
            print(
                "WARNING: --static-response is ignored when --provider openai is used.",
                file=sys.stderr,
            )
        try:
            return OpenAIProvider(
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # catches ConfigurationError et al.
            _fatal(f"Failed to initialise OpenAI provider: {exc}")

    # static provider
    if args.static_response:
        response_path = Path(args.static_response).expanduser().resolve()
        try:
            response_text = response_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            _fatal(f"Static response file not found: {response_path}")
        except OSError as exc:
            _fatal(f"Failed to read static response file: {exc}")
    else:
        response_text = _DEFAULT_STATIC_RESPONSE

    return StaticLLMProvider(response_text)


# ----------------------------------------------------------------------
# Error helpers
# ----------------------------------------------------------------------


def _fatal(message: str) -> None:
    """Print an error to stderr and exit with code 1."""
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    load_dotenv()
    parser = _build_arg_parser()
    args = parser.parse_args()

    # --- Load input text -------------------------------------------------
    input_path = Path(args.input).expanduser().resolve()
    try:
        user_request = input_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fatal(f"Input file not found: {input_path}")
    except OSError as exc:
        _fatal(f"Failed to read input file: {exc}")

    # --- Build provider ---------------------------------------------------
    provider = _build_provider(args)

    # --- LIVE mode: require real env config ------------------------------
    if args.live:
        if args.static_response:
            print(
                "WARNING: --static-response is ignored when --live is used.",
                file=sys.stderr,
            )
        try:
            config = PlaneConfig.from_env()
        except MissingEnvError as exc:
            _fatal(str(exc))

        try:
            result = run_e2e_live(user_request, provider, config)
        except ParserInputError as exc:
            _fatal(f"Invalid input: {exc}")
        except ParseError as exc:
            _fatal(f"Provider response parse error: {exc}")
        except ValidationError as exc:
            _fatal(f"TaskDraft validation failed: {exc}")

        # Emit JSON result.
        output_dict = {
            "success": result.success,
            "id": result.id,
            "sequence_id": result.sequence_id,
            "name": result.name,
            "url": result.url,
            "error_message": result.error_message,
        }
        json_output = json.dumps(output_dict, ensure_ascii=False, indent=2)
        print(json_output)
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            try:
                output_path.write_text(json_output + "\n", encoding="utf-8")
            except OSError as exc:
                _fatal(f"Failed to write output file: {exc}")
        return 0 if result.success else 1

    # --- DRY-RUN mode ----------------------------------------------------
    config = build_dry_run_plane_config(
        base_url=args.plane_base_url,
        workspace_slug=args.workspace_slug,
        project_id=args.project_id,
        state_id=args.state_id,
    )

    try:
        report = run_e2e_dry_run(user_request, provider, config)
    except ParserInputError as exc:
        _fatal(f"Invalid input: {exc}")
    except ParseError as exc:
        _fatal(f"Provider response parse error: {exc}")
    except ValidationError as exc:
        _fatal(f"TaskDraft validation failed: {exc}")

    # --- Serialise output -------------------------------------------------
    report_dict = report.to_dict()
    json_output = json.dumps(report_dict, ensure_ascii=False, indent=2)

    # Write to stdout
    print(json_output)

    # Optionally write to file
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        try:
            output_path.write_text(json_output + "\n", encoding="utf-8")
        except OSError as exc:
            _fatal(f"Failed to write output file: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
