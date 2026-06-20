"""NeuralPlane Development Kickoff — CLI entry-point.

Usage
-----
    # Dry-run (no API calls)
    python -m src.neuralplane.cli_kickoff \
        --input examples/task_request.sample.txt \
        --dry-run

    # Live kickoff (Plane issue + GitHub branch)
    python -m src.neuralplane.cli_kickoff \
        --input examples/task_request.sample.txt \
        --provider openai \
        --live

    # Live kickoff, issue only (no branch)
    python -m src.neuralplane.cli_kickoff \
        --input examples/task_request.sample.txt \
        --live \
        --no-branch

    # Override GitHub repo
    python -m src.neuralplane.cli_kickoff \
        --input examples/task_request.sample.txt \
        --live \
        --github-repo MyOrg/MyRepo

    # Custom base branch
    python -m src.neuralplane.cli_kickoff \
        --input examples/task_request.sample.txt \
        --live \
        --github-base-branch develop

Environment variables
---------------------
GITHUB_TOKEN            GitHub personal access token
GITHUB_REPO             Repository in "owner/repo" format
PLANE_API_KEY, etc.     Standard Plane env vars
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralplane.ai_parser import (
    LLMProvider,
    ParserInputError,
    ParseError,
    StaticLLMProvider,
)
from neuralplane.dotenv import load_dotenv
from neuralplane.github_client import GitHubConfig, MissingEnvError as GHMissingEnvError
from neuralplane.kickoff import run_kickoff
from neuralplane.models import ValidationError, WorkContextResult
from neuralplane.plane_client import MissingEnvError as PlaneMissingEnvError, PlaneConfig
from neuralplane.providers.openai_provider import OpenAIProvider


# ----------------------------------------------------------------------
# Default static response
# ----------------------------------------------------------------------

_DEFAULT_STATIC_RESPONSE = json.dumps(
    {
        "title": "建立會員登入 API",
        "description": "建立會員登入端點，支援 bcrypt 密碼驗證與 JWT access token 簽發。",
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
        prog="neuralplane-kickoff",
        description=(
            "Development Kickoff: parse a natural-language task description, "
            "create a Plane issue, and optionally create a GitHub branch. "
            "Use --dry-run for a no-network preview."
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
        "--provider",
        default="static",
        choices=["static", "openai"],
        help="LLM provider: 'static' (no network) or 'openai'.",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Model name for the OpenAI provider.",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="Base URL for the OpenAI provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        metavar="T",
        help="Sampling temperature for the OpenAI provider.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        help="Maximum tokens for the OpenAI provider.",
    )
    parser.add_argument(
        "--static-response",
        metavar="PATH",
        help="Path to a JSON file for the static provider response.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Actually create the Plane issue and GitHub branch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Preview the payload without making any API calls.",
    )
    parser.add_argument(
        "--no-branch",
        action="store_true",
        default=False,
        dest="no_branch",
        help="Create a Plane issue but skip GitHub branch creation.",
    )
    parser.add_argument(
        "--github-repo",
        metavar="REPO",
        help="Override GITHUB_REPO env var (format: owner/repo).",
    )
    parser.add_argument(
        "--github-base-branch",
        default="main",
        metavar="BRANCH",
        help="Base branch for GitHub branch creation (default: main).",
    )
    return parser


# ----------------------------------------------------------------------
# Provider factory
# ----------------------------------------------------------------------


def _build_provider(args: argparse.Namespace) -> LLMProvider:
    if args.provider == "openai":
        if args.static_response:
            print(
                "WARNING: --static-response is ignored when --provider openai.",
                file=sys.stderr,
            )
        try:
            return OpenAIProvider(
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
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
# Helpers
# ----------------------------------------------------------------------


def _fatal(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def _to_output_dict(result: WorkContextResult) -> dict:
    return {
        "overall_success": result.overall_success,
        "plane_success": result.plane_success,
        "plane_issue_key": result.plane_issue_key,
        "plane_issue_url": result.plane_issue_url,
        "plane_issue_id": result.plane_issue_id,
        "plane_error": result.plane_error,
        "github_success": result.github_success,
        "github_branch_name": result.github_branch_name,
        "github_repo": result.github_repo,
        "github_error": result.github_error,
        "checkout_instruction": result.checkout_instruction,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    load_dotenv()
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.live and args.dry_run:
        _fatal(
            "Conflicting flags: --dry-run and --live cannot be used together. "
            "Use one or the other."
        )

    # Load input text
    input_path = Path(args.input).expanduser().resolve()
    try:
        user_request = input_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fatal(f"Input file not found: {input_path}")
    except OSError as exc:
        _fatal(f"Failed to read input file: {exc}")

    # Build provider
    provider = _build_provider(args)

    # Determine mode
    dry_run = args.dry_run or not args.live

    # Build configs
    plane_config: PlaneConfig | None = None
    github_config: GitHubConfig | None = None

    if not dry_run:
        # Plane config
        try:
            plane_config = PlaneConfig.from_env()
        except PlaneMissingEnvError as exc:
            _fatal(str(exc))

        # GitHub config (unless --no-branch)
        if not args.no_branch:
            try:
                github_config = GitHubConfig.from_env()
            except GHMissingEnvError as exc:
                _fatal(str(exc))
        else:
            github_config = None

        # Override repo from CLI if provided
        if args.github_repo and github_config:
            # Create a new config with overridden repo
            github_config = GitHubConfig(
                token=github_config.token,
                repo=args.github_repo,
            )
    else:
        # Dry-run: use minimal configs
        plane_config = PlaneConfig(
            base_url="https://api.plane.so",
            api_key="[dry-run-no-key]",
            workspace_slug="[dry-run-workspace]",
            project_id="[dry-run-project]",
            default_state_id="[dry-run-state]",
        )
        if not args.no_branch:
            github_config = GitHubConfig(
                token="[dry-run-no-token]",
                repo=args.github_repo or "[dry-run-repo]",
            )

    # Run orchestration
    try:
        result = run_kickoff(
            user_request=user_request,
            llm_provider=provider,
            plane_config=plane_config,
            github_config=github_config,
            dry_run=dry_run,
            base_branch=args.github_base_branch,
        )
    except ParserInputError as exc:
        _fatal(f"Invalid input: {exc}")
    except ParseError as exc:
        _fatal(f"Provider response parse error: {exc}")
    except ValidationError as exc:
        _fatal(f"TaskDraft validation failed: {exc}")

    # Emit JSON result
    output_dict = _to_output_dict(result)
    json_output = json.dumps(output_dict, ensure_ascii=False, indent=2)
    print(json_output)

    return 0 if result.overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
