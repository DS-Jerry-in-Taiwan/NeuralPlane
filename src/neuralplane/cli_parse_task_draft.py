"""NeuralPlane Phase 1A — Parse-only CLI entry-point.

Reads a natural-language task description from a text file, runs it through
the AI parser using the injected LLM provider (static or openai), and prints
the resulting TaskDraft JSON to stdout.  No Plane API calls are made.

Usage
-----
    # Static provider (default — no network, no API keys required)
    python -m src.neuralplane.cli_parse_task_draft \\
        --input examples/task_request.sample.txt

    # OpenAI provider (requires LLM_API_KEY env var)
    LLM_API_KEY=sk-... python -m src.neuralplane.cli_parse_task_draft \\
        --input examples/task_request.sample.txt \\
        --provider openai

    # Write output to file
    python -m src.neuralplane.cli_parse_task_draft \\
        --input examples/task_request.sample.txt \\
        --output /tmp/task_draft.generated.json

    # Use a custom static-response fixture (static provider only)
    python -m src.neuralplane.cli_parse_task_draft \\
        --input examples/task_request.sample.txt \\
        --provider static \\
        --static-response examples/provider_response.sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src/ to the path so the package can be imported when running as a module.
# parents[1] = src/neuralplane/ → src/ → NeuralPlane/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralplane.ai_parser import (
    LLMProvider,
    ParserInputError,
    ParseError,
    StaticLLMProvider,
    parse_task_request,
)
from neuralplane.dotenv import load_dotenv
from neuralplane.models import TaskDraft, ValidationError
from neuralplane.providers.openai_provider import OpenAIProvider

# ----------------------------------------------------------------------
# Default static response (used when --static-response is not provided)
# This lets the CLI run end-to-end without any fixture file.
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
        prog="neuralplane-parse-task-draft",
        description=(
            "Parse a natural-language task description into a TaskDraft JSON "
            "using an injectable LLM provider ('static' or 'openai'). "
            "Use --provider static (the default) for no-network mode. "
            "Use --provider openai with LLM_API_KEY set for real LLM calls. "
            "--dry-run is the default and only mode (no Plane writes)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="PATH",
        help="Path to a text file containing the natural-language task description.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Path to write the TaskDraft JSON output. "
        "If omitted, output is written to stdout only.",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        default=True,
        dest="dry_run",
        help="Dry-run mode (default): parse and print JSON, do not call Plane. "
        "This flag exists for explicit clarity and is the default regardless.",
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
# TaskDraft serialiser
# ----------------------------------------------------------------------


def _task_draft_to_dict(task: TaskDraft) -> dict:
    """Serialise a TaskDraft to a plain dict for JSON output."""
    return {
        "title": task.title,
        "description": task.description,
        "dod": task.dod,
        "priority": task.priority,
        "assignee_suggestions": task.assignee_suggestions,
        "labels": task.labels,
    }


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

    # --- Parse ------------------------------------------------------------
    try:
        task = parse_task_request(user_request, provider)
    except ParserInputError as exc:
        _fatal(f"Invalid input: {exc}")
    except ParseError as exc:
        _fatal(f"Provider response parse error: {exc}")
    except ValidationError as exc:
        _fatal(f"TaskDraft validation failed: {exc}")

    # --- Serialise output -------------------------------------------------
    output_dict = _task_draft_to_dict(task)
    json_output = json.dumps(output_dict, ensure_ascii=False, indent=2)

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
