"""A lightweight .env file loader — stdlib only, ~15 lines of logic."""

from __future__ import annotations

import os


def load_dotenv(path: str = ".env") -> None:
    """Load environment variables from a .env file into os.environ.

    Uses ``setdefault`` so existing environment variables (e.g. from
    shell ``export``) take precedence over values in the file.

    Silently ignores missing files.  Skips blank lines and ``#`` comments.

    Parameters
    ----------
    path:
        Path to the .env file (default: ``".env"`` in CWD).
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip surrounding quotes if present
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                else:
                    # Strip inline comments from unquoted values (standard .env behaviour).
                    # Handles both ``value # comment`` and ``value# comment``.
                    if value.startswith("#"):
                        value = ""
                    else:
                        for delim in (" #", "# "):
                            pos = value.find(delim)
                            if pos != -1:
                                value = value[:pos].rstrip()
                                break
                os.environ.setdefault(key, value)
    except FileNotFoundError:
        pass
