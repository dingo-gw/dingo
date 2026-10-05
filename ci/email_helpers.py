"""Shared helpers for CI email scripts."""

from __future__ import annotations

import imaplib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


def load_config(config_path: str) -> dict[str, Any]:
    """Load the email JSON configuration from disk.

    Raises FileNotFoundError if the file does not exist,
    or json.JSONDecodeError if the file is not valid JSON.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return json.load(f)


@contextmanager
def imap_connection(
    config: dict[str, Any],
) -> Generator[imaplib.IMAP4_SSL, None, None]:
    """Context manager for an authenticated IMAP connection to INBOX.

    Ensures logout is always called, even when an exception occurs.
    """
    imap_cfg = config.get("imap")
    if not imap_cfg:
        raise KeyError("Missing 'imap' section in email config")

    conn = imaplib.IMAP4_SSL(imap_cfg["server"], imap_cfg["port"])
    try:
        conn.login(config["authUser"], config["authPass"])
        conn.select("INBOX")
        yield conn
    finally:
        try:
            conn.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


def die(message: str) -> None:
    """Print an error message to stderr and exit with code 1."""
    print(message, file=sys.stderr)
    sys.exit(1)
