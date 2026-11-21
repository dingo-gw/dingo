"""Mark a single IMAP email as seen by UID."""

from __future__ import annotations

import sys

from email_helpers import die, imap_connection, load_config


def mark_email_seen(config_path: str, uid: str) -> None:
    """Connect to IMAP and set the \\Seen flag on the given UID."""
    config = load_config(config_path)
    with imap_connection(config) as conn:
        conn.uid("STORE", uid, "+FLAGS", r"(\Seen)")


def main() -> None:
    if len(sys.argv) < 3:
        die("Usage: mark_email_seen.py <config_path> <uid>")

    config_path, uid = sys.argv[1], sys.argv[2]
    try:
        mark_email_seen(config_path, uid)
    except Exception as e:
        die(f"Failed to mark email {uid} as seen: {e}")


if __name__ == "__main__":
    main()
