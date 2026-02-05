"""Fetch the first unread IMAP email whose subject matches 'commit <hex_sha> [--debug]'.

Prints "UID COMMIT_HASH [--debug]" to stdout on match, nothing otherwise.
Prints verbose info about all checked emails to stderr.
Exit 0 on success (match or no match), exit 1 on error.
"""

from __future__ import annotations

import email
import email.header
import re
import sys
from typing import Any

from email_helpers import die, imap_connection, load_config

SUBJECT_RE = re.compile(r"^commit\s+([0-9a-fA-F]{7,40})(\s+--debug)?$", re.IGNORECASE)


def decode_subject(raw_subject: str) -> str:
    """Decode a possibly MIME-encoded email subject to plain text."""
    decoded_parts = email.header.decode_header(raw_subject)
    return "".join(
        part.decode(enc or "utf-8") if isinstance(part, bytes) else part
        for part, enc in decoded_parts
    )


def fetch_matching_email(config: dict[str, Any]) -> tuple[str | None, str | None, bool, list[str], list[str]]:
    """Search INBOX for an unread 'commit <sha> [--debug]' email.

    Returns (uid, commit_hash, debug_mode, verbose_lines, checked_uids).
    If no match, uid and commit_hash are None.
    checked_uids contains all UIDs that were checked (to be marked as seen).
    """
    verbose_lines: list[str] = []
    checked_uids: list[str] = []

    with imap_connection(config) as conn:
        status, data = conn.uid("SEARCH", "UNSEEN")
        if status != "OK":
            raise RuntimeError("IMAP SEARCH failed")

        uid_list = data[0].split()  # ascending order = oldest first

        if not uid_list:
            verbose_lines.append("No unread emails found")
        else:
            verbose_lines.append(f"Found {len(uid_list)} unread email(s), checking...")

        for uid_bytes in uid_list:
            uid = uid_bytes.decode()
            checked_uids.append(uid)
            # BODY.PEEK does not set \Seen flag
            status, msg_data = conn.uid(
                "FETCH", uid, "(BODY.PEEK[HEADER.FIELDS (SUBJECT)])"
            )
            if status != "OK":
                verbose_lines.append(f"  UID {uid}: failed to fetch subject")
                continue

            raw_header: bytes = msg_data[0][1]
            msg = email.message_from_bytes(raw_header)
            subject = decode_subject(msg.get("Subject", "").strip())

            match = SUBJECT_RE.match(subject.strip())
            if match:
                debug_mode = match.group(2) is not None
                debug_str = " [--debug]" if debug_mode else ""
                verbose_lines.append(
                    f"  UID {uid}: \"{subject}\" -> MATCH, triggering job{debug_str}"
                )
                # Mark all checked emails as seen (including the match)
                for mark_uid in checked_uids:
                    conn.uid("STORE", mark_uid, "+FLAGS", "\\Seen")
                    verbose_lines.append(f"  UID {mark_uid}: marked as seen")
                return uid, match.group(1), debug_mode, verbose_lines, checked_uids
            else:
                verbose_lines.append(
                    f"  UID {uid}: \"{subject}\" -> not supported, skipping"
                )

        # No match found - mark all checked emails as seen
        for mark_uid in checked_uids:
            conn.uid("STORE", mark_uid, "+FLAGS", "\\Seen")
            verbose_lines.append(f"  UID {mark_uid}: marked as seen")

    # Print verbose info when no match
    for line in verbose_lines:
        print(line, file=sys.stderr)

    return None, None, False, verbose_lines, checked_uids


def main() -> None:
    if len(sys.argv) < 2:
        die("Usage: fetch_matching_email.py <config_path>")

    try:
        config = load_config(sys.argv[1])
        result = fetch_matching_email(config)
    except Exception as e:
        die(f"IMAP error: {e}")

    uid, commit_hash, debug_mode, verbose_lines, _ = result
    # Print verbose info to stderr
    for line in verbose_lines:
        print(line, file=sys.stderr)
    # Print match result to stdout (if any)
    if uid and commit_hash:
        if debug_mode:
            print(f"{uid} {commit_hash} --debug")
        else:
            print(f"{uid} {commit_hash}")


if __name__ == "__main__":
    main()
