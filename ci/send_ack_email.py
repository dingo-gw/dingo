"""Send an acknowledgment email for a CI-triggered commit.

Failure is non-fatal: the CI job still runs even if the ack cannot be sent.
"""

from __future__ import annotations

import smtplib
import sys
from email.mime.text import MIMEText
from typing import Any

from email_helpers import load_config


def send_ack_email(config: dict[str, Any], commit_hash: str) -> None:
    """Send an ACK email to the recipients list."""
    from_addr: str = config["root"]
    recipients: list[str] = config["recipients"]

    msg = MIMEText(
        f"CI job triggered for commit {commit_hash}.\n"
        f"The job is now running.\n"
    )
    msg["Subject"] = f"dingo-ci: ACK commit {commit_hash}"
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)

    with smtplib.SMTP_SSL(config["mailhub"], config["port"]) as server:
        server.login(config["authUser"], config["authPass"])
        server.sendmail(from_addr, recipients, msg.as_string())


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: send_ack_email.py <config_path> <commit_hash>", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(sys.argv[1])
        send_ack_email(config, sys.argv[2])
    except Exception as e:
        # Non-fatal: print error but exit 0
        print(f"Failed to send ACK email: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
