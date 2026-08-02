from __future__ import annotations

import argparse
import json
from pathlib import Path

from .release_audit import audit_release


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="root-teller",
        description="Root-Teller public artifact utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit-release",
        help="verify paper alignment, sanitization, and release structure",
    )
    audit.add_argument("--release-root", type=Path, default=Path.cwd())
    audit.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = audit_release(args.release_root, args.output_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
