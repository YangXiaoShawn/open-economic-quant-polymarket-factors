"""Command-line interface for Polymarket factor research."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polymarket_factors.pipeline import PROJECT_ROOT, run_sample_pipeline


def cmd_summary(root: Path = PROJECT_ROOT) -> int:
    payload = run_sample_pipeline(root=root)
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polymarket-factors")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("summary", help="Print published factor and artifact-funnel summary.")
    sub.add_parser("reproduce-sample", help="Run deterministic sample pipeline and verify published tables.")
    args = parser.parse_args(argv)
    if args.command in (None, "summary", "reproduce-sample"):
        return cmd_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
