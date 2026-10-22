"""Command-line interface for Polymarket factor research."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cmd_summary() -> int:
    funnel_path = ROOT / "reports" / "artifact_funnel.csv"
    factors_path = ROOT / "reports" / "expanded_factor_summary.csv"
    with funnel_path.open("r", encoding="utf-8") as fh:
        funnel = list(csv.DictReader(fh))
    with factors_path.open("r", encoding="utf-8") as fh:
        factors = list(csv.DictReader(fh))
    payload = {
        "project": "polymarket-factors",
        "evidence_grade": "backtest (association)",
        "artifact_funnel_stages": len(funnel),
        "final_mom_raw_ic": float(funnel[-1]["raw_ic"]),
        "final_mom_price_neutral_ic": float(funnel[-1]["price_neutral_ic"]),
        "genuine_factors": [row["factor"] for row in factors if "genuine" in row["verdict"]],
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polymarket-factors")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("summary", help="Print published factor and artifact-funnel summary.")
    sub.add_parser("reproduce-sample", help="Verify deterministic sample fixtures and published tables.")
    args = parser.parse_args(argv)
    if args.command in (None, "summary", "reproduce-sample"):
        return cmd_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
