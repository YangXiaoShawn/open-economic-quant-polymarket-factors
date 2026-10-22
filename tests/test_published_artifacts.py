"""Verify published artifact-funnel and out-of-sample factor IC tables."""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "polymarket_factors"))

from polymarket_factors.cli import main


class TestPolymarketPublishedArtifacts(unittest.TestCase):
    def test_artifact_funnel_monotonic_magnitude(self) -> None:
        path = ROOT / "reports" / "artifact_funnel.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 5)
        raw_ics = [abs(float(r["raw_ic"])) for r in rows]
        self.assertEqual(raw_ics, [0.440, 0.340, 0.105, 0.096, 0.054])
        self.assertAlmostEqual(float(rows[-1]["price_neutral_ic"]), -0.032, places=4)

    def test_seven_factor_oos_summary(self) -> None:
        path = ROOT / "reports" / "expanded_factor_summary.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = {r["factor"]: r for r in csv.DictReader(fh)}
        self.assertEqual(len(rows), 7)
        self.assertAlmostEqual(float(rows["PM-LIQ"]["price_neutral_ic"]), -0.072, places=4)
        self.assertAlmostEqual(float(rows["PM-VOL"]["price_neutral_ic"]), -0.069, places=4)
        self.assertAlmostEqual(float(rows["PM-MOM"]["price_neutral_ic"]), -0.032, places=4)

    def test_cli_summary_runs(self) -> None:
        self.assertEqual(main(["summary"]), 0)


if __name__ == "__main__":
    unittest.main()
