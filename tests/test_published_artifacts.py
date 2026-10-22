"""Verify published artifact-funnel, out-of-sample factor IC tables, and oeq-core sample pipeline."""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from polymarket_factors import config, run_sample_pipeline
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

    def test_sample_pipeline_and_oeq_core_integration(self) -> None:
        self.assertTrue(config.FIXTURES_DIR.exists())
        result = run_sample_pipeline(root=ROOT)
        self.assertEqual(result["claim_boundary_grade"], "BACKTEST_ASSOCIATION")
        self.assertEqual(result["sample_fixture_periods"], 8)
        self.assertEqual(result["capacity_tiers_audited"], 6)
        self.assertEqual(result["verified_files"], 5)
        pn_ic = result["sample_price_neutral_ic"]
        self.assertLess(pn_ic["PM-MOM"]["mean_ic"], 0.0)
        self.assertLess(pn_ic["PM-VOL"]["mean_ic"], 0.0)
        self.assertGreater(result["sample_walkforward_gross_spread"], result["sample_walkforward_net_spread"])
        cap_curve = result["sample_capacity_curve_net_spread"]
        self.assertGreater(cap_curve["$100"], cap_curve["$1000"])
        self.assertEqual(result["real_clob_observations"], 1810)
        self.assertEqual(result["real_clob_periods"], 82)
        self.assertEqual(result["real_walkforward_oos_periods"], 70)
        self.assertAlmostEqual(result["real_walkforward_oos_ic"], 0.0671, places=4)
        self.assertAlmostEqual(result["real_walkforward_oos_nw_t"], 3.24, places=2)
        self.assertAlmostEqual(result["real_static_oos_ic"], -0.0240, places=4)
        self.assertAlmostEqual(result["real_walkforward_gross_5d_spread_pp"], 0.475, places=3)
        self.assertAlmostEqual(result["real_static_gross_5d_spread_pp"], -0.290, places=3)

    def test_real_clob_factor_ic_and_capacity_tables(self) -> None:
        ic_path = ROOT / "reports" / "real_clob_factor_ic.csv"
        with ic_path.open("r", encoding="utf-8") as fh:
            ic_rows = {r["factor"]: r for r in csv.DictReader(fh)}
        self.assertEqual(len(ic_rows), 7)
        self.assertAlmostEqual(float(ic_rows["PM-SPREAD"]["price_neutral_ic"]), -0.0606, places=4)
        self.assertAlmostEqual(float(ic_rows["PM-SPREAD"]["bh_fdr_q_value"]), 0.0131, places=4)
        self.assertAlmostEqual(float(ic_rows["PM-MOM"]["price_neutral_ic"]), 0.0680, places=4)
        self.assertAlmostEqual(float(ic_rows["PM-EXTR"]["price_neutral_ic"]), -0.0191, places=4)

        cap_path = ROOT / "reports" / "walkforward_capacity_summary.csv"
        with cap_path.open("r", encoding="utf-8") as fh:
            cap_rows = {int(r["order_size_usd"]): r for r in csv.DictReader(fh)}
        self.assertEqual(len(cap_rows), 6)
        self.assertLess(float(cap_rows[100]["walkforward_net_taker_5d_spread_pp"]), 0.0)
        self.assertGreater(float(cap_rows[100]["walkforward_net_maker_5d_spread_pp"]), 0.30)

    def test_cli_summary_runs(self) -> None:
        self.assertEqual(main(["reproduce-sample"]), 0)


if __name__ == "__main__":
    unittest.main()
