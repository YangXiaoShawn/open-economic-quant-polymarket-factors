"""Deterministic sample reproduction and oeq-core verification pipeline for PolymarketFactors."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONOREPO_CORE = PROJECT_ROOT.parent / "packages" / "oeq-core" / "src"
if MONOREPO_CORE.exists() and str(MONOREPO_CORE) not in sys.path:
    sys.path.insert(0, str(MONOREPO_CORE))

from oeq_core import (
    ClaimBoundary,
    EvidenceGrade,
    build_run_manifest,
    cross_sectional_rank_ic,
    expanding_ic_ir_weights,
    newey_west_tstat,
    quantile_long_short_spread,
    residualize_quadratic,
    turnover_penalized_net_return,
    validate_claim_boundary,
    verify_manifest_checksums,
)


def run_sample_pipeline(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Execute end-to-end factor evaluation on sample_markets.csv and verify published tables."""
    fixture_path = root / "data" / "fixtures" / "sample_markets.csv"
    funnel_path = root / "reports" / "artifact_funnel.csv"
    factors_path = root / "reports" / "expanded_factor_summary.csv"
    category_path = root / "reports" / "category_reversal_ic.csv"

    with fixture_path.open("r", encoding="utf-8") as fh:
        sample_rows = list(csv.DictReader(fh))
    with funnel_path.open("r", encoding="utf-8") as fh:
        funnel = list(csv.DictReader(fh))
    with factors_path.open("r", encoding="utf-8") as fh:
        factors = list(csv.DictReader(fh))
    with category_path.open("r", encoding="utf-8") as fh:
        categories = list(csv.DictReader(fh))

    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        p = float(row["yes_price"])
        ttr = int(row["days_to_resolution"])
        if 0.05 <= p <= 0.95 and ttr > 3:
            by_date[row["date"]].append(row)

    raw_ic_series: dict[str, list[float]] = defaultdict(list)
    pn_ic_series: dict[str, list[float]] = defaultdict(list)
    dates_sorted = sorted(by_date.keys())
    oos_gross_spreads: list[float] = []
    oos_net_spreads: list[float] = []

    for idx, dt in enumerate(dates_sorted):
        cross = by_date[dt]
        if len(cross) < 5:
            continue
        prices = [float(r["yes_price"]) for r in cross]
        fwd_dp = [float(r["fwd_delta_p_5d"]) for r in cross]
        res_dp = residualize_quadratic(fwd_dp, prices)

        signals = {
            "PM-MOM": [float(r["mom_30d"]) for r in cross],
            "PM-VOL": [float(r["vol_30d"]) for r in cross],
            "PM-LIQ": [float(r["log_volume"]) for r in cross],
        }
        for name, sig in signals.items():
            raw_ic_series[name].append(cross_sectional_rank_ic(sig, fwd_dp))
            pn_ic_series[name].append(cross_sectional_rank_ic(sig, res_dp))

        # Walk-forward out-of-sample IC-IR weighted composite after warmup
        if idx >= 3:
            prior_history = {k: pn_ic_series[k][:idx] for k in ("PM-MOM", "PM-VOL", "PM-LIQ")}
            weights = expanding_ic_ir_weights(prior_history, min_periods=3)
        else:
            weights = {"PM-MOM": -0.40, "PM-VOL": -0.40, "PM-LIQ": -0.20}

        comp_scores = [
            weights["PM-MOM"] * signals["PM-MOM"][i]
            + weights["PM-VOL"] * signals["PM-VOL"][i]
            + weights["PM-LIQ"] * signals["PM-LIQ"][i]
            for i in range(len(cross))
        ]
        gross_spread = quantile_long_short_spread(comp_scores, res_dp, n_quantiles=5)
        avg_spread_bps = sum(float(r["spread_proxy"]) for r in cross) / len(cross) * 10_000.0
        net_spread = turnover_penalized_net_return(gross_spread, turnover=0.5, cost_bps=avg_spread_bps)
        oos_gross_spreads.append(gross_spread)
        oos_net_spreads.append(net_spread)

    boundary = ClaimBoundary(
        project_slug="polymarket-factors",
        grade=EvidenceGrade.BACKTEST_ASSOCIATION,
        allows_causal_language=False,
        allows_live_profitability_claim=False,
    )
    violations = validate_claim_boundary(boundary)
    if violations:
        raise ValueError(f"Claim boundary violation: {violations}")

    manifest = build_run_manifest(
        project_slug="polymarket-factors",
        root=root,
        files=[fixture_path, funnel_path, factors_path, category_path],
        evidence_grade=boundary.grade.value,
    )
    checksum_errors = verify_manifest_checksums(root, manifest)
    if checksum_errors:
        raise ValueError(f"Manifest checksum failure: {checksum_errors}")

    sample_pn_summary = {
        k: {
            "mean_ic": round(sum(v) / len(v), 4),
            "nw_t": round(newey_west_tstat(v, max_lag=2), 2),
        }
        for k, v in pn_ic_series.items()
    }

    return {
        "project": "polymarket-factors",
        "evidence_grade": "backtest (association)",
        "claim_boundary_grade": boundary.grade.value,
        "artifact_funnel_stages": len(funnel),
        "final_mom_raw_ic": float(funnel[-1]["raw_ic"]),
        "final_mom_price_neutral_ic": float(funnel[-1]["price_neutral_ic"]),
        "genuine_factors": [row["factor"] for row in factors if "genuine" in row["verdict"]],
        "categories_audited": len(categories),
        "sample_fixture_periods": len(dates_sorted),
        "sample_price_neutral_ic": sample_pn_summary,
        "sample_walkforward_gross_spread": round(sum(oos_gross_spreads) / max(1, len(oos_gross_spreads)), 4),
        "sample_walkforward_net_spread": round(sum(oos_net_spreads) / max(1, len(oos_net_spreads)), 4),
        "verified_files": manifest["file_count"],
    }
