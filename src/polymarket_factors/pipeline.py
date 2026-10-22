"""Deterministic sample reproduction and real CLOB validation pipeline for PolymarketFactors."""
from __future__ import annotations

import csv
from collections import defaultdict
import math
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


def _zscore(vals: list[float]) -> list[float]:
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)) or 1.0
    return [(x - mean) / std for x in vals]


def evaluate_real_clob_panel(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Evaluate 7-factor ICs and walk-forward IC-IR composite on real Polymarket CLOB data."""
    real_path = root / "data" / "fixtures" / "real_polymarket_panel.csv"
    with real_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if 0.05 <= float(row["yes_price"]) <= 0.95 and int(row["days_to_resolution"]) > 3:
            by_date[row["date"]].append(row)

    dates_sorted = [d for d in sorted(by_date.keys()) if len(by_date[d]) >= 5]
    pn_ic_series: dict[str, list[float]] = defaultdict(list)
    oos_wf_ics: list[float] = []
    oos_st_ics: list[float] = []
    oos_wf_spreads: list[float] = []
    oos_st_spreads: list[float] = []

    comp_keys = ("PM-MOM", "PM-VOL", "PM-LIQ", "PM-SPREAD")
    for idx, dt in enumerate(dates_sorted):
        cross = by_date[dt]
        prices = [float(r["yes_price"]) for r in cross]
        fwd_dp = [float(r["fwd_delta_p_5d"]) for r in cross]
        res_dp = residualize_quadratic(fwd_dp, prices)

        signals = {
            "PM-MOM": _zscore([float(r["mom_30d"]) for r in cross]),
            "PM-VOL": _zscore([float(r["vol_30d"]) for r in cross]),
            "PM-LIQ": _zscore([float(r["log_volume"]) for r in cross]),
            "PM-SPREAD": _zscore([float(r["spread_proxy"]) for r in cross]),
        }
        if idx >= 12:
            prior = {k: pn_ic_series[k][:idx] for k in comp_keys}
            w = expanding_ic_ir_weights(prior, min_periods=12)
            comp = [sum(w[k] * signals[k][i] for k in comp_keys) for i in range(len(cross))]
            static = [
                -0.4 * signals["PM-MOM"][i] - 0.4 * signals["PM-VOL"][i] - 0.2 * signals["PM-LIQ"][i]
                for i in range(len(cross))
            ]
            oos_wf_ics.append(cross_sectional_rank_ic(comp, res_dp))
            oos_st_ics.append(cross_sectional_rank_ic(static, res_dp))
            oos_wf_spreads.append(quantile_long_short_spread(comp, res_dp, n_quantiles=3))
            oos_st_spreads.append(quantile_long_short_spread(static, res_dp, n_quantiles=3))

        for k, sig in signals.items():
            pn_ic_series[k].append(cross_sectional_rank_ic(sig, res_dp))

    return {
        "real_clob_observations": len(rows),
        "real_clob_periods": len(dates_sorted),
        "real_walkforward_oos_periods": len(oos_wf_ics),
        "real_walkforward_oos_ic": round(sum(oos_wf_ics) / len(oos_wf_ics), 4),
        "real_walkforward_oos_nw_t": round(newey_west_tstat(oos_wf_ics, max_lag=3), 2),
        "real_static_oos_ic": round(sum(oos_st_ics) / len(oos_st_ics), 4),
        "real_static_oos_nw_t": round(newey_west_tstat(oos_st_ics, max_lag=3), 2),
        "real_walkforward_gross_5d_spread_pp": round(sum(oos_wf_spreads) / len(oos_wf_spreads) * 100.0, 3),
        "real_static_gross_5d_spread_pp": round(sum(oos_st_spreads) / len(oos_st_spreads) * 100.0, 3),
    }


def run_sample_pipeline(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Execute end-to-end factor evaluation on sample_markets.csv and real_polymarket_panel.csv."""
    fixture_path = root / "data" / "fixtures" / "sample_markets.csv"
    funnel_path = root / "reports" / "artifact_funnel.csv"
    factors_path = root / "reports" / "expanded_factor_summary.csv"
    category_path = root / "reports" / "category_reversal_ic.csv"
    capacity_path = root / "reports" / "walkforward_capacity_summary.csv"

    with fixture_path.open("r", encoding="utf-8") as fh:
        sample_rows = list(csv.DictReader(fh))
    with funnel_path.open("r", encoding="utf-8") as fh:
        funnel = list(csv.DictReader(fh))
    with factors_path.open("r", encoding="utf-8") as fh:
        factors = list(csv.DictReader(fh))
    with category_path.open("r", encoding="utf-8") as fh:
        categories = list(csv.DictReader(fh))
    with capacity_path.open("r", encoding="utf-8") as fh:
        capacity_rows = list(csv.DictReader(fh))

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
        files=[fixture_path, funnel_path, factors_path, category_path, capacity_path],
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

    mean_gross = sum(oos_gross_spreads) / max(1, len(oos_gross_spreads))
    capacity_curve_net = {}
    for order_usd in (50, 100, 200, 500, 1000):
        eff_bps = 140.0 * (1.0 + 1.25 * math.sqrt(order_usd / 1500.0))
        capacity_curve_net[f"${order_usd}"] = round(
            turnover_penalized_net_return(mean_gross, turnover=0.5, cost_bps=eff_bps), 4
        )

    real_clob_summary = evaluate_real_clob_panel(root=root)

    return {
        "project": "polymarket-factors",
        "evidence_grade": "backtest (association)",
        "claim_boundary_grade": boundary.grade.value,
        "artifact_funnel_stages": len(funnel),
        "final_mom_raw_ic": float(funnel[-1]["raw_ic"]),
        "final_mom_price_neutral_ic": float(funnel[-1]["price_neutral_ic"]),
        "genuine_factors": [row["factor"] for row in factors if "genuine" in row["verdict"]],
        "categories_audited": len(categories),
        "capacity_tiers_audited": len(capacity_rows),
        "sample_fixture_periods": len(dates_sorted),
        "sample_price_neutral_ic": sample_pn_summary,
        "sample_walkforward_gross_spread": round(mean_gross, 4),
        "sample_walkforward_net_spread": round(sum(oos_net_spreads) / max(1, len(oos_net_spreads)), 4),
        "sample_capacity_curve_net_spread": capacity_curve_net,
        "verified_files": manifest["file_count"],
        **real_clob_summary,
    }
