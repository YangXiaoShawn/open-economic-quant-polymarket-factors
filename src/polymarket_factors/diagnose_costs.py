from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import backtest
from backtest import WalkForwardBacktest
from config import (
    BACKTEST_LOOKBACK_DAYS,
    BACKTEST_MAX_MARKETS,
    BACKTEST_MIN_AGE_DAYS,
    BACKTEST_N_MARKETS,
)
from data_fetcher import build_price_matrix, fetch_active_markets, fetch_recently_closed_markets


def build_backtest_prices() -> pd.DataFrame:
    active_markets = fetch_active_markets(limit=BACKTEST_N_MARKETS)
    closed_markets = fetch_recently_closed_markets(limit=1000, closed_within_days=365)

    seen = set()
    markets = []
    for m in active_markets + closed_markets:
        lbl = (m.get("question") or "")[:60].strip()
        if lbl and lbl not in seen:
            seen.add(lbl)
            markets.append(m)

    return build_price_matrix(
        markets,
        days=BACKTEST_LOOKBACK_DAYS,
        min_age_days=BACKTEST_MIN_AGE_DAYS,
        max_markets=BACKTEST_MAX_MARKETS,
        bypass_cache=False,
    )


def run_cost_grid(prices: pd.DataFrame, costs: list[float]) -> pd.DataFrame:
    rows = []
    original_tc = backtest.TRANSACTION_COST
    try:
        for tc in costs:
            backtest.TRANSACTION_COST = tc
            result = WalkForwardBacktest(prices=prices).run()
            m = result.metrics or {}
            rows.append(
                {
                    "tc": tc,
                    "n_trades": m.get("n_trades", 0),
                    "total_return_pct": m.get("total_return_pct", 0.0),
                    "ann_return_pct": m.get("ann_return_pct", 0.0),
                    "sharpe": m.get("sharpe_ratio", 0.0),
                    "max_dd_pct": m.get("max_drawdown_pct", 0.0),
                    "win_rate_pct": m.get("win_rate_pct", 0.0),
                    "final_equity": m.get("final_equity", 0.0),
                }
            )
    finally:
        backtest.TRANSACTION_COST = original_tc

    return pd.DataFrame(rows).sort_values("tc")


def main() -> None:
    print("Building/Loading backtest prices...")
    prices = build_backtest_prices()
    print(f"Price matrix: {prices.shape[0]} days x {prices.shape[1]} markets")

    cost_grid = [0.0, 0.0025, 0.005, 0.01, 0.015]
    df = run_cost_grid(prices, cost_grid)

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "cost_sensitivity.csv"
    json_path = out_dir / "cost_sensitivity.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Cost Sensitivity ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
