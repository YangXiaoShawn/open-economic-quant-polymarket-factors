"""
Polymarket Pair Trading Strategy -- Main Entry Point

Usage
-----
  python main.py                  # backtest on synthetic data
  python main.py --live           # fetch real Polymarket data (1-year)
  python main.py --combined       # run all 3 strategies (pair + event arb + portfolio arb)
  python main.py --live --combined  # combined 3-strategy backtest on real data
  python main.py --days 365       # custom history length
  python main.py --top 8          # max concurrent pairs
  python main.py --no-charts      # skip chart generation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

import pandas as pd

from config import (
    DEFAULT_LOOKBACK_DAYS,
    MAX_OPEN_PAIRS,
    INITIAL_CAPITAL,
)
from data_fetcher import fetch_active_markets, build_price_matrix, _synthetic_price_matrix
from pair_selector import select_pairs
from backtest import WalkForwardBacktest
from report import generate_full_report, print_report, generate_combined_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Polymarket Pair Trading Strategy & Backtester"
    )
    p.add_argument(
        "--live", action="store_true",
        help="Fetch real Polymarket data (requires internet)",
    )
    p.add_argument(
        "--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help="Number of historical days to use",
    )
    p.add_argument(
        "--top", type=int, default=MAX_OPEN_PAIRS,
        help="Max concurrent pairs",
    )
    p.add_argument(
        "--capital", type=float, default=INITIAL_CAPITAL,
        help="Initial capital in USD",
    )
    p.add_argument(
        "--combined", action="store_true",
        help="Run all three strategies: Pair Trading + Complete Event Arb + Portfolio Arb",
    )
    p.add_argument(
        "--no-charts", dest="no_charts", action="store_true",
        help="Skip chart generation",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_prices(args) -> pd.DataFrame:
    print(f"\n[1] Loading {args.days}-day price data ...")
    t0 = time.time()

    if args.live:
        print("  Fetching market list from Polymarket Gamma API ...")
        markets = fetch_active_markets(limit=500)
        if not markets:
            print("  ! No markets retrieved. Falling back to synthetic data.")
            prices = _synthetic_price_matrix(days=args.days)
        else:
            print(f"  Found {len(markets)} markets with CLOB tokens.")
            print(f"  Fetching individual price histories (this may take ~60s) ...")
            prices = build_price_matrix(markets, days=args.days, min_age_days=20)
    else:
        print("  Using synthetic data (run with --live for real Polymarket data)")
        prices = _synthetic_price_matrix(days=args.days)

    if prices.empty:
        print("  ! Price matrix is empty. Exiting.")
        sys.exit(1)

    print(f"  Price matrix: {prices.shape[0]} days x {prices.shape[1]} markets")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    return prices


def _run_pair_only(args, prices: pd.DataFrame) -> None:
    """Original single-strategy pair trading backtest."""
    # Pair selection preview
    print("\n[2/4] Screening cointegrated pairs on full dataset ...")
    t0 = time.time()
    all_pairs = select_pairs(prices, top_n=30)

    if not all_pairs:
        print("  ! No cointegrated pairs found. Exiting.")
        sys.exit(1)

    print(f"  Found {len(all_pairs)} valid pairs in {time.time() - t0:.1f}s")
    print(f"\n  Top-5 pairs by composite score:")
    print(f"  {'Pair (truncated)':<44} {'Corr':>6}  {'p-val':>6}  {'HL(d)':>5}  {'Score':>7}")
    print(f"  {'-'*44} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}")
    for pair in all_pairs[:5]:
        a_name = pair.market_a[:20]
        b_name = pair.market_b[:20]
        name = f"{a_name} / {b_name}"
        print(
            f"  {name:<44} {pair.correlation:>+6.3f}  "
            f"{pair.coint_pvalue:>6.4f}  {pair.half_life_days:>5.1f}  "
            f"{pair.score:>7.4f}"
        )

    # Walk-forward backtest
    print(f"\n[3/4] Running walk-forward backtest ({args.days} days of data) ...")
    t0 = time.time()
    engine = WalkForwardBacktest(
        prices=prices,
        initial_cap=args.capital,
        top_n_pairs=args.top,
    )
    result = engine.run()
    print(f"  Backtest complete in {time.time() - t0:.1f}s")
    print(f"  Total trades executed: {len(result.trades)}")

    # Report
    print("\n[4/4] Generating report ...")
    if args.no_charts:
        print_report(result)
    else:
        generate_full_report(result)


def _run_combined(args, prices: pd.DataFrame) -> None:
    """Three-strategy combined backtest."""
    from event_data import load_all_event_data
    from combined_backtest import CombinedBacktest

    # Fetch event groups for event arb strategies
    print("\n[2] Fetching event groups for Complete Event Arb + Portfolio Arb ...")
    t0 = time.time()
    if args.live:
        event_groups = load_all_event_data(days=args.days)
        print(f"  Loaded {len(event_groups)} event groups with price history in {time.time() - t0:.1f}s")
    else:
        print("  Skipping real event data (--live not set); event arb strategies will use auto-clustered baskets.")
        event_groups = []

    # Run combined backtest
    print(f"\n[3] Running three-strategy combined backtest ...")
    t0 = time.time()
    engine = CombinedBacktest(
        prices=prices,
        event_groups=event_groups,
        initial_cap=args.capital,
        top_n_pairs=args.top,
    )
    result = engine.run()
    print(f"  Combined backtest complete in {time.time() - t0:.1f}s")
    print(f"  Trades: Pair={len(result.pair_trades)}, PortArb={len(result.port_arb_trades)}")

    # Report
    print("\n[4] Generating combined report ...")
    if args.no_charts:
        from report import print_combined_report
        print_combined_report(result)
    else:
        generate_combined_report(result)


def main() -> None:
    args = parse_args()

    mode_str = "LIVE Polymarket Data" if args.live else "Synthetic Data"
    strategy_str = "THREE-STRATEGY COMBINED" if args.combined else "PAIR TRADING"

    print("\n" + "=" * 64)
    print(f"  POLYMARKET {strategy_str} STRATEGY")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode  : {mode_str}")
    print("=" * 64)

    prices = _load_prices(args)

    if args.combined:
        _run_combined(args, prices)
    else:
        _run_pair_only(args, prices)

    print("\nDone.")


if __name__ == "__main__":
    main()
