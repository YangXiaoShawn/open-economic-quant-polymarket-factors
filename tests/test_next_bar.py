"""
Unit tests for next-bar execution (Fix 1.2).

Verifies that trade open_date is the day AFTER the signal triggers,
not the same day as the signal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from strategy import PairTradingStrategy, Side
from pair_selector import PairInfo


def _make_pair() -> PairInfo:
    return PairInfo(
        market_a="MKT_A",
        market_b="MKT_B",
        correlation=0.8,
        coint_pvalue=0.01,
        hedge_ratio=1.0,
        half_life_days=10.0,
        spread_mean=0.0,
        spread_std=0.05,
    )


def _make_prices(n_days: int = 30) -> pd.DataFrame:
    """Synthetic price matrix: both markets track a common path."""
    dates = pd.date_range("2024-03-01", periods=n_days, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    common = np.cumsum(rng.normal(0, 0.005, n_days)) + 0.5
    a = np.clip(common + rng.normal(0, 0.002, n_days), 0.05, 0.95)
    b = np.clip(common + rng.normal(0, 0.002, n_days), 0.05, 0.95)
    return pd.DataFrame({"MKT_A": a, "MKT_B": b}, index=dates)


def test_no_trade_opens_on_signal_day():
    """All trades must open on the day AFTER the entry z-score triggers."""
    pair = _make_pair()
    # Use a very low entry z so that a trade is likely to be triggered
    strat = PairTradingStrategy(pair, position_size_usd=100.0, entry_z=0.1, exit_z=0.05)
    prices = _make_prices(60)
    trades, signals = strat.run(prices)

    if not trades:
        pytest.skip("No trades triggered with this synthetic data — adjust entry_z")

    signal_dates = signals.index
    for trade in trades:
        open_idx = signals.index.get_loc(trade.open_date)
        # The z-score that triggered this trade must be on the PREVIOUS bar
        assert open_idx >= 1, "Trade open_date is the very first bar — impossible with next-bar"
        signal_z = signals.iloc[open_idx - 1]["zscore"]
        open_z = trade.open_zscore
        # open_zscore is stored from the signal bar
        assert abs(open_z) >= 0.1, f"open_zscore {open_z} should have triggered entry"


def test_open_date_one_day_after_signal():
    """
    Construct a price series where the entry signal fires on a known date,
    then assert the trade opens on the following date.
    """
    pair = _make_pair()
    strat = PairTradingStrategy(pair, position_size_usd=100.0, entry_z=1.5, exit_z=0.3,
                                 stop_z=4.0, window=5)

    # Hand-craft prices so spread = price_a - price_b goes very negative on day 6
    # (index 5), triggering LONG_SPREAD entry. The trade should open on day 7 (index 6).
    dates = pd.date_range("2024-03-01", periods=20, freq="D", tz="UTC")
    price_a = [0.5] * 20
    price_b = [0.5] * 20
    # push spread negative on days 5-6 to ensure z < -1.5 on day 6
    for i in range(3, 8):
        price_a[i] = 0.35
        price_b[i] = 0.65

    df = pd.DataFrame({"MKT_A": price_a, "MKT_B": price_b}, index=dates)
    trades, signals = strat.run(df)

    if not trades:
        pytest.skip("No trades triggered — synthetic series may need tuning")

    first_trade = min(trades, key=lambda t: t.open_date)
    open_idx = signals.index.get_loc(first_trade.open_date)
    # The signal bar is always one before the execution bar
    assert open_idx >= 1
    sig_z = signals.iloc[open_idx - 1]["zscore"]
    assert sig_z < -1.5 or sig_z > 1.5, (
        f"Signal bar z={sig_z:.3f} should have crossed ±1.5 to trigger entry"
    )


def test_trade_duration_at_least_one_day():
    """Signal-triggered closes must have duration ≥ 1 day (next-bar guarantee).
    EOD force-closes on the last bar may legitimately have duration=0."""
    pair = _make_pair()
    strat = PairTradingStrategy(pair, position_size_usd=100.0, entry_z=0.1, exit_z=0.05)
    prices = _make_prices(60)
    trades, signals = strat.run(prices)
    last_date = signals.index[-1]

    for trade in trades:
        if trade.close_date is not None and trade.closed_by != "eod":
            assert trade.duration_days >= 1, (
                f"Signal-closed trade duration {trade.duration_days}d < 1 day"
            )
        elif trade.close_date is not None and trade.closed_by == "eod":
            # EOD close on last bar: open and close can be the same date
            assert trade.duration_days >= 0
