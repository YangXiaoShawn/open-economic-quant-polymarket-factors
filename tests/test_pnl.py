"""
Unit tests for the binary-asset P&L formula (Fix 1.1).

Core property being tested: P&L uses actual entry prices (shares = size / price),
not the old hard-coded typical_price = 0.5 approximation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pandas as pd
import pytest

from strategy import Trade, Side, PairTradingStrategy
from pair_selector import PairInfo


def _make_pair(hedge_ratio: float = 1.0, spread_std: float = 0.05) -> PairInfo:
    return PairInfo(
        market_a="MKT_A",
        market_b="MKT_B",
        correlation=0.8,
        coint_pvalue=0.01,
        hedge_ratio=hedge_ratio,
        half_life_days=10.0,
        spread_mean=0.0,
        spread_std=spread_std,
    )


def _make_strategy(pair: PairInfo, tc: float = 0.0) -> PairTradingStrategy:
    return PairTradingStrategy(pair, position_size_usd=100.0, transaction_cost=tc)


def _make_trade(side: Side, open_price_a: float, open_price_b: float) -> Trade:
    return Trade(
        pair_key="MKT_A__MKT_B",
        open_date=pd.Timestamp("2024-01-10"),
        open_zscore=-2.5 if side == Side.LONG_SPREAD else 2.5,
        side=side,
        size_usd=100.0,
        open_price_a=open_price_a,
        open_price_b=open_price_b,
    )


def _close_row(price_a: float, price_b: float, zscore: float = 0.0) -> pd.Series:
    return pd.Series({
        "price_a": price_a,
        "price_b": price_b,
        "spread": price_a - price_b,
        "zscore": zscore,
    })


class TestLongSpreadPnl:
    def test_profitable_long_spread(self):
        """Buy A at 0.4, A rises to 0.6; sell B at 0.6, B falls to 0.4. Zero TC."""
        pair = _make_pair(hedge_ratio=1.0)
        strat = _make_strategy(pair, tc=0.0)
        trade = _make_trade(Side.LONG_SPREAD, open_price_a=0.4, open_price_b=0.6)
        trade.close_zscore = 0.0

        close = _close_row(price_a=0.6, price_b=0.4)
        pnl = strat._calc_pnl(trade, close)

        # shares_a = 100/0.4 = 250; gain on A = (0.6-0.4)*250 = 50
        # shares_b = 100/0.6 ≈ 166.67; loss on B = (0.4-0.6)*166.67*1 = -33.33 → net gain 33.33
        shares_a = 100.0 / 0.4
        shares_b = 100.0 / 0.6
        expected = (0.6 - 0.4) * shares_a - (0.4 - 0.6) * shares_b * 1.0
        assert math.isclose(pnl, expected, rel_tol=1e-6)

    def test_symmetric_prices_matches_old_formula(self):
        """When open prices are both 0.5 and hedge_ratio=1, new formula is close to old.
        Old: z_sigma * spread_std / 0.5 * size = z_sigma * spread_std * 200
        New: price_delta * size/0.5 - price_delta * size/0.5 = 0  (if same delta both legs)
        This test confirms behaviour when spread DOES move profitably."""
        pair = _make_pair(hedge_ratio=1.0, spread_std=0.05)
        strat = _make_strategy(pair, tc=0.0)
        trade = _make_trade(Side.LONG_SPREAD, open_price_a=0.5, open_price_b=0.5)
        trade.close_zscore = 0.0

        # Move A up 0.05, B stays: gross = 0.05 * 200 = 10
        close = _close_row(price_a=0.55, price_b=0.5)
        pnl = strat._calc_pnl(trade, close)
        expected = 0.05 * (100.0 / 0.5)
        assert math.isclose(pnl, expected, rel_tol=1e-6)

    def test_extreme_prices_differ_from_half_formula(self):
        """At open_price = 0.9, shares = 100/0.9 ≈ 111, not 100/0.5 = 200.
        New formula must differ from the old formula by roughly 1.8x."""
        pair = _make_pair(hedge_ratio=1.0, spread_std=0.05)
        strat = _make_strategy(pair, tc=0.0)
        trade = _make_trade(Side.LONG_SPREAD, open_price_a=0.9, open_price_b=0.1)
        trade.close_zscore = 0.0

        close = _close_row(price_a=0.95, price_b=0.05)
        new_pnl = strat._calc_pnl(trade, close)

        shares_a = 100.0 / 0.9
        shares_b = 100.0 / 0.1
        expected = (0.95 - 0.9) * shares_a - (0.05 - 0.1) * shares_b * 1.0
        assert math.isclose(new_pnl, expected, rel_tol=1e-6)

        # Old formula: z_sigma ≈ spread change / spread_std
        # Just confirm new != old (price extremity changes the answer)
        old_approx = 0.05 / 0.05 * 0.05 / 0.5 * 100.0  # z=1, spread_std=0.05 old path
        assert abs(new_pnl) != pytest.approx(abs(old_approx), rel=0.01)


class TestTransactionCost:
    def test_zero_pnl_trade_still_incurs_cost(self):
        """When prices don't move, cost should be negative."""
        pair = _make_pair(hedge_ratio=1.0)
        strat = _make_strategy(pair, tc=0.02)
        trade = _make_trade(Side.LONG_SPREAD, open_price_a=0.5, open_price_b=0.5)
        trade.close_zscore = 0.0

        close = _close_row(price_a=0.5, price_b=0.5)
        pnl = strat._calc_pnl(trade, close)
        assert pnl < 0, "No-move trade must lose to transaction costs"

    def test_cost_scales_with_hedge_ratio(self):
        """With hedge_ratio=2, the B leg has 2× notional, so cost must be higher."""
        pair_1 = _make_pair(hedge_ratio=1.0)
        pair_2 = _make_pair(hedge_ratio=2.0)
        strat_1 = _make_strategy(pair_1, tc=0.02)
        strat_2 = _make_strategy(pair_2, tc=0.02)

        trade_1 = _make_trade(Side.LONG_SPREAD, open_price_a=0.5, open_price_b=0.5)
        trade_2 = _make_trade(Side.LONG_SPREAD, open_price_a=0.5, open_price_b=0.5)
        trade_1.close_zscore = trade_2.close_zscore = 0.0

        close = _close_row(price_a=0.5, price_b=0.5)
        pnl_1 = strat_1._calc_pnl(trade_1, close)
        pnl_2 = strat_2._calc_pnl(trade_2, close)
        assert pnl_2 < pnl_1, "Higher hedge_ratio means higher cost, lower P&L"
