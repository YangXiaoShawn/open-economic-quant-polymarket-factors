"""
Unit tests for PortfolioArbStrategy P&L — binary-token economics.

SHORT on Polymarket = buying NO tokens at (1 - p), so the SHORT return
denominator must be (1 - p_open), not p_open.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pandas as pd
import pytest

from portfolio_arb import PortfolioArbStrategy, PortfolioArbTrade


def _make_trade(direction: str, open_price: float) -> PortfolioArbTrade:
    return PortfolioArbTrade(
        basket_name="B_test",
        market_name="MKT",
        open_date=pd.Timestamp("2024-01-10"),
        close_date=None,
        open_zscore=2.5 if direction == "SHORT" else -2.5,
        close_zscore=None,
        direction=direction,
        size_usd=100.0,
        open_price=open_price,
        open_basket=0.5,
    )


def _strategy(tc: float = 0.0) -> PortfolioArbStrategy:
    return PortfolioArbStrategy(basket_name="B_test", tc=tc, size_usd=100.0)


def test_long_pnl_is_yes_token_return():
    strat = _strategy()
    trade = _make_trade("LONG", open_price=0.40)
    trade.close_price = 0.50
    # bought YES at 0.40, now 0.50 → +25% on $100
    assert math.isclose(strat._calc_pnl(trade), 100.0 * 0.10 / 0.40)


def test_short_pnl_uses_no_token_denominator():
    strat = _strategy()
    trade = _make_trade("SHORT", open_price=0.05)
    trade.close_price = 0.03
    # SHORT = buy NO at 0.95, now 0.97 → +2.105% on $100 (NOT +40%)
    expected = 100.0 * (0.05 - 0.03) / (1.0 - 0.05)
    assert math.isclose(strat._calc_pnl(trade), expected)
    assert strat._calc_pnl(trade) < 3.0  # old /p_open formula returned 40.0


def test_short_pnl_symmetric_loss():
    strat = _strategy()
    trade = _make_trade("SHORT", open_price=0.50)
    trade.close_price = 0.60
    # NO bought at 0.50, now worth 0.40 → -20% on $100
    assert math.isclose(strat._calc_pnl(trade), 100.0 * (0.50 - 0.60) / 0.50)


def test_transaction_cost_subtracted_both_sides():
    strat = _strategy(tc=0.015)
    trade = _make_trade("LONG", open_price=0.50)
    trade.close_price = 0.50  # flat price → pure cost
    assert math.isclose(strat._calc_pnl(trade), -0.015 * 100.0 * 2)
