"""
Pair Trading signal generator for Polymarket.

Signal logic:
  - Compute the cointegrated spread:  spread = price_A - hedge_ratio * price_B
  - Normalize spread to a rolling z-score
  - Enter LONG spread (buy A, sell B) when z-score < -ENTRY_ZSCORE
  - Enter SHORT spread (sell A, buy B) when z-score > +ENTRY_ZSCORE
  - Exit when |z-score| < EXIT_ZSCORE
  - Emergency stop-loss when |z-score| > STOP_LOSS_ZSCORE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ENTRY_ZSCORE,
    EXIT_ZSCORE,
    STOP_LOSS_ZSCORE,
    ZSCORE_WINDOW,
    POSITION_SIZE_USD,
)
from pair_selector import PairInfo, compute_rolling_zscore


class Side(Enum):
    LONG_SPREAD = "LONG_SPREAD"     # buy A, sell B
    SHORT_SPREAD = "SHORT_SPREAD"   # sell A, buy B
    FLAT = "FLAT"


@dataclass
class Trade:
    pair_key: str
    open_date: pd.Timestamp
    open_zscore: float
    side: Side
    size_usd: float                 # notional per leg (USD)
    open_price_a: float = 0.5      # actual entry price for leg A
    open_price_b: float = 0.5      # actual entry price for leg B
    close_date: Optional[pd.Timestamp] = None
    close_zscore: Optional[float] = None
    pnl_usd: float = 0.0
    closed_by: str = ""             # "signal", "stop_loss", "eod"

    @property
    def is_open(self) -> bool:
        return self.close_date is None

    @property
    def duration_days(self) -> Optional[int]:
        if self.close_date:
            return (self.close_date - self.open_date).days
        return None


@dataclass
class PairState:
    """Mutable runtime state for one pair."""
    info: PairInfo
    spread: pd.Series = field(default_factory=pd.Series)
    zscore: pd.Series = field(default_factory=pd.Series)
    current_side: Side = Side.FLAT
    open_trade: Optional[Trade] = None
    trades: List[Trade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def build_pair_signals(
    prices: pd.DataFrame,
    pair: PairInfo,
    window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      price_a, price_b, spread, zscore, signal
    where signal ∈ {1: LONG_SPREAD, -1: SHORT_SPREAD, 0: FLAT}.
    """
    joint = prices[[pair.market_a, pair.market_b]].dropna().copy()
    joint.columns = ["price_a", "price_b"]

    joint["spread"] = joint["price_a"] - pair.hedge_ratio * joint["price_b"]
    joint["zscore"] = compute_rolling_zscore(joint["spread"], window)

    joint["signal"] = 0
    joint.loc[joint["zscore"] < -ENTRY_ZSCORE, "signal"] = 1
    joint.loc[joint["zscore"] > ENTRY_ZSCORE, "signal"] = -1

    return joint


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

class PairTradingStrategy:
    """
    Stateful per-pair strategy that produces trade records from price data.

    Usage
    -----
    strategy = PairTradingStrategy(pair_info)
    trades = strategy.run(prices_df)
    """

    def __init__(
        self,
        pair: PairInfo,
        position_size_usd: float = POSITION_SIZE_USD,
        window: int = ZSCORE_WINDOW,
        entry_z: float = ENTRY_ZSCORE,
        exit_z: float = EXIT_ZSCORE,
        stop_z: float = STOP_LOSS_ZSCORE,
        transaction_cost: float = 0.02,
    ) -> None:
        self.pair = pair
        self.size = position_size_usd
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.tc = transaction_cost
        self._state = PairState(info=pair)

    # ------------------------------------------------------------------
    def run(self, prices: pd.DataFrame) -> Tuple[List[Trade], pd.DataFrame]:
        """
        Simulate the strategy day-by-day on `prices`.

        Returns
        -------
        trades : list of Trade objects
        signals_df : DataFrame with per-day signal data
        """
        signals = build_pair_signals(prices, self.pair, self.window)
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None

        dates = list(signals.index)
        # i=0 only provides signal context; execution starts at i=1 (next-bar)
        for i in range(1, len(dates)):
            exec_date = dates[i]
            exec_row  = signals.iloc[i]   # prices for execution
            sig_row   = signals.iloc[i - 1]  # z-score that triggered the decision
            z = sig_row["zscore"]

            if pd.isna(z) or pd.isna(exec_row["price_a"]) or pd.isna(exec_row["price_b"]):
                continue

            # --- Check if we should close an open position (use today's prices) ---
            if current_trade is not None:
                # Evaluate exit condition using yesterday's z (consistent with entry logic)
                closed = self._check_close(current_trade, exec_row, exec_date, z_signal=z)
                if closed:
                    trades.append(current_trade)
                    current_trade = None

            # --- Check if we should open a new position ---
            if current_trade is None:
                new_trade = self._check_open(exec_row, exec_date, z)
                if new_trade is not None:
                    current_trade = new_trade

        # Force-close any open position at end of data
        if current_trade is not None:
            last_row = signals.iloc[-1]
            last_date = signals.index[-1]
            self._close_trade(current_trade, last_row, last_date, reason="eod")
            trades.append(current_trade)

        return trades, signals

    # ------------------------------------------------------------------
    def _check_open(self, row: pd.Series, date: pd.Timestamp, z: float) -> Optional[Trade]:
        if z < -self.entry_z:
            side = Side.LONG_SPREAD
        elif z > self.entry_z:
            side = Side.SHORT_SPREAD
        else:
            return None

        return Trade(
            pair_key=f"{self.pair.market_a}__{self.pair.market_b}",
            open_date=date,
            open_zscore=z,
            side=side,
            size_usd=self.size,
            open_price_a=float(row["price_a"]),
            open_price_b=float(row["price_b"]),
        )

    def _check_close(
        self,
        trade: Trade,
        row: pd.Series,
        date: pd.Timestamp,
        z_signal: Optional[float] = None,
    ) -> bool:
        # Use signal-bar z for the exit decision; fall back to execution-bar z
        z = z_signal if z_signal is not None else row["zscore"]
        if pd.isna(z):
            return False

        # Exit signal
        if abs(z) < self.exit_z:
            self._close_trade(trade, row, date, reason="signal", z_close=z_signal)
            return True

        # Stop-loss
        if abs(z) > self.stop_z:
            self._close_trade(trade, row, date, reason="stop_loss", z_close=z_signal)
            return True

        # Opposite signal flip (regime change)
        if trade.side == Side.LONG_SPREAD and z > self.entry_z:
            self._close_trade(trade, row, date, reason="signal_flip", z_close=z_signal)
            return True
        if trade.side == Side.SHORT_SPREAD and z < -self.entry_z:
            self._close_trade(trade, row, date, reason="signal_flip", z_close=z_signal)
            return True

        return False

    def _close_trade(
        self,
        trade: Trade,
        row: pd.Series,
        date: pd.Timestamp,
        reason: str,
        z_close: Optional[float] = None,
    ) -> None:
        trade.close_date = date
        trade.close_zscore = z_close if z_close is not None else row["zscore"]
        trade.closed_by = reason
        trade.pnl_usd = self._calc_pnl(trade, row)

    def _calc_pnl(self, trade: Trade, close_row: pd.Series) -> float:
        """
        P&L for a spread trade using real binary asset prices.

        For each leg, shares bought = size_usd / open_price (YES-token economics).
        LONG_SPREAD  (buy A, sell B): profit when price_a rises, price_b falls.
        SHORT_SPREAD (sell A, buy B): profit when price_a falls, price_b rises.

        Transaction cost is applied as a fraction of dollar notional per leg (open+close).
        """
        close_price_a = float(close_row["price_a"])
        close_price_b = float(close_row["price_b"])

        # Guard against zero/near-zero open prices (shouldn't happen after clipping)
        open_a = max(trade.open_price_a, 0.01)
        open_b = max(trade.open_price_b, 0.01)

        shares_a = self.size / open_a
        shares_b = self.size / open_b

        if trade.side == Side.LONG_SPREAD:
            gross_pnl = (
                (close_price_a - open_a) * shares_a
                - (close_price_b - open_b) * shares_b * self.pair.hedge_ratio
            )
        else:  # SHORT_SPREAD
            gross_pnl = (
                (open_a - close_price_a) * shares_a
                - (open_b - close_price_b) * shares_b * self.pair.hedge_ratio
            )

        # Round-trip cost: tc fraction of notional for each leg at open and close
        cost = self.tc * (self.size + self.size * abs(self.pair.hedge_ratio)) * 2
        return gross_pnl - cost


# ---------------------------------------------------------------------------
# Multi-pair runner
# ---------------------------------------------------------------------------

def run_all_pairs(
    prices: pd.DataFrame,
    pairs: List[PairInfo],
    **strategy_kwargs,
) -> Tuple[List[Trade], Dict[str, pd.DataFrame]]:
    """Run strategy for every pair and aggregate results."""
    all_trades: List[Trade] = []
    all_signals: Dict[str, pd.DataFrame] = {}

    for pair in pairs:
        strat = PairTradingStrategy(pair, **strategy_kwargs)
        trades, signals = strat.run(prices)
        all_trades.extend(trades)
        key = f"{pair.market_a}__{pair.market_b}"
        all_signals[key] = signals

    return all_trades, all_signals
