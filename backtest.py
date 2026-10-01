"""
Walk-forward backtesting engine for Polymarket Pair Trading.

Methodology
-----------
• Training window  – pairs are selected and hedge ratios estimated
• Test window      – strategy is applied to out-of-sample prices
• Walk-forward     – windows slide forward by STEP_DAYS until end of data

This avoids look-ahead bias: no future data enters signal generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    POSITION_SIZE_USD,
    MAX_OPEN_PAIRS,
    ZSCORE_WINDOW,
    TRAIN_DAYS,
    STEP_DAYS,
)
from pair_selector import select_pairs, PairInfo
from strategy import PairTradingStrategy, Trade, run_all_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    daily_pnl: pd.Series
    metrics: Dict
    signals_by_pair: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"BacktestResult | "
            f"Total Return: {m.get('total_return_pct', 0):.2f}%  "
            f"Sharpe: {m.get('sharpe_ratio', 0):.3f}  "
            f"MaxDD: {m.get('max_drawdown_pct', 0):.2f}%  "
            f"Win Rate: {m.get('win_rate_pct', 0):.1f}%  "
            f"Trades: {m.get('n_trades', 0)}"
        )


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

class WalkForwardBacktest:
    """
    Walk-forward backtester.

    Parameters
    ----------
    prices       Full price DataFrame (dates × market_ids)
    initial_cap  Starting capital in USD
    top_n_pairs  Max pairs to trade per window
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_cap: float = INITIAL_CAPITAL,
        top_n_pairs: int = MAX_OPEN_PAIRS,
        train_days: int = TRAIN_DAYS,
        step_days: int = STEP_DAYS,
        pair_strategy_kwargs: Optional[Dict] = None,
    ) -> None:
        self.prices = prices.sort_index()
        self.initial_cap = initial_cap
        self.top_n = top_n_pairs
        self.train_days = train_days
        self.step_days = step_days
        self.pair_strategy_kwargs = pair_strategy_kwargs or {}

    def run(self) -> BacktestResult:
        """Execute full walk-forward backtest."""
        dates = self.prices.index
        all_trades: List[Trade] = []
        all_signals: Dict[str, pd.DataFrame] = {}

        # Build walk-forward windows
        windows = self._build_windows(dates)
        logger.info("Running %d walk-forward windows …", len(windows))

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
            logger.info(
                "Window %d/%d: train [%s → %s], test [%s → %s]",
                i, len(windows),
                train_start.date(), train_end.date(),
                test_start.date(), test_end.date(),
            )

            train_prices = self.prices.loc[train_start:train_end]
            test_prices = self.prices.loc[test_start:test_end]

            # Require at least 2/3 of the training window (min 30 days) for reliable
            # cointegration tests; the old threshold of train_days//2 (≈22 days) was
            # too low for statistical reliability.
            if len(train_prices) < max(self.train_days * 2 // 3, 30):
                logger.debug("Skipping window %d – insufficient training data", i)
                continue

            # Pair selection on training data
            pairs = select_pairs(train_prices, top_n=self.top_n)
            if not pairs:
                logger.debug("No valid pairs found in window %d", i)
                continue

            logger.info("  → %d pairs selected", len(pairs))

            # Run strategy on test data, using parameters from training window
            trades, signals = run_all_pairs(
                test_prices,
                pairs,
                **{
                    "position_size_usd": POSITION_SIZE_USD,
                    "transaction_cost": TRANSACTION_COST,
                    "window": min(ZSCORE_WINDOW, len(test_prices) // 2),
                    **self.pair_strategy_kwargs,
                },
            )
            all_trades.extend(trades)
            all_signals.update(signals)

        equity, daily_pnl = self._build_equity_curve(all_trades, dates)
        metrics = compute_metrics(equity, daily_pnl, all_trades, self.initial_cap)

        return BacktestResult(
            trades=all_trades,
            equity_curve=equity,
            daily_pnl=daily_pnl,
            metrics=metrics,
            signals_by_pair=all_signals,
        )

    # ------------------------------------------------------------------
    def _build_windows(
        self, dates: pd.DatetimeIndex
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        windows = []
        start_idx = 0
        while start_idx + self.train_days < len(dates):
            train_start = dates[start_idx]
            train_end = dates[min(start_idx + self.train_days - 1, len(dates) - 1)]
            test_start = dates[min(start_idx + self.train_days, len(dates) - 1)]
            test_end = dates[min(start_idx + self.train_days + self.step_days - 1, len(dates) - 1)]
            if test_start >= test_end:
                break
            windows.append((train_start, train_end, test_start, test_end))
            start_idx += self.step_days
        return windows

    def _build_equity_curve(
        self,
        trades: List[Trade],
        dates: pd.DatetimeIndex,
    ) -> Tuple[pd.Series, pd.Series]:
        """Reconstruct daily equity from trade P&L."""
        daily_pnl = pd.Series(0.0, index=dates, dtype=float)

        for trade in trades:
            if trade.close_date is not None and trade.close_date in daily_pnl.index:
                daily_pnl[trade.close_date] += trade.pnl_usd

        equity = self.initial_cap + daily_pnl.cumsum()
        return equity, daily_pnl


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    equity: pd.Series,
    daily_pnl: pd.Series,
    trades: List[Trade],
    initial_cap: float,
) -> Dict:
    if len(equity) == 0 or len(trades) == 0:
        return {}

    final_equity = equity.iloc[-1]
    total_return = (final_equity - initial_cap) / initial_cap

    # Annualised metrics — use actual calendar span (not row count) to avoid
    # gap-day bias; use geometric compounding (industry standard).
    date_span = max((equity.index[-1] - equity.index[0]).days, 1)
    ann_factor = 365 / date_span

    # Divide daily PnL by the PREVIOUS day's equity (not static initial_cap).
    # Using initial_cap as denominator understates risk when the portfolio is
    # significantly up or down — lagged equity gives a proper period return.
    equity_lagged = equity.shift(1).fillna(initial_cap)
    daily_ret = daily_pnl / equity_lagged.clip(lower=1.0)

    # Geometric annualization: (1+R)^(1/years) - 1
    ann_return = (1 + total_return) ** ann_factor - 1
    # Observation frequency: prediction markets trade 7 days/week
    obs_per_year = len(daily_pnl) / date_span * 365
    ann_vol = daily_ret.std() * np.sqrt(obs_per_year) if daily_ret.std() > 0 else 0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino (downside deviation)
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(obs_per_year)
    sortino = ann_return / downside if downside > 0 else 0

    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

    # Trade-level stats
    closed = [t for t in trades if t.close_date is not None]
    n_trades = len(closed)
    winners = [t for t in closed if t.pnl_usd > 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    avg_win = np.mean([t.pnl_usd for t in winners]) if winners else 0
    losers = [t for t in closed if t.pnl_usd <= 0]
    avg_loss = np.mean([t.pnl_usd for t in losers]) if losers else 0
    profit_factor = (
        sum(t.pnl_usd for t in winners) / abs(sum(t.pnl_usd for t in losers))
        if losers and sum(t.pnl_usd for t in losers) != 0
        else float("inf")
    )
    avg_duration = (
        np.mean([t.duration_days for t in closed if t.duration_days is not None])
        if closed else 0
    )
    exit_reasons = pd.Series([t.closed_by for t in closed]).value_counts().to_dict()

    return {
        "initial_capital": initial_cap,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct": round(ann_return * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_trade_duration_days": round(avg_duration, 1),
        "exit_reasons": exit_reasons,
    }
