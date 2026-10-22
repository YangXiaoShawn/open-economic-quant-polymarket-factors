"""
双策略组合回测引擎

策略权重分配（2026-06-11 调整）
------------------------------
  策略 1: 组合套利    (Portfolio Arb)    50% 资本
  策略 2: 做空赢家    (Short-Winners)    50% 资本

  配对交易已移除：扩样研究（4000 市场 × 574 天）显示其全部参数组合
  净收益为负，且本地面板上找不到通过协整检验的配对（资本闲置）。
  做空赢家 = 30 日动量反转空头腿（买 NO），研究净收益 +20.6%/5天
  （t=9.1，真实点差成本，核心价格带 0.15-0.85）。

回测方法
--------
  - Walk-forward: 组合套利篮子在训练窗构建（无前视）
  - 做空赢家: 每 5 天调仓，t 决策 / t+1 收盘入场 / 持有 5 天（次 bar 约定）
  - 独立仓位管理：每策略有独立的资本与仓位规模
  - 统一 P&L：合并每日收益生成综合净值曲线
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL, TRANSACTION_COST, POSITION_SIZE_USD,
    MAX_OPEN_PAIRS, TRAIN_DAYS, STEP_DAYS, ZSCORE_WINDOW,
)
try:
    from config import PAIR_STEP_DAYS
except ImportError:
    PAIR_STEP_DAYS = STEP_DAYS
from portfolio_arb import run_portfolio_arb, run_portfolio_arb_walkforward, PortfolioArbTrade
from backtest import compute_metrics

logger = logging.getLogger(__name__)

# Capital allocation ratios (pair trading removed — see module docstring)
ALLOC_PORT_ARB = 0.50
ALLOC_SW       = 0.50

# Short-winners parameters (from momrev_strategy.py research)
SW_REB_DAYS  = 5      # rebalance / holding period (days)
SW_MOM_LB    = 30     # momentum lookback
SW_MIN_PX    = 0.15   # core tradeable band
SW_MAX_PX    = 0.85
SW_MAX_NAMES = 25     # max names per rebalance (capacity realism)
SW_MIN_UNIV  = 20     # min universe size to trade a rebalance
SW_WARMUP    = 35     # bars before first rebalance


@dataclass
class ShortWinnersTrade:
    """One buy-NO position from a 5-day rebalance cycle."""
    market_name: str
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    open_price: float          # YES price at entry (NO cost = 1 - open_price)
    close_price: float
    mom30: float               # 30d momentum that triggered the short
    size_usd: float
    pnl_usd: float
    direction: str = "SHORT_NO"
    closed_by: str = "rebalance"

    @property
    def is_open(self) -> bool:
        return False

    @property
    def duration_days(self) -> Optional[int]:
        return (self.close_date - self.open_date).days


def run_short_winners(
    prices: pd.DataFrame,
    size_usd: float = POSITION_SIZE_USD,
    tc: float = TRANSACTION_COST,
) -> List[ShortWinnersTrade]:
    """做空赢家：每 SW_REB_DAYS 天做空 30 日动量最高的名字（买 NO）。

    与研究脚本相同的执行约定：t 收盘决策、t+1 收盘入场（须在核心
    价格带内）、持有 SW_REB_DAYS 天按最后成交价退出。
    NO 代币盈亏：size × (p_open − p_close) / (1 − p_open) − 双边成本。
    """
    trades: List[ShortWinnersTrade] = []
    dates = prices.index
    prices_ff = prices.ffill()

    for i in range(SW_WARMUP, len(dates) - SW_REB_DAYS - 1, SW_REB_DAYS):
        p_t = prices.iloc[i]
        p_past = prices.iloc[i - SW_MOM_LB]
        mom = (p_t / p_past.replace(0, np.nan) - 1)
        univ = mom[(p_t > SW_MIN_PX) & (p_t < SW_MAX_PX)].dropna()
        univ = univ[univ > 0]          # short candidates = winners only
        if len(univ) < SW_MIN_UNIV:
            continue
        n_top = min(SW_MAX_NAMES, max(len(univ) // 5, 5))
        top = univ.sort_values(ascending=False).head(n_top)

        p_entry = prices.iloc[i + 1]
        p_exit = prices_ff.iloc[i + 1 + SW_REB_DAYS]
        d_open, d_close = dates[i + 1], dates[i + 1 + SW_REB_DAYS]
        for mkt, m30 in top.items():
            po, px = p_entry.get(mkt), p_exit.get(mkt)
            if pd.isna(po) or pd.isna(px) or not (SW_MIN_PX < po < SW_MAX_PX):
                continue
            pnl = size_usd * ((po - px) / (1.0 - po)) - tc * size_usd * 2
            trades.append(ShortWinnersTrade(
                market_name=str(mkt), open_date=d_open, close_date=d_close,
                open_price=float(po), close_price=float(px),
                mom30=float(m30), size_usd=size_usd, pnl_usd=float(pnl),
            ))
    logger.info("Short-winners: %d trades (%d-day cycles, band %.2f-%.2f)",
                len(trades), SW_REB_DAYS, SW_MIN_PX, SW_MAX_PX)
    return trades


AnyTrade = Union[ShortWinnersTrade, PortfolioArbTrade]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CombinedResult:
    # Per-strategy results
    sw_trades:       List[ShortWinnersTrade]
    port_arb_trades: List[PortfolioArbTrade]

    # Portfolio-level curves
    equity_combined: pd.Series
    equity_sw:       pd.Series
    equity_port_arb: pd.Series

    daily_pnl_combined: pd.Series
    daily_pnl_sw:       pd.Series
    daily_pnl_port_arb: pd.Series

    metrics_combined: Dict
    metrics_sw:       Dict
    metrics_port_arb: Dict

    signals_by_pair: Dict = field(default_factory=dict)   # legacy, always empty

    # ── legacy aliases (main.py / report.py still reference pair_* names) ──
    @property
    def pair_trades(self):
        return self.sw_trades

    @property
    def equity_pair(self):
        return self.equity_sw

    @property
    def metrics_pair(self):
        return self.metrics_sw

    @property
    def daily_pnl_pair(self):
        return self.daily_pnl_sw

    @property
    def all_trades(self) -> List[AnyTrade]:
        return self.sw_trades + self.port_arb_trades

    def __repr__(self) -> str:
        m = self.metrics_combined
        return (
            f"CombinedResult | "
            f"Return={m.get('total_return_pct', 0):+.2f}%  "
            f"Sharpe={m.get('sharpe_ratio', 0):.3f}  "
            f"MaxDD={m.get('max_drawdown_pct', 0):.2f}%  "
            f"Trades={m.get('n_trades', 0)}"
        )


# ---------------------------------------------------------------------------
# Combined backtester
# ---------------------------------------------------------------------------

class CombinedBacktest:
    """
    双策略组合回测器。

    参数
    ----
    prices        : 宽格式价格矩阵（日期 × 市场）
    event_groups  : EventGroup 列表（用于组合套利的事件组篮子）
    initial_cap   : 初始资本（USD）
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        event_groups: Optional[list] = None,
        initial_cap: float = INITIAL_CAPITAL,
        top_n_pairs: int   = MAX_OPEN_PAIRS,
        train_days: int    = TRAIN_DAYS,
        step_days: int     = STEP_DAYS,
    ) -> None:
        self.prices       = prices.sort_index()
        self.event_groups = event_groups or []
        self.initial_cap  = initial_cap
        self.train_days   = train_days
        self.step_days    = step_days          # portfolio arb step size

        # Capital per strategy
        self.cap_sw       = initial_cap * ALLOC_SW
        self.cap_port_arb = initial_cap * ALLOC_PORT_ARB

        # Position sizes: short-winners deploys cap across SW_MAX_NAMES slots;
        # portfolio arb keeps its historical sizing proportional to allocation.
        self.size_sw       = self.cap_sw / SW_MAX_NAMES
        self.size_port_arb = POSITION_SIZE_USD * ALLOC_PORT_ARB / 0.30

    def run(self) -> CombinedResult:
        dates = self.prices.index

        # ---- Strategy 1: Short-Winners (5d momentum-reversal short leg) ----
        sw_trades = run_short_winners(
            self.prices, size_usd=self.size_sw, tc=TRANSACTION_COST,
        )

        # ---- Strategy 2: Portfolio Arb (walk-forward — baskets built on train window) ----
        port_trades, _ = run_portfolio_arb_walkforward(
            self.prices,
            train_days = self.train_days,
            step_days  = self.step_days,
            size_usd   = self.size_port_arb,
            tc         = TRANSACTION_COST,
        )

        # ---- Build equity curves ----
        eq_sw,   dpnl_sw   = self._equity_from_generic_trades(sw_trades, dates, self.cap_sw)
        eq_port, dpnl_port = self._equity_from_generic_trades(port_trades, dates, self.cap_port_arb)

        dpnl_combined = dpnl_sw + dpnl_port
        eq_combined   = self.initial_cap + dpnl_combined.cumsum()

        # ---- Metrics ----
        all_t = self._merge_trades(sw_trades, port_trades)
        m_combined = compute_metrics(eq_combined, dpnl_combined, all_t,      self.initial_cap)
        m_sw       = compute_metrics(eq_sw,       dpnl_sw,       sw_trades,  self.cap_sw)
        m_port     = compute_metrics(eq_port,     dpnl_port,     port_trades, self.cap_port_arb)

        return CombinedResult(
            sw_trades        = sw_trades,
            port_arb_trades  = port_trades,
            equity_combined  = eq_combined,
            equity_sw        = eq_sw,
            equity_port_arb  = eq_port,
            daily_pnl_combined = dpnl_combined,
            daily_pnl_sw       = dpnl_sw,
            daily_pnl_port_arb = dpnl_port,
            metrics_combined = m_combined,
            metrics_sw       = m_sw,
            metrics_port_arb = m_port,
        )

    # ------------------------------------------------------------------
    def _build_windows(self, dates: pd.DatetimeIndex):
        """Walk-forward windows for portfolio arb (uses step_days)."""
        windows = []
        i = 0
        while i + self.train_days < len(dates):
            tr_end   = dates[min(i + self.train_days - 1, len(dates) - 1)]
            te_start = dates[min(i + self.train_days,     len(dates) - 1)]
            te_end   = dates[min(i + self.train_days + self.step_days - 1, len(dates) - 1)]
            if te_start >= te_end:
                break
            windows.append((dates[i], tr_end, te_start, te_end))
            i += self.step_days
        return windows

    def _equity_from_generic_trades(self, trades, dates, cap):
        dpnl = pd.Series(0.0, index=dates)
        for t in trades:
            close = getattr(t, "close_date", None)
            pnl   = getattr(t, "pnl_usd",   0.0)
            if close is not None and close in dpnl.index:
                dpnl[close] += pnl
        return cap + dpnl.cumsum(), dpnl

    @staticmethod
    def _merge_trades(sw_trades, port_trades):
        return list(sw_trades) + list(port_trades)
