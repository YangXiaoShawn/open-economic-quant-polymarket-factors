"""
完备事件套利 (Complete Event Arbitrage)

理论基础
--------
对于互斥且完备的事件（如选举中所有候选人），其 YES 价格之和必须等于 1.0。
当实际市场中价格之和出现偏离时（超过手续费阈值），存在套利机会：

  sum > 1 + threshold  →  做空超价格的候选人（按比例）
  sum < 1 - threshold  →  做多低价格的候选人（按比例）

信号生成
--------
  deviation_t = sum(YES_prices_t) - 1.0
  当 |deviation_t| > ENTRY_THRESHOLD 时开仓
  当 |deviation_t| < EXIT_THRESHOLD 时平仓

收益来源
--------
  假设市场最终回归 sum = 1.0，则可获取偏离量与手续费之差的收益
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import TRANSACTION_COST, POSITION_SIZE_USD

logger = logging.getLogger(__name__)

# Strategy parameters
ENTRY_THRESHOLD = 0.05   # 5% sum deviation to enter
EXIT_THRESHOLD  = 0.015  # close when deviation shrinks below 1.5%
STOP_THRESHOLD  = 0.25   # stop-loss at 25% deviation (structural break)
MIN_CANDIDATES  = 3      # minimum markets in event group


@dataclass
class EventArbTrade:
    event_title: str
    open_date: pd.Timestamp
    close_date: Optional[pd.Timestamp]
    open_deviation: float      # sum - 1.0 at entry
    close_deviation: Optional[float]
    direction: str             # "SELL_BASKET" or "BUY_BASKET"
    n_candidates: int
    size_usd: float
    pnl_usd: float = 0.0
    closed_by: str = ""

    @property
    def is_open(self) -> bool:
        return self.close_date is None

    @property
    def duration_days(self) -> Optional[int]:
        if self.close_date:
            return (self.close_date - self.open_date).days
        return None


class CompleteEventArbStrategy:
    """
    完备事件套利策略。

    参数
    ----
    price_matrix : DataFrame
        列为各候选人 YES 价格，行为日期
    event_title  : str
        事件名称（用于记录）
    """

    def __init__(
        self,
        event_title: str,
        entry_threshold: float = ENTRY_THRESHOLD,
        exit_threshold: float  = EXIT_THRESHOLD,
        stop_threshold: float  = STOP_THRESHOLD,
        position_size_usd: float = POSITION_SIZE_USD,
        transaction_cost: float  = TRANSACTION_COST,
    ) -> None:
        self.event_title = event_title
        self.entry_thr = entry_threshold
        self.exit_thr  = exit_threshold
        self.stop_thr  = stop_threshold
        self.size      = position_size_usd
        self.tc        = transaction_cost

    def run(self, price_matrix: pd.DataFrame) -> Tuple[List[EventArbTrade], pd.Series]:
        """
        在 price_matrix 上逐日模拟策略。

        返回
        ----
        trades      : 所有完成的交易记录
        sum_series  : 每日 YES 价格之和
        """
        if len(price_matrix.columns) < MIN_CANDIDATES:
            return [], pd.Series(dtype=float)

        # 填充缺失值（前向填充）
        df = price_matrix.ffill().dropna(how="all")
        sum_series = df.sum(axis=1)

        # Skip events with structural overround/underround: median sum far from 1.0
        # (e.g., tournaments with many eliminated teams where sum >> 1 is normal)
        median_sum = float(sum_series.median())
        if not (0.7 <= median_sum <= 1.5):
            return [], sum_series

        trades: List[EventArbTrade] = []
        current_trade: Optional[EventArbTrade] = None

        for date, row in df.iterrows():
            deviation = sum_series[date] - 1.0
            if pd.isna(deviation):
                continue

            # --- 检查是否需要平仓 ---
            if current_trade is not None:
                if self._should_close(current_trade, deviation):
                    self._close(current_trade, date, deviation)
                    trades.append(current_trade)
                    current_trade = None

            # --- 检查是否需要开仓 ---
            if current_trade is None:
                if abs(deviation) > self.entry_thr:
                    current_trade = EventArbTrade(
                        event_title    = self.event_title,
                        open_date      = date,
                        close_date     = None,
                        open_deviation = deviation,
                        close_deviation= None,
                        direction      = "SELL_BASKET" if deviation > 0 else "BUY_BASKET",
                        n_candidates   = len(df.columns),
                        size_usd       = self.size,
                    )

        # 强制平仓
        if current_trade is not None:
            last_date = df.index[-1]
            last_dev  = sum_series.iloc[-1] - 1.0
            self._close(current_trade, last_date, last_dev, reason="eod")
            trades.append(current_trade)

        return trades, sum_series

    def _should_close(self, trade: EventArbTrade, deviation: float) -> bool:
        # 偏离回归至出口阈值
        if abs(deviation) < self.exit_thr:
            return True
        # 止损
        if abs(deviation) > self.stop_thr:
            return True
        # 方向反转
        if trade.direction == "SELL_BASKET" and deviation < -self.entry_thr:
            return True
        if trade.direction == "BUY_BASKET"  and deviation >  self.entry_thr:
            return True
        return False

    def _close(
        self,
        trade: EventArbTrade,
        date: pd.Timestamp,
        deviation: float,
        reason: str = "signal",
    ) -> None:
        trade.close_date      = date
        trade.close_deviation = deviation
        trade.closed_by       = reason
        trade.pnl_usd         = self._calc_pnl(trade, deviation)

    def _calc_pnl(self, trade: EventArbTrade, close_dev: float) -> float:
        """
        Basket P&L = deviation_captured * total_position_size.

        We trade the basket as a unit: when sum > 1+entry, we short the basket
        (sell each candidate proportionally). The basket's "price" = sum of YES prices.
        If sum falls from 1.06 to 1.01, the basket returned 0.05 per dollar of notional.

        SELL_BASKET: profitable when deviation shrinks (open_dev > close_dev)
        BUY_BASKET:  profitable when deviation grows back toward 0 (close_dev > open_dev)
        """
        open_dev  = trade.open_deviation
        close_dev = close_dev if close_dev is not None else open_dev

        if trade.direction == "SELL_BASKET":
            captured = open_dev - close_dev   # positive when spread tightened
        else:
            captured = close_dev - open_dev   # positive when spread widened back

        # Gross P&L = total deviation captured × total notional
        gross_pnl = captured * self.size

        # Cost: single round-trip on the basket position
        cost = self.tc * self.size * 2
        return gross_pnl - cost


def run_complete_event_arb(
    event_groups: list,  # List[EventGroup] from event_data
    **strategy_kwargs,
) -> Tuple[List[EventArbTrade], Dict[str, pd.Series]]:
    """在所有事件组上运行完备事件套利策略。"""
    all_trades: List[EventArbTrade] = []
    sum_series_dict: Dict[str, pd.Series] = {}

    for group in event_groups:
        if group.price_matrix is None or len(group.price_matrix.columns) < MIN_CANDIDATES:
            continue

        strat = CompleteEventArbStrategy(
            event_title=group.title,
            **strategy_kwargs,
        )
        trades, sums = strat.run(group.price_matrix)
        all_trades.extend(trades)
        sum_series_dict[group.title] = sums
        logger.debug(
            "Event '%s': %d trades over %d days",
            group.title[:40], len(trades), len(group.price_matrix),
        )

    logger.info("CompleteEventArb: %d total trades across %d events",
                len(all_trades), len(event_groups))
    return all_trades, sum_series_dict
