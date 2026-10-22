"""
组合套利 (Portfolio Arbitrage)

理论基础
--------
在 Polymarket 上，同一事件的不同市场之间存在**价格一致性约束**。
组合套利通过识别并利用这些约束被打破时的机会：

约束类型 1 — 单调性约束（Monotonicity）
  对于同一底层资产的不同阈值市场（如"比特币是否超过 X"），价格必须满足：
    P(>50k) > P(>100k) > P(>150k) > ...
  违反单调性时可执行无风险套利。

约束类型 2 — 组合偏差（Basket Deviation）
  同类市场的价格中位数形成一个"行业指数"，
  个别市场与行业中位数的偏离超过阈值时，存在均值回归机会。

约束类型 3 — 跨事件一致性（Cross-event）
  类似事件（同类体育赛事的胜负）应具有相似的价格分布。
  当某事件的整体价格水平相对其他同类事件异常时，可以买低卖高。

本实现重点
----------
使用**组合偏差信号**——将市场按主题分组，
当某市场价格相对同组其他市场显著偏高或偏低时进行交易。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import TRANSACTION_COST, POSITION_SIZE_USD, ZSCORE_WINDOW

logger = logging.getLogger(__name__)

# Strategy parameters
BASKET_ENTRY_ZSCORE  = 2.0   # enter when individual price z-score vs basket exceeds this
BASKET_EXIT_ZSCORE   = 0.5   # exit when z-score returns below this
BASKET_STOP_ZSCORE   = 4.0   # stop-loss
MAX_HOLD_DAYS        = 30    # time-based stop: close any position held longer than this
MIN_BASKET_SIZE      = 4     # minimum markets per basket
ROLLING_WINDOW       = 20    # z-score lookback window
MIN_TRADE_PRICE      = 0.10  # core tradeable band per expanded-sample research:
MAX_TRADE_PRICE      = 0.90  # boundary entries (<0.10 / >0.90) are lottery-dominated —
                             # token leverage 10x+ makes P&L estimates unreliable


@dataclass
class PortfolioArbTrade:
    basket_name: str
    market_name: str           # the individual market being traded
    open_date: pd.Timestamp
    close_date: Optional[pd.Timestamp]
    open_zscore: float
    close_zscore: Optional[float]
    direction: str             # "LONG" (buy underpriced) or "SHORT" (sell overpriced)
    size_usd: float
    # Actual market price and basket median at entry/exit — used for realistic P&L
    open_price: float = 0.0
    open_basket: float = 0.0
    close_price: Optional[float] = None
    close_basket: Optional[float] = None
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


class PortfolioArbStrategy:
    """
    组合套利策略。

    每个市场与其"篮子均值"（同类市场的等权中位数）进行比较，
    当相对偏离的滚动 z-score 超过阈值时开仓。
    """

    def __init__(
        self,
        basket_name: str,
        entry_z: float    = BASKET_ENTRY_ZSCORE,
        exit_z: float     = BASKET_EXIT_ZSCORE,
        stop_z: float     = BASKET_STOP_ZSCORE,
        max_hold_days: int = MAX_HOLD_DAYS,
        window: int       = ROLLING_WINDOW,
        size_usd: float   = POSITION_SIZE_USD,
        tc: float         = TRANSACTION_COST,
    ) -> None:
        self.basket_name  = basket_name
        self.entry_z      = entry_z
        self.exit_z       = exit_z
        self.stop_z       = stop_z
        self.max_hold_days = max_hold_days
        self.window       = window
        self.size         = size_usd
        self.tc           = tc

    def run(
        self,
        price_matrix: pd.DataFrame,
        force_close_at_end: bool = True,
        initial_positions: Optional[Dict[str, "PortfolioArbTrade"]] = None,
    ) -> Tuple[List[PortfolioArbTrade], Dict[str, PortfolioArbTrade], pd.DataFrame]:
        """
        Run basket arbitrage on price_matrix.

        Parameters
        ----------
        force_close_at_end : if True (default), EOD-close any remaining open
            positions at the last bar of price_matrix.  Set False when the
            caller will carry those positions into the next window.
        initial_positions  : positions already open at the start of this window
            (carry-forward from a previous window).  They are processed for
            exits but NOT counted as new opens for the te_start filter.

        Returns
        -------
        closed_trades    : list of trades closed during this window
        remaining_open   : dict {mkt -> trade} still open at end (only populated
            when force_close_at_end=False)
        zscore_df        : full z-score DataFrame for this window
        """
        if len(price_matrix.columns) < MIN_BASKET_SIZE:
            return [], {}, pd.DataFrame()

        df = price_matrix.ffill().dropna(how="all")
        basket_median = df.median(axis=1)

        last_bmed = float(basket_median.iloc[-1])
        valid_cols = [
            c for c in df.columns
            if last_bmed * 0.1 <= df[c].iloc[-1] <= last_bmed * 10
        ]
        if len(valid_cols) < MIN_BASKET_SIZE:
            return [], {}, pd.DataFrame()
        df = df[valid_cols]
        basket_median = df.median(axis=1)

        pct_devs = df.subtract(basket_median, axis=0).divide(
            basket_median.clip(lower=0.01), axis=0
        )
        zscores = pct_devs.apply(
            lambda col: self._rolling_zscore(col, self.window)
        )

        closed_trades: List[PortfolioArbTrade] = []
        # Seed with carry-forward positions (filter to markets present in this basket)
        open_positions: Dict[str, PortfolioArbTrade] = {}
        if initial_positions:
            for mkt, trade in initial_positions.items():
                if mkt in df.columns:
                    open_positions[mkt] = trade

        dates_arr   = df.index
        mkt_list    = list(df.columns)
        mkt_idx     = {m: i for i, m in enumerate(mkt_list)}
        prices_arr  = df.values
        bmed_arr    = basket_median.values
        zscores_arr = zscores.values

        for t_idx, date in enumerate(dates_arr):
            bmed = float(bmed_arr[t_idx])

            for mkt, trade in list(open_positions.items()):
                mi = mkt_idx.get(mkt)
                if mi is None:
                    continue
                # Guard: never close a carry-forward position before its own open date
                if date < trade.open_date:
                    continue
                z = float(zscores_arr[t_idx, mi])
                if np.isnan(z):
                    continue
                if self._should_close(trade, z, date):
                    cp = float(prices_arr[t_idx, mi])
                    self._close_trade(trade, date, z, cp, bmed)
                    closed_trades.append(trade)
                    del open_positions[mkt]

            if len(open_positions) < 2:
                for mi, mkt in enumerate(mkt_list):
                    if mkt in open_positions:
                        continue
                    z = float(zscores_arr[t_idx, mi])
                    if np.isnan(z):
                        continue
                    op = float(prices_arr[t_idx, mi])
                    if op < MIN_TRADE_PRICE or op > MAX_TRADE_PRICE:
                        continue
                    if z > self.entry_z:
                        open_positions[mkt] = PortfolioArbTrade(
                            basket_name  = self.basket_name,
                            market_name  = mkt,
                            open_date    = date,
                            close_date   = None,
                            open_zscore  = z,
                            close_zscore = None,
                            direction    = "SHORT",
                            size_usd     = self.size,
                            open_price   = op,
                            open_basket  = bmed,
                        )
                    elif z < -self.entry_z:
                        open_positions[mkt] = PortfolioArbTrade(
                            basket_name  = self.basket_name,
                            market_name  = mkt,
                            open_date    = date,
                            close_date   = None,
                            open_zscore  = z,
                            close_zscore = None,
                            direction    = "LONG",
                            size_usd     = self.size,
                            open_price   = op,
                            open_basket  = bmed,
                        )

        if force_close_at_end:
            last_date = df.index[-1]
            cbm = float(basket_median.iloc[-1])
            for mkt, trade in open_positions.items():
                z  = float(zscores.loc[last_date, mkt]) if mkt in zscores.columns else 0.0
                cp = float(df.loc[last_date, mkt])      if mkt in df.columns      else trade.open_price
                self._close_trade(trade, last_date, z, cp, cbm, reason="eod")
                closed_trades.append(trade)
            return closed_trades, {}, zscores

        return closed_trades, open_positions, zscores

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        mu  = series.rolling(window, min_periods=window // 2).mean()
        std = series.rolling(window, min_periods=window // 2).std()
        return (series - mu) / std.replace(0, np.nan)

    def _should_close(self, trade: PortfolioArbTrade, z: float,
                      current_date: Optional[pd.Timestamp] = None) -> bool:
        if abs(z) < self.exit_z:
            return True
        if abs(z) > self.stop_z:
            return True
        if trade.direction == "SHORT" and z < -self.entry_z:
            return True
        if trade.direction == "LONG"  and z >  self.entry_z:
            return True
        # Time-based stop: exit positions held longer than max_hold_days
        if (current_date is not None and self.max_hold_days > 0
                and (current_date - trade.open_date).days >= self.max_hold_days):
            return True
        return False

    def _close_trade(
        self,
        trade: PortfolioArbTrade,
        date: pd.Timestamp,
        z: float,
        close_price: float,
        close_basket: float,
        reason: str = "signal",
    ) -> None:
        trade.close_date   = date
        trade.close_zscore = z
        trade.close_price  = close_price
        trade.close_basket = close_basket
        trade.closed_by    = reason
        trade.pnl_usd      = self._calc_pnl(trade)

    def _calc_pnl(self, trade: PortfolioArbTrade) -> float:
        """
        Percentage-return P&L model (binary-token economics).

          LONG  (buy YES at p_open, expect price to rise toward basket):
            P&L = size_usd × (p_close - p_open) / p_open

          SHORT (Polymarket has no native shorting — selling YES exposure
          means buying NO tokens at 1 - p_open):
            P&L = size_usd × (p_open - p_close) / (1 - p_open)

        The SHORT denominator must be the NO-token entry price.  Dividing by
        p_open instead overstates SHORT returns on low-priced markets by up
        to ~19x (p=0.05: (0.05-0.03)/0.05 = 40% vs the true NO-token return
        (0.97-0.95)/0.95 = 2.1%).
        """
        p_open  = min(max(trade.open_price, 0.001), 0.999)
        p_close = trade.close_price if trade.close_price is not None else p_open

        if trade.direction == "LONG":
            pct_return = (p_close - p_open) / p_open
        else:  # SHORT — NO-token economics
            pct_return = (p_open - p_close) / (1.0 - p_open)

        gross_pnl = trade.size_usd * pct_return
        cost      = self.tc * trade.size_usd * 2
        return gross_pnl - cost


# ---------------------------------------------------------------------------
# Basket builder: group the broad price matrix into themed clusters
# ---------------------------------------------------------------------------

def build_baskets_from_prices(
    prices: pd.DataFrame,
    min_corr: float = 0.75,
    min_size: int   = MIN_BASKET_SIZE,
) -> Dict[str, pd.DataFrame]:
    """
    从宽格式价格矩阵中，通过正相关性聚类自动提取市场篮子。

    关键约束：
      - 只使用正相关（不用 abs）—— 负相关市场不属于同一篮子
      - 篮子内所有市场价格须在中位数 ±4x 范围内（价格同质性）
      - 篮子最少 4 个市场
    """
    # Sort by data coverage (non-null count) descending: more history ≈ more liquid.
    # High-volume markets become cluster seeds first, producing economically stable baskets.
    cols = sorted(prices.columns, key=lambda c: prices[c].notna().sum(), reverse=True)
    # Use positive-only correlation (NOT abs) — negatively correlated markets move inversely
    corr = prices.corr().fillna(0)

    assigned: set = set()
    baskets: Dict[str, pd.DataFrame] = {}
    basket_idx = 0

    for col in cols:
        if col in assigned:
            continue
        col_med = prices[col].median()
        if col_med < MIN_TRADE_PRICE:   # skip very-low-price seed markets
            continue

        # Only POSITIVE correlation AND similar price level
        neighbors = []
        for c in cols:
            if c == col or c in assigned:
                continue
            if corr.loc[col, c] < min_corr:     # positive corr only
                continue
            c_med = prices[c].median()
            if c_med < MIN_TRADE_PRICE:          # skip low-price neighbors
                continue
            # Price homogeneity: both markets within 3x of each other
            ratio = max(col_med, c_med) / max(min(col_med, c_med), 0.001)
            if ratio > 3.0:
                continue
            neighbors.append(c)

        if len(neighbors) + 1 < min_size:
            continue

        cluster = [col] + neighbors[:19]
        for c in cluster:
            assigned.add(c)
        # Use a content hash so the same set of markets gets the same basket name
        # across walk-forward windows.  Sequential numbering caused carry-forward
        # to silently match positions to wrong baskets when windows rebuilt baskets.
        cluster_key = ",".join(sorted(cluster))
        basket_name = "B_" + hashlib.md5(cluster_key.encode()).hexdigest()[:8]
        baskets[basket_name] = prices[cluster]
        basket_idx += 1

    logger.info(
        "Built %d baskets from %d markets (assigned %d, min_corr=%.2f, min_price=%.2f)",
        len(baskets), len(cols), len(assigned), min_corr, MIN_TRADE_PRICE,
    )
    return baskets


def run_portfolio_arb_walkforward(
    prices: pd.DataFrame,
    train_days: int = 45,
    step_days: int  = 30,
    min_corr: float = 0.75,
    **strategy_kwargs,
) -> Tuple[List[PortfolioArbTrade], Dict[str, pd.DataFrame]]:
    """
    Walk-forward portfolio arb with position carry-forward.

    Baskets are rebuilt every step_days from the training window (no look-ahead),
    but open positions are NOT force-closed at window boundaries.  A position
    stays open until its exit signal fires or the end of all data — exactly as
    it would in live trading.  Only the basket FORMATION is walk-forward; the
    position MANAGEMENT is continuous.
    """
    all_trades: List[PortfolioArbTrade] = []
    dates = prices.index

    # carry_fwd[basket_name] = {mkt -> trade}  — positions open at end of last window
    # carry_cols[basket_name] = list of market columns for that basket
    carry_fwd:  Dict[str, Dict[str, PortfolioArbTrade]] = {}
    carry_cols: Dict[str, List[str]] = {}

    i = 0
    while i + train_days < len(dates):
        tr_start = dates[i]
        tr_end   = dates[min(i + train_days - 1, len(dates) - 1)]
        te_start = dates[min(i + train_days,     len(dates) - 1)]
        te_end   = dates[min(i + train_days + step_days - 1, len(dates) - 1)]
        if te_start >= te_end:
            break

        train_prices = prices.loc[tr_start:tr_end]
        exec_prices  = prices.loc[tr_start:te_end]   # warmup + test

        # ── Rebuild baskets from training data ──────────────────────────
        new_baskets = build_baskets_from_prices(train_prices, min_corr=min_corr)
        new_basket_cols: Dict[str, List[str]] = {}

        # ── Run each new basket, carrying forward any matching positions ──
        new_carry: Dict[str, Dict[str, PortfolioArbTrade]] = {}

        for basket_name, basket_train in new_baskets.items():
            valid_cols = [c for c in basket_train.columns if c in exec_prices.columns]
            if len(valid_cols) < MIN_BASKET_SIZE:
                continue
            new_basket_cols[basket_name] = valid_cols

            # Positions from a previous window for the same basket
            init_pos = carry_fwd.get(basket_name, {})

            strat = PortfolioArbStrategy(basket_name=basket_name, **strategy_kwargs)
            closed, still_open, _ = strat.run(
                exec_prices[valid_cols],
                force_close_at_end=False,
                initial_positions=init_pos,
            )
            # Closed trades: either genuinely new (opened in te_start+) or carry-forward
            for t in closed:
                if t.open_date >= te_start or t.market_name in (init_pos or {}):
                    all_trades.append(t)
            if still_open:
                new_carry[basket_name] = still_open

        # ── Carry-forward positions from baskets that are no longer active ──
        # These were in a basket that wasn't rebuilt this window.  Run them on
        # their original columns so they can still exit on signal.
        for old_bname, old_positions in carry_fwd.items():
            if old_bname in new_basket_cols:
                continue                      # already handled above
            if not old_positions:
                continue
            old_vcols = [c for c in carry_cols.get(old_bname, [])
                         if c in exec_prices.columns]
            if len(old_vcols) < MIN_BASKET_SIZE:
                # Basket lost too many markets — close remaining positions now
                last_date = exec_prices.index[-1]
                for mkt, trade in old_positions.items():
                    cp = float(exec_prices[mkt].iloc[-1]) if mkt in exec_prices.columns else trade.open_price
                    bm = float(exec_prices[old_vcols].median(axis=1).iloc[-1]) if old_vcols else 0.0
                    strat_tmp = PortfolioArbStrategy(old_bname, **strategy_kwargs)
                    strat_tmp._close_trade(trade, last_date, 0.0, cp, bm, reason="eod")
                    all_trades.append(trade)
                continue

            strat_old = PortfolioArbStrategy(old_bname, **strategy_kwargs)
            closed_old, still_open_old, _ = strat_old.run(
                exec_prices[old_vcols],
                force_close_at_end=False,
                initial_positions=old_positions,
            )
            for t in closed_old:
                if t.open_date >= te_start or t.market_name in (old_positions or {}):
                    all_trades.append(t)
            if still_open_old:
                new_carry[old_bname] = still_open_old
                new_basket_cols.setdefault(old_bname, old_vcols)

        # Update carry state for next window
        carry_fwd  = new_carry
        carry_cols = {**carry_cols, **new_basket_cols}
        i += step_days

    # ── End of all data: force-close every remaining position ────────────
    last_date = dates[-1]
    for bname, positions in carry_fwd.items():
        vcols = [c for c in carry_cols.get(bname, []) if c in prices.columns]
        for mkt, trade in positions.items():
            if mkt in prices.columns:
                # Use last valid price (ffill so NaN on final day doesn't propagate)
                col_ser = prices[mkt].ffill()
                cp = float(col_ser.iloc[-1]) if not pd.isna(col_ser.iloc[-1]) else trade.open_price
            else:
                cp = trade.open_price
            if vcols:
                bm_ser = prices[vcols].ffill().iloc[-1]
                bm = float(bm_ser.median())
            else:
                bm = 0.0
            strat_fin = PortfolioArbStrategy(bname, **strategy_kwargs)
            strat_fin._close_trade(trade, last_date, 0.0, cp, bm, reason="eod")
            all_trades.append(trade)

    logger.info(
        "PortfolioArb walk-forward: %d trades (train=%dd step=%dd) [carry-forward enabled]",
        len(all_trades), train_days, step_days,
    )
    return all_trades, {}


def run_portfolio_arb(
    prices: pd.DataFrame,
    event_groups: Optional[list] = None,  # EventGroup list from event_data
    min_corr: float = 0.75,
    **strategy_kwargs,
) -> Tuple[List[PortfolioArbTrade], Dict[str, pd.DataFrame]]:
    """
    运行组合套利策略：
      1. 若提供事件组，则每个事件组作为一个篮子
      2. 从宽格式价格矩阵自动构建相关性篮子
    """
    from typing import Optional as Opt

    all_trades: List[PortfolioArbTrade] = []
    all_zscores: Dict[str, pd.DataFrame] = {}

    # From event groups
    if event_groups:
        for group in event_groups:
            if group.price_matrix is None or len(group.price_matrix.columns) < MIN_BASKET_SIZE:
                continue
            strat = PortfolioArbStrategy(basket_name=group.title[:40], **strategy_kwargs)
            trades, _, zscores = strat.run(group.price_matrix)
            all_trades.extend(trades)
            all_zscores[group.title[:40]] = zscores

    # From price matrix auto-clusters
    baskets = build_baskets_from_prices(prices, min_corr=min_corr)
    for basket_name, basket_prices in baskets.items():
        strat = PortfolioArbStrategy(basket_name=basket_name, **strategy_kwargs)
        trades, _, zscores = strat.run(basket_prices)
        all_trades.extend(trades)
        all_zscores[basket_name] = zscores

    logger.info("PortfolioArb: %d total trades across %d baskets",
                len(all_trades), len(baskets) + (len(event_groups) if event_groups else 0))
    return all_trades, all_zscores
