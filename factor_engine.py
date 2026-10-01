"""
预测市场因子研究引擎 — Prediction Market Factor Engine

Phase 1 因子
------------
  PM-MOM  : 概率动量 — 30 日价格变化率，测试动量效应
  PM-LIQ  : 流动性   — Amihud 非流动性比率，测试流动性溢价
  PM-TTR  : 距结算时间 — log(1 + days_to_end)，测试时间效应

分析方法
--------
  1. 截面因子得分（今日快照）
  2. Walk-Forward IC 时间序列 + t-stat（预测能力验证）
  3. 十分位组合收益（因子单调性验证）
  4. 多空组合净值曲线

Phase 2 预留接口
--------------
  run_fama_macbeth() — Fama-MacBeth 两步回归（需 PM-LEAD 小时线因子就绪后启用）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import DEFAULT_LOOKBACK_DAYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOM_LOOKBACK    = 30    # 动量回溯窗口（天）
IC_FORWARD_DAYS = 5     # IC 计算的前向收益天数
IC_MIN_MARKETS  = 10    # 每期截面回归需要的最少市场数
N_DECILES       = 10    # 分位组合数

INITIAL_FACTOR_EQUITY = 10_000.0  # 多空组合初始净值（USD，仅用于归一化）


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FactorResult:
    name: str
    description_zh: str
    scores: pd.Series            # 今日截面得分，index=market_label，越高越好
    ic_series: pd.Series         # Walk-forward IC 时间序列，index=date
    mean_ic: float               # 平均 IC
    ic_tstat: float              # t-stat = mean_ic / std_ic * sqrt(n)
    ic_pvalue: float             # 双尾 p 值
    decile_returns: pd.Series    # index=1..N_DECILES，value=该分位平均收益率(%)
    portfolio_equity: pd.Series  # 多空组合净值曲线（等权多 top 20% - 空 bottom 20%）
    n_markets_used: int          # 参与计算的市场数
    # MOM 多窗口对比结果（仅 PM-MOM 填充）{window_days: (mean_ic, tstat, pvalue)}
    mom_windows: Optional[Dict[int, Tuple[float, float, float]]] = field(default=None)


@dataclass
class FamaMacBethResult:
    """Phase 2 Fama-MacBeth 结果容器（预留）。"""
    factor_names: List[str]
    factor_premia: pd.Series      # β̄_k — 因子风险溢价
    t_statistics: pd.Series       # Newey-West 调整后的 t-stat
    p_values: pd.Series
    mean_r_squared: float         # 截面回归平均 R²
    alpha: float
    alpha_tstat: float
    beta_time_series: pd.DataFrame  # T×K


# ---------------------------------------------------------------------------
# Historical IC helpers (price-derived proxies for look-ahead-free computation)
# ---------------------------------------------------------------------------

def _build_volume_proxy(prices: pd.DataFrame) -> Dict[str, float]:
    """
    Build a per-market volume proxy from price activity in a historical price slice.

    The Gamma API's volume24hr is a snapshot of today's data. Using it for
    historical IC periods introduces look-ahead bias. This function estimates
    relative liquidity from price history alone:

        proxy_i = active_days_i × mean_price_i

    where active_days counts days with |Δp| > 0.1% (a proxy for trade count).
    Higher proxy value ↔ more liquid (used as denominator in Amihud ILLIQ).
    """
    daily_ret = prices.pct_change().abs()
    active_days = (daily_ret > 1e-3).sum().clip(lower=1)
    mean_price = prices.mean().clip(lower=0.01)
    return (active_days * mean_price).to_dict()


def _spread_proxy_from_prices(prices: pd.DataFrame) -> pd.Series:
    """
    Estimate relative bid-ask spread from price history alone.

    The Gamma API's bestBid/bestAsk/spread are real-time snapshots — using
    them for historical IC periods is time-contaminated. Instead we approximate:

        spread_proxy_i = 2 × p10(|Δp_i|) / mid_price_i

    where p10(|Δp|) is the 10th percentile of non-zero daily absolute price
    changes (≈ minimum observable tick size). Higher value = wider spread.
    """
    if prices.empty or len(prices) < 5:
        return pd.Series(dtype=float)
    daily_chg = prices.diff().abs()
    result: Dict[str, float] = {}
    for col in prices.columns:
        chg = daily_chg[col].dropna()
        nonzero = chg[chg > 1e-4]
        if len(nonzero) < 3:
            continue
        min_tick = float(nonzero.quantile(0.10))
        mid = float(prices[col].dropna().iloc[-1])
        mid = max(mid, 0.01)
        result[col] = 2.0 * min_tick / mid
    return pd.Series(result, dtype=float).rename("PM-SPREAD")


# ---------------------------------------------------------------------------
# Factor computation functions
# ---------------------------------------------------------------------------

def compute_momentum(
    prices: pd.DataFrame,
    lookback: int = MOM_LOOKBACK,
) -> pd.Series:
    """
    PM-MOM: 30 日价格变化率。
    越高 = 最近价格上涨越快（强动量）。
    用 prices.iloc[-1] / prices.iloc[-1-lookback] - 1，忽略中间缺失。
    """
    if len(prices) < lookback + 1:
        return pd.Series(dtype=float)
    recent = prices.iloc[-1]
    past   = prices.iloc[-(lookback + 1)]
    mom    = (recent / past.replace(0, np.nan) - 1).dropna()
    return mom.rename("PM-MOM")


def compute_liquidity(
    prices: pd.DataFrame,
    volume_map: Dict[str, float],
) -> pd.Series:
    """
    PM-LIQ: Amihud 非流动性比率近似值。
    ILLIQ_i = mean(|daily_return_i|) / volume24hr_i

    值越高 = 越非流动（小成交量但价格波动大）。
    在做截面排序时，非流动性高的市场排名靠后（信号：低流动性可能被错误定价）。
    """
    if prices.empty:
        return pd.Series(dtype=float)
    daily_ret     = prices.pct_change().abs().mean()   # 各市场平均绝对日收益
    volume_series = pd.Series(volume_map, dtype=float)
    aligned       = daily_ret.align(volume_series, join="inner")
    illiq         = aligned[0] / aligned[1].clip(lower=1.0)
    return illiq.dropna().rename("PM-LIQ")


def compute_ttr(
    markets: List[Dict],
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    PM-TTR: log(1 + days_to_resolution) 相对于 reference_date（默认 UTC 今日）。

    在 walk-forward IC 中应传入历史日期，使 TTR 随时间正确变化：
    TTR_t = log(1 + max(endDate - t, 0))
    """
    ref = reference_date if reference_date is not None else pd.Timestamp.now(tz="UTC")
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    result: Dict[str, float] = {}
    for m in markets:
        end_raw = None
        for field_name in ("endDate", "resolutionDate"):
            raw = m.get(field_name)
            if raw and len(raw) >= 10:
                end_raw = raw[:10]
                break
        if not end_raw:
            continue
        try:
            end_ts = pd.Timestamp(end_raw[:10]).tz_localize("UTC")
            days   = max((end_ts - ref).days, 0)
            label  = m.get("question", "")[:60].strip()
            if label:
                result[label] = float(np.log1p(days))
        except Exception:
            pass
    return pd.Series(result, dtype=float).rename("PM-TTR")


def compute_extremity(prices: pd.DataFrame) -> pd.Series:
    """
    PM-EXTR: 价格极端性 = 2 × |p - 0.5| ∈ [0, 1]。

    0.5 → 0（最大不确定性），0.95 → 0.9（强度极端）。
    假设：极端价格市场受 Longshot bias 系统性影响，倾向于向校准曲线修正。
    PM-CAL 数据（RMSE=0.153）已证实这种偏差存在。
    预期 IC 方向：负（越极端 → 修正越大）。
    """
    if prices.empty:
        return pd.Series(dtype=float)
    p = prices.iloc[-1].clip(0.0, 1.0)
    return (2 * (p - 0.5).abs()).rename("PM-EXTR")


def compute_drift(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    PM-DRIFT: 过去 window 天的价格方向一致性 = (up_days - down_days) / total_days。

    取值 [-1, 1]：+1 = 全部上涨，-1 = 全部下跌，0 = 涨跌各半。
    与 PM-MOM 区别：MOM 测量涨幅大小，DRIFT 测量方向连续性。
    假设：持续单向运动反映参与者过度定价趋势 → 短期反转。
    预期 IC 方向：负（持续上涨 → 随后修正）。
    """
    if len(prices) < window + 2:
        return pd.Series(dtype=float)
    rets = prices.iloc[-(window + 1):].pct_change()
    # NaN-safe column-wise count: sum() skips NaN, no row-level dropna()
    up    = (rets > 0).sum()
    down  = (rets < 0).sum()
    valid = up + down
    # Require at least half the window of valid observations per column
    mask  = valid >= max(window // 2, 3)
    if not mask.any():
        return pd.Series(dtype=float)
    total = valid.clip(lower=1)
    return ((up - down) / total)[mask].rename("PM-DRIFT")


def compute_spread(markets: List[Dict], prices: pd.DataFrame) -> pd.Series:
    """
    PM-SPREAD: 相对买卖价差 = spread / mid_price。

    值越大 = 市场价差越宽 = 流动性越差 = 潜在错误定价空间越大。
    Amihud 的改进版：直接用订单簿价差而非价格冲击近似。
    使用 Gamma API 的 spread 字段（无需额外 API 调用）。
    """
    result: Dict[str, float] = {}
    for m in markets:
        label = m.get("question", "")[:60].strip()
        if not label or label not in prices.columns:
            continue
        try:
            raw_spread = m.get("spread")
            if raw_spread is None:
                continue
            spread = float(raw_spread)
            # Use mid_price for normalization
            bid = m.get("bestBid")
            ask = m.get("bestAsk")
            if bid is not None and ask is not None:
                mid = (float(bid) + float(ask)) / 2
            elif ask is not None:
                mid = float(ask)
            else:
                mid = float(prices.iloc[-1].get(label, 0.5))
            mid = max(abs(mid), 0.01)
            result[label] = float(spread / mid)
        except Exception:
            pass
    return pd.Series(result, dtype=float).rename("PM-SPREAD")


def compute_volatility(
    prices: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """
    PM-VOL: 21 日已实现波动率（log 收益率标准差 × √365 年化）。

    高波动率 = 价格更不确定 → 可能反映噪声交易或信息到达加速。
    我们检验高波动市场相对于低波动市场是否有系统性的方向偏差。
    预测市场 7 天/周交易，年化用 365 而非股票市场的 252。
    """
    if len(prices) < window + 1:
        return pd.Series(dtype=float)
    log_ret = np.log(prices / prices.shift(1)).iloc[-window:]
    vol     = log_ret.std() * np.sqrt(365)
    return vol.dropna().rename("PM-VOL")


# ---------------------------------------------------------------------------
# IC (Information Coefficient) walk-forward
# ---------------------------------------------------------------------------

def compute_ic_series(
    prices: pd.DataFrame,
    factor_fn: Callable[[pd.DataFrame], pd.Series],
    forward_days: int = IC_FORWARD_DAYS,
    step: int = 5,
) -> pd.Series:
    """
    Walk-Forward Spearman IC 时间序列。

    每隔 step 个交易日：
      1. 用 prices[:t] 计算截面因子得分
      2. 计算 prices[t:t+forward_days] 的前向收益
      3. IC_t = Spearman_corr(scores, fwd_returns)

    参数
    ----
    factor_fn : 接受价格矩阵，返回市场得分 Series 的函数
    forward_days : 前向收益天数（预测窗口）
    step : 每 step 个交易日计算一次 IC

    返回
    ----
    pd.Series  index=date, value=IC（NaN 表示该期市场数不足）
    """
    dates  = prices.index
    n      = len(dates)
    ic_records: Dict[pd.Timestamp, float] = {}

    min_history = 30  # 计算因子需要的最少历史天数

    for i in range(min_history, n - forward_days, step):
        t_date = dates[i]
        try:
            # 截面因子得分（仅用 prices[:i+1] 即到 t 为止的历史）
            hist_prices = prices.iloc[:i + 1]
            scores      = factor_fn(hist_prices).dropna()
            if len(scores) < IC_MIN_MARKETS:
                continue

            # 前向收益：t 到 t+forward_days
            fwd_slice   = prices.iloc[i: i + forward_days + 1]
            fwd_returns = fwd_slice.pct_change(forward_days).iloc[-1].dropna()

            # 对齐
            common  = scores.index.intersection(fwd_returns.index)
            if len(common) < IC_MIN_MARKETS:
                continue

            s_aligned = scores[common]
            r_aligned = fwd_returns[common]

            corr, _ = stats.spearmanr(s_aligned.values, r_aligned.values)
            if not np.isnan(corr):
                ic_records[t_date] = float(corr)

        except Exception as exc:
            logger.debug("IC computation at %s failed: %s", t_date.date(), exc)

    return pd.Series(ic_records, dtype=float).sort_index()


def _summarize_ic(ic_series: pd.Series) -> Tuple[float, float, float]:
    """Returns (mean_ic, ic_tstat, ic_pvalue)."""
    valid = ic_series.dropna()
    n = len(valid)
    if n < 3:
        return 0.0, 0.0, 1.0
    mean_ic = float(valid.mean())
    std_ic  = float(valid.std())
    tstat   = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0
    pvalue  = float(2 * (1 - stats.t.cdf(abs(tstat), df=n - 1)))
    return mean_ic, tstat, pvalue


# ---------------------------------------------------------------------------
# Decile portfolio
# ---------------------------------------------------------------------------

def compute_decile_portfolio(
    prices: pd.DataFrame,
    factor_fn: Callable[[pd.DataFrame], pd.Series],
    forward_days: int = IC_FORWARD_DAYS,
    step: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Walk-Forward 十分位组合：
      - 按因子得分将市场分成 N_DECILES 组（1=低分，N_DECILES=高分）
      - 统计各分位的平均前向收益率
      - 多空组合净值 = long top 20% - short bottom 20%

    返回
    ----
    decile_returns : pd.Series  index=1..N_DECILES, value=平均收益率(%)
    portfolio_equity : pd.Series  index=date, value=多空组合净值
    """
    dates   = prices.index
    n       = len(dates)
    min_history = 30

    # 累积多空收益
    ls_returns: Dict[pd.Timestamp, float] = {}
    # 每个分位的收益列表
    decile_ret_lists: Dict[int, List[float]] = {d: [] for d in range(1, N_DECILES + 1)}

    for i in range(min_history, n - forward_days, step):
        t_date = dates[i]
        try:
            hist_prices = prices.iloc[:i + 1]
            scores      = factor_fn(hist_prices).dropna()
            if len(scores) < N_DECILES * 2:   # 每个分位至少 2 个市场
                continue

            fwd_slice   = prices.iloc[i: i + forward_days + 1]
            fwd_returns = fwd_slice.pct_change(forward_days).iloc[-1].dropna()
            common      = scores.index.intersection(fwd_returns.index)
            if len(common) < N_DECILES * 2:
                continue

            s_aligned = scores[common].sort_values()
            r_aligned = fwd_returns[common]

            # Split into deciles
            n_mkt   = len(s_aligned)
            indices = np.array_split(np.arange(n_mkt), N_DECILES)
            for d_idx, idx_group in enumerate(indices, start=1):
                mkt_labels = s_aligned.iloc[idx_group].index
                avg_ret    = float(r_aligned[mkt_labels].mean())
                decile_ret_lists[d_idx].append(avg_ret)

            # Long top 20%, short bottom 20%
            top_n   = max(n_mkt // 5, 1)
            long_r  = float(r_aligned[s_aligned.iloc[-top_n:].index].mean())
            short_r = float(r_aligned[s_aligned.iloc[:top_n].index].mean())
            ls_returns[t_date] = long_r - short_r

        except Exception as exc:
            logger.debug("Decile portfolio at %s failed: %s", t_date.date(), exc)

    # Decile average returns (%)
    decile_returns = pd.Series(
        {d: float(np.mean(v)) * 100 if v else 0.0
         for d, v in decile_ret_lists.items()},
        dtype=float,
    )

    # Long-short equity curve
    ls_ser = pd.Series(ls_returns, dtype=float).sort_index()
    if ls_ser.empty:
        portfolio_equity = pd.Series(dtype=float)
    else:
        portfolio_equity = INITIAL_FACTOR_EQUITY * (1 + ls_ser).cumprod()

    return decile_returns, portfolio_equity


# ---------------------------------------------------------------------------
# Factor score cross-correlation matrix (factor independence test)
# ---------------------------------------------------------------------------

def compute_factor_correlation(
    factor_scores: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    计算各因子截面得分之间的 Spearman 相关矩阵，验证因子独立性。
    理想情况下，各因子相关性应低于 0.3。
    使用 pairwise dropna（min_periods=5）避免任一因子得分缺失时矩阵变空。
    """
    non_empty = {k: v for k, v in factor_scores.items() if v.dropna().shape[0] >= 5}
    if len(non_empty) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(non_empty)
    # pairwise Spearman: each pair uses only markets where BOTH have valid scores
    cols = list(df.columns)
    n = len(cols)
    mat = np.full((n, n), 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            mask = df[cols[i]].notna() & df[cols[j]].notna()
            if mask.sum() >= 5:
                val = float(stats.spearmanr(df[cols[i]][mask], df[cols[j]][mask])[0])
            else:
                val = float("nan")
            mat[i, j] = mat[j, i] = val
    return pd.DataFrame(mat, index=cols, columns=cols).round(3)


# ---------------------------------------------------------------------------
# Master runner — Phase 1
# ---------------------------------------------------------------------------

def run_all_factors(
    prices: pd.DataFrame,
    markets: List[Dict],
    volume_map: Dict[str, float],
) -> Dict[str, FactorResult]:
    """
    运行 Phase 1 全部三个因子（PM-MOM, PM-LIQ, PM-TTR）。
    PM-CAL 由 resolved_data.py 单独处理，不计算 IC（结构不同）。

    参数
    ----
    prices     : 日线价格矩阵（date × market_label）
    markets    : Gamma API active 市场元数据列表
    volume_map : {market_label: volume24hr}

    返回
    ----
    Dict[factor_name, FactorResult]
    """
    results: Dict[str, FactorResult] = {}

    # ── PM-MOM ──────────────────────────────────────────────────────────────
    logger.info("Computing PM-MOM…")
    mom_fn     = lambda p: compute_momentum(p, lookback=MOM_LOOKBACK)
    mom_scores = mom_fn(prices)
    mom_ic     = compute_ic_series(prices, mom_fn)
    mom_mean, mom_t, mom_p = _summarize_ic(mom_ic)
    mom_decile, mom_equity = compute_decile_portfolio(prices, mom_fn)

    results["PM-MOM"] = FactorResult(
        name            = "PM-MOM",
        description_zh  = f"概率动量（{MOM_LOOKBACK}日）",
        scores          = mom_scores,
        ic_series       = mom_ic,
        mean_ic         = mom_mean,
        ic_tstat        = mom_t,
        ic_pvalue       = mom_p,
        decile_returns  = mom_decile,
        portfolio_equity= mom_equity,
        n_markets_used  = len(mom_scores),
    )

    # ── PM-LIQ ──────────────────────────────────────────────────────────────
    logger.info("Computing PM-LIQ…")
    # Current snapshot: use actual API volume24hr for today's cross-section display.
    liq_scores = compute_liquidity(prices, volume_map)
    # Historical IC: use price-derived volume proxy — avoids look-ahead from
    # today's volume24hr being used to score historical periods.
    def liq_fn(p: pd.DataFrame) -> pd.Series:
        return compute_liquidity(p, _build_volume_proxy(p))
    liq_ic     = compute_ic_series(prices, liq_fn)
    liq_mean, liq_t, liq_p = _summarize_ic(liq_ic)
    liq_decile, liq_equity = compute_decile_portfolio(prices, liq_fn)

    results["PM-LIQ"] = FactorResult(
        name            = "PM-LIQ",
        description_zh  = "流动性（Amihud 非流动性）",
        scores          = liq_scores,
        ic_series       = liq_ic,
        mean_ic         = liq_mean,
        ic_tstat        = liq_t,
        ic_pvalue       = liq_p,
        decile_returns  = liq_decile,
        portfolio_equity= liq_equity,
        n_markets_used  = len(liq_scores),
    )

    # ── PM-TTR ──────────────────────────────────────────────────────────────
    logger.info("Computing PM-TTR…")
    # Today's snapshot for display
    ttr_scores = compute_ttr(markets).reindex(prices.columns).dropna()

    # Dynamic walk-forward: TTR is computed at the historical date of each slice,
    # NOT at today's date. This is the correct specification.
    def ttr_fn(p: pd.DataFrame) -> pd.Series:
        ref_date = p.index[-1]  # the historical "today" for this slice
        return compute_ttr(markets, reference_date=ref_date).reindex(p.columns).dropna()

    ttr_ic     = compute_ic_series(prices, ttr_fn)
    ttr_mean, ttr_t, ttr_p = _summarize_ic(ttr_ic)
    ttr_decile, ttr_equity = compute_decile_portfolio(prices, ttr_fn)

    results["PM-TTR"] = FactorResult(
        name            = "PM-TTR",
        description_zh  = "距结算时间 log(1+天数)",
        scores          = ttr_scores,
        ic_series       = ttr_ic,
        mean_ic         = ttr_mean,
        ic_tstat        = ttr_t,
        ic_pvalue       = ttr_p,
        decile_returns  = ttr_decile,
        portfolio_equity= ttr_equity,
        n_markets_used  = len(ttr_scores),
    )

    # ── PM-VOL ──────────────────────────────────────────────────────────────
    logger.info("Computing PM-VOL…")
    vol_fn     = lambda p: compute_volatility(p, window=21)
    vol_scores = vol_fn(prices)
    vol_ic     = compute_ic_series(prices, vol_fn)
    vol_mean, vol_t, vol_p = _summarize_ic(vol_ic)
    vol_decile, vol_equity = compute_decile_portfolio(prices, vol_fn)

    results["PM-VOL"] = FactorResult(
        name            = "PM-VOL",
        description_zh  = "已实现波动率（21日年化）",
        scores          = vol_scores,
        ic_series       = vol_ic,
        mean_ic         = vol_mean,
        ic_tstat        = vol_t,
        ic_pvalue       = vol_p,
        decile_returns  = vol_decile,
        portfolio_equity= vol_equity,
        n_markets_used  = len(vol_scores),
    )

    # ── PM-EXTR ─────────────────────────────────────────────────────────────
    logger.info("Computing PM-EXTR…")
    extr_fn     = lambda p: compute_extremity(p)
    extr_scores = extr_fn(prices)
    extr_ic     = compute_ic_series(prices, extr_fn)
    extr_mean, extr_t, extr_p = _summarize_ic(extr_ic)
    extr_decile, extr_equity = compute_decile_portfolio(prices, extr_fn)

    results["PM-EXTR"] = FactorResult(
        name            = "PM-EXTR",
        description_zh  = "价格极端性 2×|p−0.5|",
        scores          = extr_scores,
        ic_series       = extr_ic,
        mean_ic         = extr_mean,
        ic_tstat        = extr_t,
        ic_pvalue       = extr_p,
        decile_returns  = extr_decile,
        portfolio_equity= extr_equity,
        n_markets_used  = len(extr_scores),
    )

    # ── PM-DRIFT ─────────────────────────────────────────────────────────────
    logger.info("Computing PM-DRIFT…")
    drift_fn     = lambda p: compute_drift(p, window=14)
    drift_scores = drift_fn(prices)
    drift_ic     = compute_ic_series(prices, drift_fn)
    drift_mean, drift_t, drift_p = _summarize_ic(drift_ic)
    drift_decile, drift_equity = compute_decile_portfolio(prices, drift_fn)

    results["PM-DRIFT"] = FactorResult(
        name            = "PM-DRIFT",
        description_zh  = "方向漂移一致性（14日）",
        scores          = drift_scores,
        ic_series       = drift_ic,
        mean_ic         = drift_mean,
        ic_tstat        = drift_t,
        ic_pvalue       = drift_p,
        decile_returns  = drift_decile,
        portfolio_equity= drift_equity,
        n_markets_used  = len(drift_scores),
    )

    # ── PM-SPREAD ────────────────────────────────────────────────────────────
    logger.info("Computing PM-SPREAD…")
    # Current snapshot: use actual API bestBid/bestAsk for today's display.
    spread_scores = compute_spread(markets, prices)
    # Historical IC: bestBid/bestAsk are real-time snapshots — using them for
    # historical periods is time-contaminated.  Use a price-derived tick proxy.
    def spread_fn(p: pd.DataFrame) -> pd.Series:
        return _spread_proxy_from_prices(p)
    spread_ic     = compute_ic_series(prices, spread_fn)
    spread_mean, spread_t, spread_p = _summarize_ic(spread_ic)
    spread_decile, spread_equity = compute_decile_portfolio(prices, spread_fn)

    results["PM-SPREAD"] = FactorResult(
        name            = "PM-SPREAD",
        description_zh  = "相对买卖价差（bid-ask spread）",
        scores          = spread_scores,
        ic_series       = spread_ic,
        mean_ic         = spread_mean,
        ic_tstat        = spread_t,
        ic_pvalue       = spread_p,
        decile_returns  = spread_decile,
        portfolio_equity= spread_equity,
        n_markets_used  = len(spread_scores),
    )

    # ── MOM 多窗口对比（附加研究：不进入主结果，单独返回）──────────────────────
    logger.info("Computing multi-window MOM scan (7/14/21/30d)…")
    mom_windows: Dict[int, Tuple[float, float, float]] = {}
    for w in [7, 14, 21, 30]:
        fn  = lambda p, w=w: compute_momentum(p, lookback=w)
        ic  = compute_ic_series(prices, fn)
        m, t_, p_ = _summarize_ic(ic)
        mom_windows[w] = (m, t_, p_)
        logger.info("  MOM-%dd: IC=%.4f  t=%.2f", w, m, t_)

    # Store on the MOM result for the UI to access
    results["PM-MOM"].mom_windows = mom_windows

    logger.info(
        "Factor engine done | MOM IC=%.3f(t=%.2f) LIQ IC=%.3f(t=%.2f) "
        "TTR IC=%.3f(t=%.2f) VOL IC=%.3f(t=%.2f) "
        "EXTR IC=%.3f(t=%.2f) DRIFT IC=%.3f(t=%.2f) "
        "SPREAD IC=%.3f(t=%.2f)",
        mom_mean, mom_t, liq_mean, liq_t, ttr_mean, ttr_t, vol_mean, vol_t,
        extr_mean, extr_t, drift_mean, drift_t,
        spread_mean, spread_t,
    )
    return results


# ---------------------------------------------------------------------------
# Phase 2 stub: Fama-MacBeth two-step regression
# ---------------------------------------------------------------------------

def run_fama_macbeth(
    factor_scores_by_date: Dict[pd.Timestamp, pd.DataFrame],
    future_returns_by_date: Dict[pd.Timestamp, pd.Series],
    newey_west_lags: int = 6,
) -> FamaMacBethResult:
    """
    Fama-MacBeth 两步回归 (Phase 2)。

    Step 1 — 每期截面 OLS (statsmodels):
        r_{i,t→t+k} = α_t + Σ_k β_{k,t} F_{k,i,t} + ε

    Step 2 — 时序均值 + Newey-West 标准误:
        β̄_k = mean(β_{k,t})
        NW_SE = sandwich(β_{k,t}, lags=newey_west_lags)
        t-stat = β̄_k / NW_SE

    参数
    ----
    factor_scores_by_date : date → DataFrame(markets × factors) — 各期因子得分
    future_returns_by_date : date → Series(market → forward_return) — 各期前向收益
    newey_west_lags : Newey-West 自相关修正窗口

    返回
    ----
    FamaMacBethResult
    """
    import statsmodels.api as sm

    common_dates = sorted(
        set(factor_scores_by_date) & set(future_returns_by_date)
    )
    if len(common_dates) < 10:
        raise ValueError(
            f"Too few dates for FM regression ({len(common_dates)} < 10). "
            "Run more IC periods first."
        )

    factor_names = None
    beta_records: List[Dict] = []
    r2_list: List[float]     = []

    for date in common_dates:
        X_df  = factor_scores_by_date[date]
        y_ser = future_returns_by_date[date].dropna()

        common_mkt = X_df.index.intersection(y_ser.index)
        if len(common_mkt) < IC_MIN_MARKETS:
            continue

        X_raw = X_df.loc[common_mkt]
        y     = y_ser.loc[common_mkt]

        # 1. Fill missing factor scores with cross-sectional mean (standard FM practice).
        #    Markets missing one factor (e.g. PM-SPREAD) keep the mean = 0 after z-score,
        #    so they don't distort the regression but are still included.
        X = X_raw.apply(lambda col: col.fillna(col.mean()))

        # 2. Drop any column entirely NaN this period (factor produced no data)
        X = X.dropna(axis=1, how="all")

        # 3. Cross-sectional z-score: each factor → mean=0, std=1 within this period.
        #    This makes betas directly comparable across factors and periods.
        col_std = X.std().clip(lower=1e-8)
        X = (X - X.mean()) / col_std

        # 4. Drop markets still NaN after imputation (all their factors were NaN)
        valid_rows = X.notna().all(axis=1)
        X = X[valid_rows]
        y = y[valid_rows]

        if len(X) < IC_MIN_MARKETS:
            continue

        try:
            X_const = sm.add_constant(X, has_constant="add")
            res     = sm.OLS(y, X_const).fit()
            cols    = ["const"] + list(X.columns)
            row     = dict(zip(cols, res.params.tolist()))
            row["date"] = date
            row["_r2"]  = float(res.rsquared)   # store per-period R²
            row["_n"]   = int(len(y))            # sample size per period
            beta_records.append(row)
            r2_list.append(res.rsquared)
        except Exception as exc:
            logger.debug("FM Step1 at %s failed: %s", date, exc)

    if not beta_records:
        raise ValueError("FM regression: no valid periods produced betas.")

    beta_df = pd.DataFrame(beta_records).set_index("date")
    mean_r2 = float(np.mean(r2_list))

    # Collect all factor names that appeared in at least one period
    _internal = {"const", "_r2", "_n"}
    factor_names = sorted(c for c in beta_df.columns if c not in _internal)

    # Step 2: Newey-West adjusted means
    premia: Dict[str, float]  = {}
    tstats: Dict[str, float]  = {}
    pvals:  Dict[str, float]  = {}

    for col in beta_df.columns:
        b_series = beta_df[col].dropna()
        n        = len(b_series)
        if n < 3:
            premia[col] = tstats[col] = pvals[col] = np.nan
            continue
        mean_b = float(b_series.mean())

        # Newey-West HAC variance
        try:
            from statsmodels.stats.sandwich_covariance import cov_hac
            nw_var = float(cov_hac(
                sm.OLS(b_series, np.ones(n)).fit(),
                nlags=newey_west_lags
            )[0, 0])
            # cov_hac returns Var(β̄) directly — do NOT divide by n again
            nw_se  = np.sqrt(nw_var)
        except Exception:
            # Fallback: standard OLS SE
            nw_se = float(b_series.std() / np.sqrt(n))

        tstat = mean_b / nw_se if nw_se > 0 else 0.0
        pval  = float(2 * (1 - stats.t.cdf(abs(tstat), df=n - 1)))
        premia[col]  = round(mean_b, 6)
        tstats[col]  = round(tstat, 3)
        pvals[col]   = round(pval, 4)

    factor_cols = [c for c in factor_names if c in premia]
    # Include per-period diagnostic columns in beta_time_series
    ts_cols = factor_cols + [c for c in ["_r2", "_n"] if c in beta_df.columns]
    return FamaMacBethResult(
        factor_names      = factor_cols,
        factor_premia     = pd.Series({k: premia[k] for k in factor_cols}),
        t_statistics      = pd.Series({k: tstats[k] for k in factor_cols}),
        p_values          = pd.Series({k: pvals[k]  for k in factor_cols}),
        mean_r_squared    = mean_r2,
        alpha             = premia.get("const", float("nan")),
        alpha_tstat       = tstats.get("const", float("nan")),
        beta_time_series  = beta_df[ts_cols] if ts_cols else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# Phase 2: joint walk-forward input builder for Fama-MacBeth
# ---------------------------------------------------------------------------

def build_fm_inputs(
    prices: pd.DataFrame,
    markets: List[Dict],
    volume_map: Dict[str, float],
    forward_days: int = IC_FORWARD_DAYS,
    step: int = 1,
) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], Dict[pd.Timestamp, pd.Series]]:
    """
    联合 Walk-Forward：遍历时间轴一次，同时计算所有 8 个因子截面得分
    和对应的前向收益，作为 run_fama_macbeth() 的输入。

    比分别跑 8 个独立 IC 循环更高效，且保证截面对齐。

    返回
    ----
    scores_by_date  : date → DataFrame(markets × 8 factors)
    returns_by_date : date → Series(market → 5d forward return)
    """
    dates       = prices.index
    n           = len(dates)
    min_history = 30

    scores_by_date:  Dict[pd.Timestamp, pd.DataFrame] = {}
    returns_by_date: Dict[pd.Timestamp, pd.Series]    = {}

    for i in range(min_history, n - forward_days, step):
        t_date = dates[i]
        try:
            hist = prices.iloc[:i + 1]

            # All factors computed at historical "today" = t_date.
            # PM-LIQ uses price-derived volume proxy (no look-ahead from today's API data).
            # PM-SPREAD uses price-derived tick proxy (no look-ahead from today's order book).
            score_map: Dict[str, pd.Series] = {
                "PM-MOM":    compute_momentum(hist, lookback=MOM_LOOKBACK),
                "PM-VOL":    compute_volatility(hist, window=21),
                "PM-EXTR":   compute_extremity(hist),
                "PM-DRIFT":  compute_drift(hist, window=14),
                "PM-LIQ":    compute_liquidity(hist, _build_volume_proxy(hist)),
                "PM-TTR":    compute_ttr(markets, reference_date=t_date)
                              .reindex(hist.columns).dropna(),
                "PM-SPREAD": _spread_proxy_from_prices(hist),
            }

            X = pd.DataFrame(score_map).dropna(how="all")
            if len(X) < IC_MIN_MARKETS:
                continue

            fwd_returns = (
                prices.iloc[i: i + forward_days + 1]
                .pct_change(forward_days).iloc[-1].dropna()
            )

            common = X.index.intersection(fwd_returns.index)
            if len(common) < IC_MIN_MARKETS:
                continue

            scores_by_date[t_date]  = X.loc[common]
            returns_by_date[t_date] = fwd_returns.loc[common]

        except Exception as exc:
            logger.debug("FM inputs at %s failed: %s", t_date.date(), exc)

    logger.info(
        "FM inputs: %d periods, avg %.0f markets/period",
        len(scores_by_date),
        np.mean([len(v) for v in scores_by_date.values()]) if scores_by_date else 0,
    )
    return scores_by_date, returns_by_date


def compute_fm_composite_score(
    factor_results: Dict[str, "FactorResult"],
    fm_result: "FamaMacBethResult",
) -> pd.Series:
    """
    FM 合成得分：用 FM beta 加权的截面 z-score 线性组合。

    对每个当前市场：
        FM_score_i = Σ_k β̄_k × z_k(i)   [单位：预测 5日收益，bp]

    其中 z_k(i) = (score_k(i) - mean_k) / std_k 是当日截面 z-score，
    β̄_k 是 Fama-MacBeth 估计的因子风险溢价（bp/5d per 1σ）。

    高 FM_score → 模型预测正收益；低 FM_score → 模型预测负收益。
    """
    if fm_result is None or not fm_result.factor_names:
        return pd.Series(dtype=float)

    score_df = pd.DataFrame({
        k: factor_results[k].scores
        for k in fm_result.factor_names
        if k in factor_results
    })
    if score_df.empty:
        return pd.Series(dtype=float)

    # Cross-sectional z-score of today's factor scores
    col_std = score_df.std().clip(lower=1e-8)
    z = (score_df - score_df.mean()) / col_std

    # Weighted sum by FM betas (convert to bp)
    betas = pd.Series({k: fm_result.factor_premia[k] * 10000
                       for k in fm_result.factor_names if k in z.columns})
    composite = z[betas.index].mul(betas).sum(axis=1)
    return composite.dropna().rename("FM-COMPOSITE")
