"""
扩大样本的因子研究 — 基于本地 DuckDB parquet 数据集（polymarket-users）。

相对 API 版本（optimize_three_factor_and_strategies.py）的改进
------------------------------------------------------------
1. 样本：~4000 个市场 × ~570 天（vs API 的 125 × 197），含已结算市场
   → 消除幸存者偏差，IC 观测点从 ~33 → ~100+
2. PM-LIQ 用真实日成交量算 Amihud（vs 价格活动代理）
3. PM-TTR 用 markets.parquet 的 close_time（vs 今日 API 快照 endDate）
4. Fama-MacBeth 两步回归（Phase 2）首次在足量数据上运行
5. 方法论与 2026-06-10 修复后一致：
   - 方向（orientation）在前半段估计，后半段评估
   - LS Sharpe 按实际调仓间隔年化
   - 净收益扣除全换手成本（每次调仓双腿各一次往返）

关键方法论（与 API 管线不同，修正其机制性假象）：
   - 不做价格 clip；形成日要求 0.05 < p < 0.95 且当日有成交（消除截断假象）
   - skip-day：t 决策、t+1 收盘入场、收益量 t+1 → t+1+5（消除端点噪声共享）
   - 结算收益入样：持有窗口内结束的市场用最后成交价作退出价（结算代理；
     outcome_yes 字段经验证不代表结算结果，不可用）
     （消除"提前结算市场被剔除"的幸存者过滤泄漏）
   - 双口径 IC：原始 Δp 与价格中性 Δp（对 [1,p,p²] 残差化）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

import db
from config import (
    MIN_DAILY_VOLUME_USD,
    TRANSACTION_COST,
)
from factor_engine import run_fama_macbeth

START, END = "2024-09-01", "2026-03-29"
MIN_OBS      = 60       # min daily observations per market in window
MAX_MARKETS  = 4000     # cap, ranked by total dollar volume
FWD          = 5        # forward-return horizon (days)
STEP         = 5        # evaluation every STEP days (non-overlapping returns)
MIN_HIST     = 35       # warmup before first evaluation date
MIN_XS       = 50       # min cross-section size per evaluation date

# Formation-date tradeability filter.  Prices are deliberately NOT clipped:
# clipping to [0.03, 0.97] censors forward returns at the boundaries (a market
# pinned at the 0.03 floor can only go up, one at the 0.97 cap can only go
# down), which manufactures enormous fake reversal/extremity ICs.  Instead we
# require a mid-range, traded price at formation.
MIN_PX, MAX_PX = 0.05, 0.95
MOM_LB       = 30
VOL_WIN      = 21
DRIFT_WIN    = 14
LIQ_WIN      = 30
SPREAD_WIN   = 60

FACTORS = ["PM-MOM", "PM-VOL", "PM-EXTR", "PM-DRIFT", "PM-LIQ", "PM-TTR", "PM-SPREAD"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panel() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (prices, volume, meta) — wide daily panels keyed by market_id."""
    sql = f"""
    WITH px AS (
      SELECT o.market_id,
             CAST(o.timestamp AS DATE)   AS d,
             o.close                     AS p,
             o.volume                    AS v,
             m.category                  AS category,
             m.question                  AS question,
             m.outcome_yes               AS outcome_yes,
             CAST(m.close_time AS DATE)  AS close_d
      FROM '{db.USERS_BASE}/ohlcv_1d.parquet' o
      JOIN '{db.USERS_BASE}/markets.parquet' m
        ON o.market_id = CAST(m.market_id AS VARCHAR)
      WHERE o.outcome IS NULL
        AND o.timestamp >= '{START}' AND o.timestamp < '{END}'
        AND m.close_time IS NOT NULL
        AND o.timestamp <= m.close_time
    ),
    keep AS (
      SELECT market_id
      FROM px
      GROUP BY market_id
      HAVING count(*) >= {MIN_OBS} AND avg(v) >= {MIN_DAILY_VOLUME_USD}
      ORDER BY sum(v) DESC
      LIMIT {MAX_MARKETS}
    )
    SELECT px.* FROM px JOIN keep USING (market_id)
    """
    long = db.query(sql)
    print(f"Loaded {len(long):,} daily rows, {long['market_id'].nunique():,} markets")

    prices = long.pivot_table(index="d", columns="market_id", values="p").sort_index()
    volume = long.pivot_table(index="d", columns="market_id", values="v").sort_index()
    meta = long.groupby("market_id").agg(
        category=("category", "first"),
        question=("question", "first"),
        close_d=("close_d", "first"),
        outcome_yes=("outcome_yes", "first"),
    )

    prices.index = pd.to_datetime(prices.index)
    volume.index = pd.to_datetime(volume.index)
    meta["close_d"] = pd.to_datetime(meta["close_d"])

    # Fill short gaps only (<=3d, same as pair pipeline).  No price clipping —
    # see MIN_PX/MAX_PX comment above.
    prices = prices.ffill(limit=3)
    volume = volume.reindex(prices.index)
    return prices, volume, meta


# ---------------------------------------------------------------------------
# Vectorized factor computations (hist = data up to and including date t)
# ---------------------------------------------------------------------------

def f_mom(hp: pd.DataFrame) -> pd.Series:
    if len(hp) < MOM_LB + 1:
        return pd.Series(dtype=float)
    return (hp.iloc[-1] / hp.iloc[-(MOM_LB + 1)].replace(0, np.nan) - 1).dropna()


def f_vol(hp: pd.DataFrame) -> pd.Series:
    if len(hp) < VOL_WIN + 1:
        return pd.Series(dtype=float)
    lr = np.log(hp.iloc[-(VOL_WIN + 1):] / hp.iloc[-(VOL_WIN + 1):].shift(1))
    return (lr.std() * np.sqrt(365)).dropna()


def f_extr(hp: pd.DataFrame) -> pd.Series:
    return (2 * (hp.iloc[-1].clip(0, 1) - 0.5).abs()).dropna()


def f_drift(hp: pd.DataFrame) -> pd.Series:
    if len(hp) < DRIFT_WIN + 2:
        return pd.Series(dtype=float)
    rets = hp.iloc[-(DRIFT_WIN + 1):].pct_change()
    up, down = (rets > 0).sum(), (rets < 0).sum()
    valid = up + down
    mask = valid >= max(DRIFT_WIN // 2, 3)
    return ((up - down) / valid.clip(lower=1))[mask]


def f_liq(hp: pd.DataFrame, hv: pd.DataFrame) -> pd.Series:
    """Amihud ILLIQ from REAL volume: mean(|ret_d| / $vol_d) over trailing window."""
    if len(hp) < LIQ_WIN + 1:
        return pd.Series(dtype=float)
    ret = hp.iloc[-(LIQ_WIN + 1):].pct_change().abs()
    vol = hv.iloc[-(LIQ_WIN + 1):].where(hv.iloc[-(LIQ_WIN + 1):] > 0)
    daily_illiq = ret / vol
    valid = daily_illiq.notna().sum()
    illiq = daily_illiq.mean()
    return illiq[valid >= 8].dropna()


def f_ttr(meta: pd.DataFrame, t: pd.Timestamp, cols: pd.Index) -> pd.Series:
    days = (meta["close_d"] - t).dt.days.clip(lower=0)
    return np.log1p(days).reindex(cols).dropna()


def f_spread(hp: pd.DataFrame) -> pd.Series:
    """Tick-size spread proxy, vectorized: 2 x p10(|dp| > 1e-4) / last price."""
    win = hp.iloc[-SPREAD_WIN:]
    d = win.diff().abs()
    d = d.where(d > 1e-4)
    q10 = d.quantile(0.10)
    n_valid = d.notna().sum()
    mid = win.ffill().iloc[-1].clip(lower=0.01)
    out = 2.0 * q10 / mid
    return out[n_valid >= 3].dropna()


# ---------------------------------------------------------------------------
# Walk-forward panel build
# ---------------------------------------------------------------------------

def _neutralize_price_level(r: pd.Series, p: pd.Series) -> pd.Series:
    """Residualize Δp on [1, p, p²] cross-sectionally — removes the mechanical
    price-level structure of probability changes (bounded asymmetry, longshot
    drift) so factor ICs measure predictability BEYOND price level."""
    X = np.column_stack([np.ones(len(p)), p.values, p.values ** 2])
    beta, *_ = np.linalg.lstsq(X, r.values, rcond=None)
    return pd.Series(r.values - X @ beta, index=r.index)


def build_scores_and_returns(prices, volume, meta):
    dates = prices.index
    scores_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    returns_by_date: Dict[pd.Timestamp, pd.Series] = {}    # Δp, skip-day
    returns_pn_by_date: Dict[pd.Timestamp, pd.Series] = {} # price-neutral Δp
    prices_by_date: Dict[pd.Timestamp, pd.Series] = {}     # entry price p_{t+1}

    # Full-ffill grid: for markets whose data ends inside the holding window
    # (early settlement), the final traded price is carried forward and used
    # as the settlement proxy (convergence ⇒ final price ≈ payout).  NOTE:
    # markets.parquet outcome_yes does NOT encode settlement (validated:
    # avg final price 0.34 for False vs 0.38 for True) — do not use it.
    prices_ff = prices.ffill()

    # Skip-day convention: factors use data up to t (decision), the position
    # is entered at the NEXT close p_{t+1}, and the return runs t+1 → t+1+FWD.
    # Without the skip, the endpoint noise ε_t sits in the factor score (+)
    # and in the return base (−), manufacturing huge fake reversal ICs.
    for i in range(MIN_HIST, len(dates) - FWD - 1, STEP):
        t = dates[i]
        p_t = prices.iloc[i]
        v_t = volume.iloc[i]

        # Tradeable universe at t: mid-range price AND actual trades that day.
        live = p_t[(p_t > MIN_PX) & (p_t < MAX_PX) & (v_t.fillna(0) > 0)].index
        if len(live) < MIN_XS:
            continue
        hp = prices.iloc[: i + 1][live]
        hv = volume.iloc[: i + 1][live]

        X = pd.DataFrame({
            "PM-MOM":    f_mom(hp),
            "PM-VOL":    f_vol(hp),
            "PM-EXTR":   f_extr(hp),
            "PM-DRIFT":  f_drift(hp),
            "PM-LIQ":    f_liq(hp, hv),
            "PM-TTR":    f_ttr(meta, t, hp.columns),
            "PM-SPREAD": f_spread(hp),
        }).dropna(how="all")

        # Forward return = Δ probability from the entry close (skip-day),
        # SETTLEMENT-INCLUSIVE: markets whose data ends inside the holding
        # window exit at their final traded price (settlement proxy).
        # Requiring a fresh price at t+1+FWD instead would silently drop
        # early-resolved markets — a survival filter conditioned on future
        # information (cheap markets that crashed and settled NO would
        # vanish from the sample, leaving only the survivors).
        d1, d2 = dates[i + 1], dates[i + 1 + FWD]
        p_entry = prices.iloc[i + 1]

        # Exit marks at the last traded price as of d2 for EVERY name (full
        # ffill).  Conditioning the exit on fresh trading or on close_time
        # is an asymmetric survival filter: losers drift to worthlessness
        # and quietly stop trading (dropped), winners keep trading (kept).
        exit_px = prices_ff.iloc[i + 1 + FWD]
        fwd = (exit_px - p_entry).dropna()
        # must still be open at the entry close
        alive = (meta["close_d"] > d1).reindex(fwd.index).fillna(False)
        fwd = fwd[alive]

        # Entry must be executable: the entry day actually TRADED and the
        # price is still mid-range.  Without this, p_{t+1} can be a stale
        # forward-filled quote — "buying" at a pre-news price right before a
        # resolution jump, the classic fake-alpha in prediction-market
        # backtests.
        v_entry = volume.iloc[i + 1]
        entry_ok = p_entry[
            (p_entry > MIN_PX) & (p_entry < MAX_PX) & (v_entry.fillna(0) > 0)
        ].index

        common = X.index.intersection(fwd.index).intersection(entry_ok)
        if len(common) < MIN_XS:
            continue
        scores_by_date[t] = X.loc[common]
        returns_by_date[t] = fwd.loc[common]
        returns_pn_by_date[t] = _neutralize_price_level(
            fwd.loc[common], p_entry.loc[common]
        )
        prices_by_date[t] = p_entry.loc[common]

    print(f"Walk-forward: {len(scores_by_date)} evaluation dates, "
          f"avg cross-section {np.mean([len(v) for v in scores_by_date.values()]):.0f} markets")
    return scores_by_date, returns_by_date, returns_pn_by_date, prices_by_date


# ---------------------------------------------------------------------------
# IC / portfolio analytics
# ---------------------------------------------------------------------------

def ic_series_for(factor: str, scores_by_date, returns_by_date) -> pd.Series:
    out = {}
    for t, X in scores_by_date.items():
        s = X[factor].dropna()
        r = returns_by_date[t].reindex(s.index).dropna()
        s = s.reindex(r.index)
        if len(s) < MIN_XS:
            continue
        c, _ = sps.spearmanr(s.values, r.values)
        if np.isfinite(c):
            out[t] = float(c)
    return pd.Series(out).sort_index()


def summarize_ic(ic: pd.Series) -> Tuple[float, float, float, int]:
    ic = ic.dropna()
    n = len(ic)
    if n < 3:
        return 0.0, 0.0, 1.0, n
    m, s = float(ic.mean()), float(ic.std())
    t = m / (s / np.sqrt(n)) if s > 0 else 0.0
    p = float(2 * (1 - sps.t.cdf(abs(t), df=n - 1)))
    return m, t, p, n


def bh_fdr(p: pd.Series) -> pd.Series:
    p = p.fillna(1.0).astype(float)
    n = len(p)
    ranked = p.sort_values()
    q = ranked * n / pd.Series(range(1, n + 1), index=ranked.index)
    q = q[::-1].cummin()[::-1].clip(upper=1.0)
    out = pd.Series(index=p.index, dtype=float)
    out.loc[q.index] = q.values
    return out


def ls_portfolio(signal_by_date, returns_by_date, prices_by_date, eval_dates) -> Dict[str, float]:
    """
    Quintile long-short on `signal` (high = long), token economics:
      long leg  : buy YES at p  → period return Δp / p
      short leg : buy NO at 1-p → period return -Δp / (1 - p)
    Each leg gets half the capital.  Deciles are reported in Δp (prob points).
    """
    ls = {}
    diag = {"long": [], "short": [], "p_long": [], "p_short": []}
    dec_rets = {d: [] for d in range(1, 11)}
    for t in eval_dates:
        s = signal_by_date.get(t)
        if s is None:
            continue
        r = returns_by_date[t].reindex(s.index).dropna()   # Δp
        s = s.reindex(r.index).dropna()
        r = r.reindex(s.index)
        p = prices_by_date[t].reindex(s.index)
        if len(s) < MIN_XS:
            continue
        s_sorted = s.sort_values()
        n = len(s_sorted)
        for d_idx, grp in enumerate(np.array_split(np.arange(n), 10), start=1):
            dec_rets[d_idx].append(float(r[s_sorted.iloc[grp].index].mean()))
        top = max(n // 5, 1)
        long_idx  = s_sorted.iloc[-top:].index
        short_idx = s_sorted.iloc[:top].index
        long_ret  = float((r[long_idx] / p[long_idx]).mean())
        short_ret = float((-r[short_idx] / (1.0 - p[short_idx])).mean())
        ls[t] = 0.5 * (long_ret + short_ret)
        diag["long"].append(long_ret)
        diag["short"].append(short_ret)
        diag["p_long"].append(float(p[long_idx].mean()))
        diag["p_short"].append(float(p[short_idx].mean()))

    ls = pd.Series(ls).sort_index()
    if len(ls) < 3:
        return {}
    spacing = float(np.median(np.diff(ls.index.values).astype("timedelta64[D]").astype(float)))
    ppy = 365.0 / max(spacing, 1.0)
    net = ls - 2.0 * TRANSACTION_COST
    out = {
        "n_periods": len(ls),
        "gross_per_period_pct": float(ls.mean() * 100),
        "gross_sharpe": float(ls.mean() / ls.std() * np.sqrt(ppy)) if ls.std() > 0 else 0.0,
        "gross_total_pct": float(((1 + ls).prod() - 1) * 100),
        "net_per_period_pct": float(net.mean() * 100),
        "net_sharpe": float(net.mean() / net.std() * np.sqrt(ppy)) if net.std() > 0 else 0.0,
        "net_total_pct": float(((1 + net).prod() - 1) * 100),
        "long_leg_pct": float(np.mean(diag["long"]) * 100),
        "short_leg_pct": float(np.mean(diag["short"]) * 100),
        "avg_p_long": float(np.mean(diag["p_long"])),
        "avg_p_short": float(np.mean(diag["p_short"])),
        "median_period_pct": float(ls.median() * 100),
        "decile_returns_pct": {d: float(np.mean(v)) * 100 for d, v in dec_rets.items() if v},
    }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)

    prices, volume, meta = load_panel()
    print(f"Panel: {prices.shape[0]} days x {prices.shape[1]} markets "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    scores_by_date, returns_by_date, returns_pn_by_date, prices_by_date = \
        build_scores_and_returns(prices, volume, meta)
    all_dates = sorted(scores_by_date)
    split = len(all_dates) // 2
    train_dates, eval_dates = all_dates[:split], all_dates[split:]
    print(f"Orientation/train: {len(train_dates)} dates "
          f"({train_dates[0].date()} → {train_dates[-1].date()}); "
          f"eval: {len(eval_dates)} dates "
          f"({eval_dates[0].date()} → {eval_dates[-1].date()})")

    # ── Per-factor IC: raw Δp AND price-neutral Δp (eval half) ───────────
    # Raw IC includes the mechanical price-level structure of Δp; the
    # price-neutral IC is the honest "predictability beyond price level"
    # number.  PM-EXTR is itself a price-level factor, so its neutral IC is
    # ~0 by construction — its raw IC measures the longshot-bias channel.
    rows = []
    orientation: Dict[str, int] = {}
    for f in FACTORS:
        ic = ic_series_for(f, scores_by_date, returns_by_date)
        ic_pn = ic_series_for(f, scores_by_date, returns_pn_by_date)
        m_full, t_full, p_full, n_full = summarize_ic(ic)
        ic_train = ic[ic.index.isin(train_dates)]
        m_tr = float(ic_train.mean()) if len(ic_train) else 0.0
        orientation[f] = 1 if m_tr >= 0 else -1
        m_ev, t_ev, p_ev, n_ev = summarize_ic(ic[ic.index.isin(eval_dates)])
        m_pn, t_pn, p_pn, _ = summarize_ic(ic_pn[ic_pn.index.isin(eval_dates)])
        rows.append({
            "factor": f,
            "mean_ic": m_full, "tstat": t_full, "n_ic_points": n_full,
            "train_ic": m_tr, "orient": orientation[f],
            "eval_ic": m_ev, "eval_tstat": t_ev, "eval_pvalue": p_ev,
            "eval_ic_pn": m_pn, "eval_tstat_pn": t_pn, "eval_pvalue_pn": p_pn,
        })
    factor_df = pd.DataFrame(rows).sort_values("eval_pvalue_pn")
    factor_df["fdr_q_pn"] = bh_fdr(factor_df["eval_pvalue_pn"])
    factor_df.to_csv(out_dir / "expanded_factor_summary.csv", index=False)
    print("\n=== Expanded-sample factor summary (raw Δp IC / price-neutral IC, skip-day) ===")
    print(factor_df.to_string(index=False))

    # ── Single-factor LS portfolios on eval half (orientation from train) ─
    print("\n=== Single-factor quintile L/S on eval half (oriented by train half) ===")
    port_rows = {}
    for f in FACTORS:
        sig = {t: X[f].dropna() * orientation[f] for t, X in scores_by_date.items()}
        res = ls_portfolio(sig, returns_by_date, prices_by_date, eval_dates)
        if res:
            port_rows[f] = {k: v for k, v in res.items() if k != "decile_returns_pct"}
    port_df = pd.DataFrame(port_rows).T.sort_values("net_sharpe", ascending=False)
    port_df.to_csv(out_dir / "expanded_factor_portfolios.csv")
    print(port_df.round(3).to_string())

    # ── Best-combo validation: PM-EXTR + PM-DRIFT + PM-SPREAD composite ──
    combo = ("PM-EXTR", "PM-DRIFT", "PM-SPREAD")
    comp_by_date = {}
    for t, X in scores_by_date.items():
        z = pd.DataFrame(index=X.index)
        for f in combo:
            s = X[f].dropna()
            if len(s) < 5 or s.std() <= 0:
                continue
            z[f] = (X[f] - s.mean()) / s.std() * orientation[f]
        if not z.empty:
            comp_by_date[t] = z.mean(axis=1, skipna=True).dropna()

    def _comp_ic(returns_dict) -> pd.Series:
        out = {}
        for t in eval_dates:
            s = comp_by_date.get(t)
            if s is None:
                continue
            r = returns_dict[t].reindex(s.index).dropna()
            s2 = s.reindex(r.index)
            if len(s2) < MIN_XS:
                continue
            c, _ = sps.spearmanr(s2.values, r.values)
            if np.isfinite(c):
                out[t] = float(c)
        return pd.Series(out).sort_index()

    m, tt, pp, nn = summarize_ic(_comp_ic(returns_by_date))
    m_pn, tt_pn, pp_pn, _ = summarize_ic(_comp_ic(returns_pn_by_date))
    comp_port = ls_portfolio(comp_by_date, returns_by_date, prices_by_date, eval_dates)
    print(f"\n=== Combo {'+'.join(combo)} on eval half ===")
    print(f"raw IC={m:.4f} (t={tt:.2f}, p={pp:.4f}, n={nn}) | "
          f"price-neutral IC={m_pn:.4f} (t={tt_pn:.2f}, p={pp_pn:.4f})")
    if comp_port:
        print(f"L/S gross: {comp_port['gross_per_period_pct']:.2f}%/period, Sharpe {comp_port['gross_sharpe']:.2f}, total {comp_port['gross_total_pct']:.1f}%")
        print(f"L/S net  : {comp_port['net_per_period_pct']:.2f}%/period, Sharpe {comp_port['net_sharpe']:.2f}, total {comp_port['net_total_pct']:.1f}%")
        print("Deciles (Δp pp/5d):", {k: round(v, 2) for k, v in comp_port["decile_returns_pct"].items()})
    (out_dir / "expanded_combo.json").write_text(
        json.dumps({"combo": combo, "eval_ic": m, "eval_tstat": tt, "eval_pvalue": pp,
                    "eval_ic_pn": m_pn, "eval_tstat_pn": tt_pn, "eval_pvalue_pn": pp_pn,
                    "n_eval": nn, "portfolio": comp_port}, indent=2, default=float),
        encoding="utf-8",
    )

    # ── Fama-MacBeth (Phase 2) on the full panel ─────────────────────────
    # Price-level controls (entry price and its square) are included as
    # regressors so factor premia measure effects BEYOND price level.
    print("\n=== Fama-MacBeth two-step regression (Newey-West lags=6, price controls) ===")
    try:
        fm_scores = {}
        for t, X in scores_by_date.items():
            Xc = X.copy()
            p = prices_by_date[t].reindex(X.index)
            Xc["CTRL-P"] = p
            Xc["CTRL-P2"] = p ** 2
            fm_scores[t] = Xc
        fm = run_fama_macbeth(fm_scores, returns_by_date, newey_west_lags=6)
        fm_df = pd.DataFrame({
            "premium_bp_per_5d": fm.factor_premia * 10000,
            "nw_tstat": fm.t_statistics,
            "pvalue": fm.p_values,
        })
        print(fm_df.round(3).to_string())
        print(f"alpha={fm.alpha * 10000:.1f}bp (t={fm.alpha_tstat:.2f}), mean R2={fm.mean_r_squared:.3f}, "
              f"periods={len(fm.beta_time_series)}")
        fm_df.to_csv(out_dir / "expanded_fm_results.csv")
    except Exception as exc:
        print(f"FM failed: {exc}")

    # ── Per-category PRICE-NEUTRAL IC (eval half) for headline factors ───
    print("\n=== Per-category eval-half price-neutral IC (top categories) ===")
    top_cats = meta["category"].value_counts().head(6).index.tolist()
    cat_rows = []
    for cat in top_cats:
        cat_mkts = set(meta.index[meta["category"] == cat])
        for f in ["PM-EXTR", "PM-DRIFT", "PM-MOM"]:
            vals = []
            for t in eval_dates:
                X = scores_by_date[t]
                idx = [m_ for m_ in X.index if m_ in cat_mkts]
                if len(idx) < 20:
                    continue
                s = X.loc[idx, f].dropna()
                r = returns_pn_by_date[t].reindex(s.index).dropna()
                s = s.reindex(r.index)
                if len(s) < 20:
                    continue
                c, _ = sps.spearmanr(s.values, r.values)
                if np.isfinite(c):
                    vals.append(c)
            if len(vals) >= 5:
                arr = np.array(vals)
                tst = arr.mean() / (arr.std() / np.sqrt(len(arr))) if arr.std() > 0 else 0.0
                cat_rows.append({"category": cat, "factor": f, "mean_ic": arr.mean(),
                                 "tstat": tst, "n": len(arr)})
    if cat_rows:
        cat_df = pd.DataFrame(cat_rows).sort_values(["factor", "mean_ic"])
        cat_df.to_csv(out_dir / "expanded_category_ic.csv", index=False)
        print(cat_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
