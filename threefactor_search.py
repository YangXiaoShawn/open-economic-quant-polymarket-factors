"""
三因子结构全搜索 + 因子长期回测序列（前端展示用）。

在修正后的扩样方法论上（skip-day / 入场可执行 / 结算收益入样 /
价格中性双口径 / 真实成交点差成本）：

1. 每个因子的长期多空回测序列（全样本 107 期，方向由前半段定）：
   - 累计 Δp 多空价差（pp，稳健口径）
   - 累计净 token 收益（按名字扣真实点差，杠杆口径，仅供参考）
2. C(7,3)=35 个三因子组合全搜索：
   方向与组合构建只用前半段，全部指标在后半段样本外评估。
   score = 价格中性 IC t-stat + 净收益 t-stat（两个稳健性维度之和）。

输出
----
logs/factor_longterm.json      — 每因子累计曲线（前端画图）
logs/threefactor_search.csv    — 35 组合排名
logs/best_threefactor.json     — 最优组合明细
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from expanded_factor_research import (
    FACTORS,
    MIN_XS,
    STEP,
    build_scores_and_returns,
    load_panel,
    summarize_ic,
)
from momrev_strategy import (
    SPREAD_DEFAULT,
    hourly_spread_estimates,
    trade_spread_estimates,
)

QUINTILE = 5


def _period_metrics(sig: pd.Series, r: pd.Series, p: pd.Series,
                    spread: pd.Series) -> Tuple[float, float]:
    """One period: (Δp loser-spread in raw units, net token L/S return).
    sig: oriented signal (high = long)."""
    idx = sig.index
    s_sorted = sig.sort_values()
    top = max(len(s_sorted) // QUINTILE, 1)
    long_idx = s_sorted.iloc[-top:].index
    short_idx = s_sorted.iloc[:top].index
    dp_spread = float(r[long_idx].mean() - r[short_idx].mean())

    sp = spread.reindex(idx).fillna(SPREAD_DEFAULT)
    long_net = float(((r[long_idx] - sp[long_idx]) / p[long_idx]).mean())
    short_net = float(((-r[short_idx] - sp[short_idx]) / (1 - p[short_idx])).mean())
    return dp_spread, 0.5 * (long_net + short_net)


def main():
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)

    prices, volume, meta = load_panel()
    scores_by_date, returns_by_date, returns_pn_by_date, prices_by_date = \
        build_scores_and_returns(prices, volume, meta)

    all_dates = sorted(scores_by_date)
    split = len(all_dates) // 2
    train_dates, eval_dates = all_dates[:split], all_dates[split:]
    split_date = str(eval_dates[0].date())

    print("Estimating spreads (trades + hourly fallback)...")
    spread = trade_spread_estimates(list(prices.columns))
    spread = spread.combine_first(hourly_spread_estimates(list(prices.columns)))

    # ── Orientation from train half (raw IC sign) ────────────────────────
    orientation: Dict[str, int] = {}
    for f in FACTORS:
        vals = []
        for t in train_dates:
            s = scores_by_date[t][f].dropna()
            r = returns_by_date[t].reindex(s.index).dropna()
            s = s.reindex(r.index)
            if len(s) < MIN_XS:
                continue
            c, _ = sps.spearmanr(s.values, r.values)
            if np.isfinite(c):
                vals.append(c)
        orientation[f] = 1 if (np.mean(vals) if vals else 0) >= 0 else -1

    # ── Per-factor long-term series (full sample) ────────────────────────
    print("Building per-factor long-term backtest series...")
    longterm: Dict[str, Dict] = {}
    for f in FACTORS:
        recs = []
        for t in all_dates:
            X = scores_by_date[t]
            sig = (X[f].dropna() * orientation[f])
            r = returns_by_date[t].reindex(sig.index).dropna()
            sig = sig.reindex(r.index)
            p = prices_by_date[t].reindex(sig.index)
            if len(sig) < MIN_XS:
                continue
            dp, net = _period_metrics(sig, r, p, spread)
            recs.append((t, dp, net))
        if not recs:
            continue
        idx = [str(t.date()) for t, _, _ in recs]
        dp_arr = np.array([d for _, d, _ in recs])
        net_arr = np.array([n for _, _, n in recs])
        longterm[f] = {
            "orient": orientation[f],
            "dates": idx,
            "cum_dp_pp": list(np.round(np.cumsum(dp_arr) * 100, 2)),
            "cum_net_pct": list(np.round((np.cumprod(1 + net_arr) - 1) * 100, 2)),
        }

    (out_dir / "factor_longterm.json").write_text(
        json.dumps({
            "split_date": split_date,
            "n_periods": len(all_dates),
            "rebalance_days": STEP,
            "factors": longterm,
        }, default=float), encoding="utf-8")
    print(f"factor_longterm.json: {len(longterm)} factors, split at {split_date}")

    # ── 35-combo exhaustive search (train build, eval judge) ─────────────
    print("Searching all 35 three-factor combos...")

    def combo_signal(combo, t):
        X = scores_by_date[t]
        z = pd.DataFrame(index=X.index)
        for f in combo:
            s = X[f].dropna()
            if len(s) < 5 or s.std() <= 0:
                continue
            z[f] = (X[f] - s.mean()) / s.std() * orientation[f]
        if z.empty:
            return None
        return z.mean(axis=1, skipna=True).dropna()

    rows = []
    best_cache = {}
    for combo in itertools.combinations(FACTORS, 3):
        ic_raw, ic_pn = {}, {}
        dp_list, net_list, dates_used = [], [], []
        for t in eval_dates:
            sig = combo_signal(combo, t)
            if sig is None:
                continue
            r = returns_by_date[t].reindex(sig.index).dropna()
            sig2 = sig.reindex(r.index).dropna()
            r = r.reindex(sig2.index)
            if len(sig2) < MIN_XS:
                continue
            c, _ = sps.spearmanr(sig2.values, r.values)
            if np.isfinite(c):
                ic_raw[t] = float(c)
            rp = returns_pn_by_date[t].reindex(sig2.index)
            c2, _ = sps.spearmanr(sig2.values, rp.values)
            if np.isfinite(c2):
                ic_pn[t] = float(c2)
            p = prices_by_date[t].reindex(sig2.index)
            dp, net = _period_metrics(sig2, r, p, spread)
            dp_list.append(dp)
            net_list.append(net)
            dates_used.append(t)

        m_raw, t_raw, p_raw, n_raw = summarize_ic(pd.Series(ic_raw))
        m_pn, t_pn, p_pn, _ = summarize_ic(pd.Series(ic_pn))
        dp_arr, net_arr = np.array(dp_list), np.array(net_list)
        dp_t = float(dp_arr.mean() / (dp_arr.std() / np.sqrt(len(dp_arr)))) if len(dp_arr) > 2 and dp_arr.std() > 0 else 0.0
        net_t = float(net_arr.mean() / (net_arr.std() / np.sqrt(len(net_arr)))) if len(net_arr) > 2 and net_arr.std() > 0 else 0.0
        ppy = 365.0 / STEP
        net_sharpe = float(net_arr.mean() / net_arr.std() * np.sqrt(ppy)) if len(net_arr) > 2 and net_arr.std() > 0 else 0.0
        score = t_pn + net_t

        rows.append({
            "factors": "+".join(combo),
            "eval_ic": round(m_raw, 4), "eval_tstat": round(t_raw, 2),
            "eval_ic_pn": round(m_pn, 4), "eval_tstat_pn": round(t_pn, 2),
            "dp_spread_pp": round(float(dp_arr.mean()) * 100, 2) if len(dp_arr) else 0.0,
            "dp_tstat": round(dp_t, 2),
            "net_per_period_pct": round(float(net_arr.mean()) * 100, 2) if len(net_arr) else 0.0,
            "net_tstat": round(net_t, 2),
            "net_sharpe_ann": round(net_sharpe, 2),
            "n_eval": n_raw,
            "score": round(score, 2),
        })
        best_cache["+".join(combo)] = (dates_used, dp_arr, net_arr)

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "threefactor_search.csv", index=False)
    print("\n=== Top 10 combos (eval half, orientation from train half) ===")
    print(df.head(10).to_string(index=False))

    # ── Best combo detail ────────────────────────────────────────────────
    best = df.iloc[0]
    dates_used, dp_arr, net_arr = best_cache[best["factors"]]
    (out_dir / "best_threefactor.json").write_text(
        json.dumps({
            "factors": best["factors"],
            "orientation": {f: orientation[f] for f in best["factors"].split("+")},
            "stats": {k: (None if pd.isna(v) else v) for k, v in best.items()},
            "split_date": split_date,
            "equity": {
                "dates": [str(t.date()) for t in dates_used],
                "cum_dp_pp": list(np.round(np.cumsum(dp_arr) * 100, 2)),
                "cum_net_pct": list(np.round((np.cumprod(1 + net_arr) - 1) * 100, 2)),
            },
        }, default=float), encoding="utf-8")
    print(f"\nBest combo: {best['factors']} | score={best['score']} | "
          f"pn IC={best['eval_ic_pn']} (t={best['eval_tstat_pn']}) | "
          f"net {best['net_per_period_pct']}%/period (t={best['net_tstat']})")


if __name__ == "__main__":
    main()
