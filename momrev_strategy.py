"""
MOM 反转策略原型 — Sports/Politics，真实点差成本。

三个组成部分
------------
1. 点差成本模型：每市场的小时线非零 |Δp| 中位数（小时级微小变动主要
   反映 bid-ask 跳动，其典型幅度 ≈ 有效点差）。日线 Roll 估计量被验证
   不可用 — 日级波动远大于点差，估计值全部打到上限。
   成本按名字计：多头往返成本 = s/p_entry，空头 = s/(1-p_entry)。
   （扩样研究证明统一 1.5% 成本对边界价市场严重失真。）

2. 策略：每 5 天调仓，做多 30 日动量最差的五分位（输家反弹）、
   做空动量最好的五分位。方向来自扩样研究的先验（pn IC=-0.032，
   t=-3.4，Sports/Politics 最强），非本回测内优化 — 无选择偏差。

3. 执行约定与扩样研究一致：t 决策 / t+1 收盘入场（须有成交且
   0.05<p<0.95）/ t+6 按最后成交价退出。

输出: logs/momrev_strategy_summary.json, logs/momrev_period_returns.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sps

from expanded_factor_research import (
    FWD,
    MIN_HIST,
    MIN_PX,
    MAX_PX,
    MOM_LB,
    STEP,
    load_panel,
)

QUINTILE = 5
MIN_NAMES = 40          # min universe size per rebalance
SPREAD_FLOOR, SPREAD_CAP = 0.005, 0.10
SPREAD_DEFAULT = 0.05   # for names with no hourly history


def trade_spread_estimates(market_ids, since: str = "2025-06-01") -> pd.Series:
    """Effective spread from REAL fills.

    Encoding in the trades table (validated): each fill is recorded under the
    token the TAKER bought — outcome='Yes' rows print at the YES ask,
    outcome='No' rows at the NO ask = 1 − YES bid (taker_bought is always
    True on these rows and always False on outcome-NULL rows, so the flag
    itself is useless).  Effective spread per market-day:
        median(YES-buy px) − median(1 − NO-buy px)
    then median across days per market.
    """
    import db
    ids = ",".join("'" + str(m).replace("'", "''") + "'" for m in market_ids)
    sql = f"""
    WITH t AS (
      SELECT market_id, CAST(timestamp AS DATE) AS d, outcome,
             CASE WHEN outcome = 'Yes' THEN price ELSE 1 - price END AS yes_px
      FROM read_parquet('{db.USERS_BASE}/trades/**/*.parquet', hive_partitioning=true)
      WHERE timestamp >= '{since}'
        AND outcome IN ('Yes', 'No')
        AND price > 0.01 AND price < 0.99
        AND market_id IN ({ids})
    ),
    daily AS (
      SELECT market_id, d,
             median(CASE WHEN outcome = 'Yes' THEN yes_px END) AS ask_px,
             median(CASE WHEN outcome = 'No'  THEN yes_px END) AS bid_px,
             count(CASE WHEN outcome = 'Yes' THEN 1 END)       AS nb,
             count(CASE WHEN outcome = 'No'  THEN 1 END)       AS ns
      FROM t GROUP BY market_id, d
    )
    SELECT market_id, median(ask_px - bid_px) AS s, count(*) AS n_days
    FROM daily
    WHERE nb >= 3 AND ns >= 3
    GROUP BY market_id
    HAVING count(*) >= 5
    """
    res = db.query(sql).set_index("market_id")["s"]
    return res.clip(0.003, SPREAD_CAP)


def hourly_spread_estimates(market_ids) -> pd.Series:
    """Per-market effective spread (price units) = median nonzero hourly |Δp|.

    Small hourly close-to-close moves in prediction markets are dominated by
    bid-ask bounce, so their typical magnitude approximates the effective
    spread.  Aggregated inside DuckDB — only one row per market comes back.
    """
    import db
    ids = ",".join("'" + str(m).replace("'", "''") + "'" for m in market_ids)
    sql = f"""
    WITH h AS (
      SELECT market_id,
             close - lag(close) OVER (PARTITION BY market_id ORDER BY timestamp) AS dp
      FROM read_parquet('{db.USERS_BASE}/ohlcv_1h/**/*.parquet', hive_partitioning=true)
      WHERE timestamp >= '2025-01-01'
        AND outcome IS NULL
        AND market_id IN ({ids})
    )
    SELECT market_id, median(abs(dp)) AS s
    FROM h
    WHERE dp IS NOT NULL AND abs(dp) > 1e-6
    GROUP BY market_id
    HAVING count(*) >= 50
    """
    res = db.query(sql).set_index("market_id")["s"]
    return res.clip(SPREAD_FLOOR, SPREAD_CAP)


def run_strategy(prices, volume, meta, spread, categories=None,
                 min_px=MIN_PX, max_px=MAX_PX) -> Dict:
    dates = prices.index
    prices_ff = prices.ffill()

    if categories:
        univ_cols = prices.columns.intersection(
            meta.index[meta["category"].isin(categories)]
        )
    else:
        univ_cols = prices.columns

    period_rows: List[Dict] = []
    for i in range(MIN_HIST, len(dates) - FWD - 1, STEP):
        t = dates[i]
        d1 = dates[i + 1]
        p_t = prices.iloc[i][univ_cols]
        v_t = volume.iloc[i][univ_cols]

        live = p_t[(p_t > min_px) & (p_t < max_px) & (v_t.fillna(0) > 0)].index
        if len(live) < MIN_NAMES:
            continue

        hp = prices.iloc[: i + 1][live]
        if len(hp) < MOM_LB + 1:
            continue
        mom = (hp.iloc[-1] / hp.iloc[-(MOM_LB + 1)].replace(0, np.nan) - 1).dropna()

        p_entry = prices.iloc[i + 1]
        v_entry = volume.iloc[i + 1]
        entry_ok = p_entry[
            (p_entry > min_px) & (p_entry < max_px) & (v_entry.fillna(0) > 0)
        ].index
        exit_px = prices_ff.iloc[i + 1 + FWD]
        fwd = (exit_px - p_entry).dropna()
        alive = (meta["close_d"] > d1).reindex(fwd.index).fillna(False)
        fwd = fwd[alive]

        names = mom.index.intersection(entry_ok).intersection(fwd.index)
        if len(names) < MIN_NAMES:
            continue

        # Reversal: long the biggest LOSERS (lowest momentum), short winners
        mom_n = mom[names].sort_values()
        top = max(len(mom_n) // QUINTILE, 1)
        long_idx = mom_n.iloc[:top].index
        short_idx = mom_n.iloc[-top:].index

        p_l, p_s = p_entry[long_idx], p_entry[short_idx]
        r_l, r_s = fwd[long_idx], fwd[short_idx]
        s_l = spread.reindex(long_idx).fillna(SPREAD_DEFAULT)
        s_s = spread.reindex(short_idx).fillna(SPREAD_DEFAULT)

        long_g = (r_l / p_l).mean()
        short_g = (-r_s / (1 - p_s)).mean()
        long_n = ((r_l - s_l) / p_l).mean()
        short_n = ((-r_s - s_s) / (1 - p_s)).mean()

        period_rows.append({
            "date": t,
            "n_names": len(names), "n_leg": top,
            "ls_gross": 0.5 * (long_g + short_g),
            "ls_net": 0.5 * (long_n + short_n),
            "long_net": long_n, "short_net": short_n,
            "dp_spread_pp": float((r_s.mean() - r_l.mean()) * -100),  # loser-minus-winner Δp
            "avg_p_long": float(p_l.mean()), "avg_p_short": float(p_s.mean()),
            "avg_spread_long": float(s_l.mean()), "avg_spread_short": float(s_s.mean()),
            "cost_drag": 0.5 * (long_g + short_g) - 0.5 * (long_n + short_n),
        })

    df = pd.DataFrame(period_rows).set_index("date")
    if df.empty:
        return {"error": "no periods"}

    ppy = 365.0 / STEP
    def _stats(col):
        r = df[col]
        t_stat = float(r.mean() / (r.std() / np.sqrt(len(r)))) if r.std() > 0 else 0.0
        return {
            "mean_pct": float(r.mean() * 100),
            "median_pct": float(r.median() * 100),
            "tstat": t_stat,
            "sharpe_ann": float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else 0.0,
            "total_pct": float(((1 + r).prod() - 1) * 100),
            "win_rate_pct": float((r > 0).mean() * 100),
        }

    return {
        "universe": ",".join(categories) if categories else "ALL",
        "n_periods": len(df),
        "avg_names": float(df["n_names"].mean()),
        "avg_leg_size": float(df["n_leg"].mean()),
        "avg_p_long": float(df["avg_p_long"].mean()),
        "avg_p_short": float(df["avg_p_short"].mean()),
        "avg_spread_cost_drag_pct": float(df["cost_drag"].mean() * 100),
        "dp_spread_pp_per_5d": float(df["dp_spread_pp"].mean()),
        "dp_spread_tstat": float(
            df["dp_spread_pp"].mean() / (df["dp_spread_pp"].std() / np.sqrt(len(df)))
        ),
        "gross": _stats("ls_gross"),
        "net": _stats("ls_net"),
        "long_net": _stats("long_net"),
        "short_net": _stats("short_net"),
        "_period_df": df,
    }


def main():
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)

    prices, volume, meta = load_panel()
    print(f"Panel: {prices.shape[0]} days x {prices.shape[1]} markets")

    print("Estimating spreads from real fills (trades table, DuckDB aggregate)...")
    spread_tr = trade_spread_estimates(list(prices.columns))
    print(f"Trade-based spreads for {len(spread_tr)}/{prices.shape[1]} markets: "
          f"median={spread_tr.median():.3f}, p25={spread_tr.quantile(0.25):.3f}, "
          f"p75={spread_tr.quantile(0.75):.3f} (price units)")

    print("Estimating spreads from hourly data (fallback for uncovered names)...")
    spread_h = hourly_spread_estimates(list(prices.columns))
    both = spread_tr.index.intersection(spread_h.index)
    if len(both) > 50:
        print(f"Overlap {len(both)} names: trade-based median {spread_tr[both].median():.3f} "
              f"vs hourly-proxy median {spread_h[both].median():.3f}")
    spread = spread_tr.combine_first(spread_h)

    results = {}
    for label, cats, lo, hi in [
        # Wide band: includes boundary names — $-returns are lottery-dominated
        ("Sports+Politics 0.05-0.95", ["Sports", "Politics"], 0.05, 0.95),
        # Core band: max leverage 6.7x, tame tails — the credible tradeable estimate
        ("Sports+Politics 0.15-0.85", ["Sports", "Politics"], 0.15, 0.85),
        ("ALL 0.15-0.85", None, 0.15, 0.85),
    ]:
        print(f"\n=== MOM reversal strategy — universe: {label} ===")
        res = run_strategy(prices, volume, meta, spread, cats, min_px=lo, max_px=hi)
        if "error" in res:
            print("  no valid periods")
            continue
        df = res.pop("_period_df")
        if label == "Sports+Politics 0.15-0.85":
            df.to_csv(out_dir / "momrev_period_returns.csv")
        results[label] = res
        print(f"periods={res['n_periods']}  avg_universe={res['avg_names']:.0f}  "
              f"leg={res['avg_leg_size']:.0f} names  "
              f"avg_p long/short={res['avg_p_long']:.2f}/{res['avg_p_short']:.2f}")
        print(f"Δp loser-minus-winner: {res['dp_spread_pp_per_5d']:+.2f} pp/5d "
              f"(t={res['dp_spread_tstat']:.2f})")
        print(f"cost drag: {res['avg_spread_cost_drag_pct']:.2f}%/period")
        for k in ["gross", "net", "long_net", "short_net"]:
            s = res[k]
            print(f"  {k:10s}: mean {s['mean_pct']:+7.2f}%/5d  median {s['median_pct']:+7.2f}%  "
                  f"t={s['tstat']:+6.2f}  Sharpe(ann) {s['sharpe_ann']:+6.2f}  "
                  f"win {s['win_rate_pct']:.0f}%")

    (out_dir / "momrev_strategy_summary.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
