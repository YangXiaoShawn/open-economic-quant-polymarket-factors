"""
PM-LEAD: 小时级 lead-lag 研究（Phase 2）。

问题：同一事件内的市场是否存在价格传导延迟（一条腿先动、其他腿补跌/补涨）？

两种信号（按事件的 mutually_exclusive 标志区分）
------------------------------------------------
1. ARB-GAP（互斥事件，Σp=1 约束）：
   预测本腿 24h 变动 = −Σ(兄弟腿 24h 变动)
   score = 预测变动 − 实际变动 = −Σpeers_24h − own_24h
   若约束执行有延迟，score 应正向预测未来 24h Δp。

2. CATCHUP（非互斥事件，同向相关腿，如阈值阶梯）：
   score = mean(peers_24h) − own_24h（兄弟动了我没动 → 补涨）

基准对照：REV-24H = −own_24h（自身小时级反转）。

执行约定与日线研究一致：t 决策、t+1h 入场、持有 24h、入场须可交易
（0.05<p<0.95）；IC 用 Δp（原始）与价格中性（对 [1,p,p²] 残差）双口径。

输出: logs/leadlag_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

import db

H_START, H_END = "2025-10-01", "2026-03-29"
MIN_DAILY_VOL = 500.0
MIN_OBS_D = 30          # min daily rows in window for market selection
MAX_MARKETS = 800
MIN_EVENT_LEGS = 3      # event must have >= this many selected markets
MIN_PX, MAX_PX = 0.05, 0.95
HOLD_H = 24             # holding horizon (hours)
LOOK_H = 24             # lookback for own/peer moves (hours)
MIN_XS = 30             # min cross-section per evaluation


def select_markets() -> pd.DataFrame:
    """Markets in multi-leg events, liquid in the hourly window."""
    sql = f"""
    WITH stats AS (
      SELECT market_id, count(*) AS n_obs, avg(volume) AS avg_vol, sum(volume) AS tot_vol
      FROM '{db.USERS_BASE}/ohlcv_1d.parquet'
      WHERE outcome IS NULL AND timestamp >= '{H_START}' AND timestamp < '{H_END}'
      GROUP BY market_id
      HAVING count(*) >= {MIN_OBS_D} AND avg(volume) >= {MIN_DAILY_VOL}
    ),
    cand AS (
      SELECT s.market_id, s.tot_vol, m.event_id, m.category,
             e.mutually_exclusive
      FROM stats s
      JOIN '{db.USERS_BASE}/markets.parquet' m ON s.market_id = CAST(m.market_id AS VARCHAR)
      JOIN '{db.USERS_BASE}/events.parquet' e ON m.event_id = e.event_id
      WHERE m.close_time IS NOT NULL
    ),
    ev AS (
      SELECT event_id FROM cand GROUP BY event_id
      HAVING count(*) >= {MIN_EVENT_LEGS}
    )
    SELECT cand.* FROM cand JOIN ev USING (event_id)
    ORDER BY tot_vol DESC
    LIMIT {MAX_MARKETS}
    """
    meta = db.query(sql).set_index("market_id")
    # Re-check leg counts after the volume cap
    legs = meta.groupby("event_id").size()
    keep_ev = legs[legs >= MIN_EVENT_LEGS].index
    meta = meta[meta["event_id"].isin(keep_ev)]
    return meta


def load_hourly(market_ids) -> pd.DataFrame:
    ids = ",".join("'" + str(m).replace("'", "''") + "'" for m in market_ids)
    sql = f"""
    SELECT market_id, timestamp, close
    FROM read_parquet('{db.USERS_BASE}/ohlcv_1h/**/*.parquet', hive_partitioning=true)
    WHERE timestamp >= '{H_START}' AND timestamp < '{H_END}'
      AND outcome IS NULL
      AND market_id IN ({ids})
    """
    long = db.query(sql)
    px = long.pivot_table(index="timestamp", columns="market_id", values="close")
    px.index = pd.to_datetime(px.index, utc=True)
    px = px.sort_index()
    # Regular hourly grid; carry quotes max 3h
    grid = pd.date_range(px.index.min(), px.index.max(), freq="1h")
    px = px.reindex(grid).ffill(limit=3)
    return px


def _neutralize(r: pd.Series, p: pd.Series) -> pd.Series:
    X = np.column_stack([np.ones(len(p)), p.values, p.values ** 2])
    beta, *_ = np.linalg.lstsq(X, r.values, rcond=None)
    return pd.Series(r.values - X @ beta, index=r.index)


def summarize(ic: pd.Series) -> Tuple[float, float, float, int]:
    ic = ic.dropna()
    n = len(ic)
    if n < 3:
        return 0.0, 0.0, 1.0, n
    m, s = float(ic.mean()), float(ic.std())
    t = m / (s / np.sqrt(n)) if s > 0 else 0.0
    p = float(2 * (1 - sps.t.cdf(abs(t), df=n - 1)))
    return m, t, p, n


def main():
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)

    meta = select_markets()
    n_ev = meta["event_id"].nunique()
    print(f"Selected {len(meta)} markets in {n_ev} events "
          f"(ME events: {meta[meta['mutually_exclusive'] == True]['event_id'].nunique()})")

    px = load_hourly(meta.index.tolist())
    print(f"Hourly grid: {px.shape[0]} hours x {px.shape[1]} markets")
    meta = meta.loc[meta.index.intersection(px.columns)]

    event_legs: Dict[str, List[str]] = {
        ev: list(g.index) for ev, g in meta.groupby("event_id")
    }
    is_me = meta["mutually_exclusive"].fillna(False).astype(bool)

    # Daily evaluations at 00:00 UTC
    eval_rows = [i for i, ts in enumerate(px.index)
                 if ts.hour == 0 and i >= LOOK_H and i + 1 + HOLD_H < len(px)]

    # PEER-24H isolates the cross-market term: peers' mean 24h move alone.
    # CATCHUP = PEER-24H − own_24h mixes it with own reversal; if PEER-24H
    # has no IC of its own, "lead-lag" is just own reversal in disguise.
    ics: Dict[str, Dict] = {k: {} for k in
                            ["ARB-GAP", "ARB-GAP_pn", "CATCHUP", "CATCHUP_pn",
                             "PEER-24H", "PEER-24H_pn",
                             "REV-24H", "REV-24H_pn"]}

    for i in eval_rows:
        t = px.index[i]
        p_t = px.iloc[i]
        own24 = p_t - px.iloc[i - LOOK_H]

        live = p_t[(p_t > MIN_PX) & (p_t < MAX_PX)].index
        valid = own24.dropna().index.intersection(live)
        if len(valid) < MIN_XS:
            continue

        # Build peer aggregates per event using only valid legs
        gap_me, gap_nme, peer_only = {}, {}, {}
        for ev, legs in event_legs.items():
            vl = [m for m in legs if m in valid]
            if len(vl) < MIN_EVENT_LEGS:
                continue
            moves = own24[vl]
            tot = float(moves.sum())
            n = len(vl)
            for m_ in vl:
                peer_sum = tot - float(moves[m_])
                peer_mean = (tot - float(moves[m_])) / (n - 1)
                peer_only[m_] = peer_mean
                if bool(is_me.get(m_, False)):
                    gap_me[m_] = -peer_sum - float(moves[m_])
                else:
                    gap_nme[m_] = peer_mean - float(moves[m_])

        # Forward: enter next hour close, hold 24h
        p_entry = px.iloc[i + 1]
        entry_ok = p_entry[(p_entry > MIN_PX) & (p_entry < MAX_PX)].index
        fwd = (px.iloc[i + 1 + HOLD_H] - p_entry).dropna()
        ok = fwd.index.intersection(entry_ok)

        def _ic(score_map, key):
            s = pd.Series(score_map, dtype=float).dropna()
            idx = s.index.intersection(ok)
            if len(idx) < MIN_XS:
                return
            s2, r2, pe = s[idx], fwd[idx], p_entry[idx]
            c, _ = sps.spearmanr(s2.values, r2.values)
            if np.isfinite(c):
                ics[key][t] = float(c)
            r_pn = _neutralize(r2, pe)
            c2, _ = sps.spearmanr(s2.values, r_pn.values)
            if np.isfinite(c2):
                ics[key + "_pn"][t] = float(c2)

        _ic(gap_me, "ARB-GAP")
        _ic(gap_nme, "CATCHUP")
        _ic(peer_only, "PEER-24H")
        rev = {m_: -float(own24[m_]) for m_ in valid}
        _ic(rev, "REV-24H")

    rows = []
    for key, d in ics.items():
        m, t_, p_, n = summarize(pd.Series(d))
        rows.append({"signal": key, "mean_ic": m, "tstat": t_, "pvalue": p_, "n_days": n})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "leadlag_summary.csv", index=False)
    print("\n=== Hourly lead-lag ICs (24h horizon, skip-1h, daily evals) ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
