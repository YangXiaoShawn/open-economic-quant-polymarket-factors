"""
做空赢家扫描器 — MOM 反转策略的实盘信号端。

研究依据（momrev_strategy.py，2024-09→2026-03，107 期样本外）
------------------------------------------------------------
- 输家-赢家 Δp 差 +8.66pp/5天（t=7.5），核心价格带 0.15-0.85
- alpha 集中在空头腿：做空 30 日涨幅最大的市场（= 买 NO），
  扣保守点差成本后 +10.6%/期（t=4.6，年化 Sharpe≈3.8）
- 多头腿（买输家）净亏 — 本扫描器只输出空头候选

用法
----
    python short_winners_scanner.py            # 控制台表格 + CSV
输出: logs/short_winners_<date>.csv

执行建议（与回测约定一致）：今日收盘信号 → 次日入场买 NO →
持有 ~5 天平仓或滚动调仓。单名等额资金，10-20 个名字分散。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_fetcher import build_price_matrix, fetch_active_markets

MOM_LB = 30            # momentum lookback (days)
MIN_PX, MAX_PX = 0.15, 0.85   # core band from research
MIN_VOL24 = 1000.0     # min 24h volume USD (live tradability)
MAX_SPREAD = 0.05      # skip wider live spreads
MIN_DAYS_TO_END = 10   # avoid imminent resolution
TOP_N = 20


def scan() -> pd.DataFrame:
    markets = fetch_active_markets(limit=2000)
    prices = build_price_matrix(markets, days=45, min_age_days=MOM_LB + 1, max_markets=800)
    if len(prices) < MOM_LB + 1:
        raise RuntimeError(f"Price history too short: {len(prices)} days < {MOM_LB + 1}")

    by_label = {}
    for m in markets:
        label = (m.get("question") or "")[:60].strip()
        if label and label not in by_label:
            by_label[label] = m

    p_now = prices.iloc[-1]
    p_past = prices.iloc[-(MOM_LB + 1)]
    now = pd.Timestamp.now(tz="UTC")

    rows = []
    for label in prices.columns:
        m = by_label.get(label)
        if m is None:
            continue
        p, p0 = p_now.get(label), p_past.get(label)
        if pd.isna(p) or pd.isna(p0) or p0 <= 0:
            continue
        if not (MIN_PX < p < MAX_PX):
            continue

        try:
            vol24 = float(m.get("volume24hr") or 0)
        except (TypeError, ValueError):
            vol24 = 0.0
        if vol24 < MIN_VOL24:
            continue

        bid = m.get("bestBid")
        ask = m.get("bestAsk")
        try:
            bid = float(bid) if bid is not None else None
            ask = float(ask) if ask is not None else None
        except (TypeError, ValueError):
            bid = ask = None
        spread = (ask - bid) if (bid is not None and ask is not None) else None
        if spread is not None and spread > MAX_SPREAD:
            continue

        end_raw = m.get("endDate") or m.get("resolutionDate") or ""
        days_to_end = None
        if len(end_raw) >= 10:
            try:
                days_to_end = (pd.Timestamp(end_raw[:10], tz="UTC") - now).days
            except Exception:
                pass
        if days_to_end is not None and days_to_end < MIN_DAYS_TO_END:
            continue

        mom = float(p / p0 - 1)
        if mom <= 0:
            continue   # short candidates = winners only

        # Shorting YES = buying NO; NO entry cost = 1 - bestBid
        no_entry = (1.0 - bid) if bid is not None else (1.0 - float(p))
        rows.append({
            "question": label,
            "category": m.get("category") or "",
            "price": round(float(p), 3),
            "mom30_pct": round(mom * 100, 1),
            "dp30_pp": round(float(p - p0) * 100, 1),
            "no_entry": round(no_entry, 3),
            "spread": round(spread, 3) if spread is not None else None,
            "vol24h_usd": round(vol24),
            "days_to_end": days_to_end,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("mom30_pct", ascending=False).head(TOP_N).reset_index(drop=True)


def scan_payload() -> dict:
    """JSON-serialisable scan result for the web API / WebSocket push.

    Momentum comes from daily CLOB candles (data_fetcher 4h disk cache);
    price / bid-ask / volume come from the live Gamma API on every call,
    so repeated calls give fresh tradability data cheaply.
    """
    generated = datetime.now(timezone.utc).isoformat()
    try:
        df = scan()
        candidates = df.to_dict(orient="records") if not df.empty else []
        # NaN → None for strict JSON
        for row in candidates:
            for k, v in row.items():
                if isinstance(v, float) and pd.isna(v):
                    row[k] = None
        return {
            "generated_at": generated,
            "params": {
                "mom_lookback_days": MOM_LB,
                "price_band": [MIN_PX, MAX_PX],
                "min_vol24h_usd": MIN_VOL24,
                "max_spread": MAX_SPREAD,
                "min_days_to_end": MIN_DAYS_TO_END,
                "top_n": TOP_N,
            },
            "n_candidates": len(candidates),
            "candidates": candidates,
            "strategy_note": (
                "研究依据: 30日动量反转, 空头腿净收益 +20.6%/5天 (t=9.1, 真实点差成本, "
                "2024-09~2026-03 样本外)。执行: 买NO, 单名等额, 持有~5天滚动。"
            ),
        }
    except Exception as exc:
        return {"generated_at": generated, "error": str(exc), "candidates": []}


def main():
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    df = scan()
    if df.empty:
        print("No short candidates passed the filters today.")
        return
    today = datetime.now(timezone.utc).date()
    out = out_dir / f"short_winners_{today}.csv"
    df.to_csv(out, index=False)
    print(f"\n=== SHORT-WINNERS candidates {today} "
          f"(band {MIN_PX}-{MAX_PX}, vol24h>=${MIN_VOL24:.0f}, spread<={MAX_SPREAD}) ===")
    print(df.to_string(index=True))
    print(f"\nSaved: {out}")
    print("Execution: buy NO at ~no_entry next session; equal dollars per name; "
          "hold ~5 days or roll weekly; 10-20 names for diversification.")


if __name__ == "__main__":
    main()
