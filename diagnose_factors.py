from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    BACKTEST_LOOKBACK_DAYS,
    BACKTEST_MAX_MARKETS,
    BACKTEST_MIN_AGE_DAYS,
    BACKTEST_N_MARKETS,
)
from data_fetcher import build_price_matrix, fetch_active_markets
from factor_engine import run_all_factors


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pvalues.fillna(1.0).astype(float)
    n = len(p)
    order = p.sort_values().index
    ranked = p.loc[order]

    q = ranked * n / pd.Series(range(1, n + 1), index=ranked.index)
    q = q[::-1].cummin()[::-1].clip(upper=1.0)

    out = pd.Series(index=p.index, dtype=float)
    out.loc[q.index] = q.values
    return out


def main() -> None:
    print("Loading market metadata...")
    markets = fetch_active_markets(limit=BACKTEST_N_MARKETS)

    print("Building/Loading prices...")
    prices = build_price_matrix(
        markets,
        days=BACKTEST_LOOKBACK_DAYS,
        min_age_days=BACKTEST_MIN_AGE_DAYS,
        max_markets=BACKTEST_MAX_MARKETS,
        bypass_cache=False,
    )

    volume_map = {
        (m.get("question") or "")[:60].strip(): float(m.get("volume24hr") or m.get("volume") or 0.0)
        for m in markets
    }

    print("Running factor engine...")
    results = run_all_factors(prices, markets, volume_map)

    rows = []
    for name, r in results.items():
        rows.append(
            {
                "factor": name,
                "mean_ic": r.mean_ic,
                "tstat": r.ic_tstat,
                "pvalue": r.ic_pvalue,
                "n_markets_used": r.n_markets_used,
                "n_ic_points": int(r.ic_series.dropna().shape[0]),
            }
        )

    df = pd.DataFrame(rows).sort_values("pvalue")
    df["fdr_q"] = bh_fdr(df["pvalue"])
    df["sig_5pct_raw"] = df["pvalue"] < 0.05
    df["sig_5pct_fdr"] = df["fdr_q"] < 0.05

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "factor_significance.csv"
    json_path = out_dir / "factor_significance.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Factor Significance (with FDR) ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
