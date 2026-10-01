from __future__ import annotations

import json
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from backtest import WalkForwardBacktest
from config import (
    BACKTEST_LOOKBACK_DAYS,
    BACKTEST_MAX_MARKETS,
    BACKTEST_MIN_AGE_DAYS,
    BACKTEST_N_MARKETS,
    POSITION_SIZE_USD,
    TRANSACTION_COST,
)
from data_fetcher import build_price_matrix, fetch_active_markets
from factor_engine import (
    IC_FORWARD_DAYS,
    MOM_LOOKBACK,
    N_DECILES,
    _build_volume_proxy,
    _spread_proxy_from_prices,
    compute_decile_portfolio,
    compute_drift,
    compute_extremity,
    compute_factor_correlation,
    compute_ic_series,
    compute_liquidity,
    compute_momentum,
    compute_ttr,
    compute_volatility,
    run_all_factors,
)
from portfolio_arb import run_portfolio_arb_walkforward


FACTOR_NAMES = [
    "PM-MOM",
    "PM-LIQ",
    "PM-TTR",
    "PM-VOL",
    "PM-EXTR",
    "PM-DRIFT",
    "PM-SPREAD",
]


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pvalues.fillna(1.0).astype(float)
    n = len(p)
    ranked = p.sort_values()
    q = ranked * n / pd.Series(range(1, n + 1), index=ranked.index)
    q = q[::-1].cummin()[::-1].clip(upper=1.0)
    out = pd.Series(index=p.index, dtype=float)
    out.loc[q.index] = q.values
    return out


def _factor_quality(fr) -> float:
    decile_spread = 0.0
    if fr.decile_returns is not None and not fr.decile_returns.empty:
        decile_spread = float(fr.decile_returns.iloc[-1] - fr.decile_returns.iloc[0])
    return (
        abs(float(fr.mean_ic)) * 100.0
        + abs(float(fr.ic_tstat)) * 10.0
        + abs(decile_spread)
    )


def _make_factor_fn(names: Tuple[str, ...], factor_orientation: Dict[str, int]):
    def _factor_fn(hist: pd.DataFrame) -> pd.Series:
        volume_proxy = _build_volume_proxy(hist)
        ref_date = hist.index[-1]

        score_map: Dict[str, pd.Series] = {}
        for name in names:
            if name == "PM-MOM":
                score_map[name] = compute_momentum(hist, lookback=MOM_LOOKBACK)
            elif name == "PM-LIQ":
                score_map[name] = compute_liquidity(hist, volume_proxy)
            elif name == "PM-TTR":
                score_map[name] = compute_ttr(_MARKETS, reference_date=ref_date).reindex(hist.columns).dropna()
            elif name == "PM-VOL":
                score_map[name] = compute_volatility(hist, window=21)
            elif name == "PM-EXTR":
                score_map[name] = compute_extremity(hist)
            elif name == "PM-DRIFT":
                score_map[name] = compute_drift(hist, window=14)
            elif name == "PM-SPREAD":
                score_map[name] = _spread_proxy_from_prices(hist)
            else:
                raise KeyError(f"Unknown factor {name}")

        df = pd.DataFrame(score_map)
        if df.empty:
            return pd.Series(dtype=float)

        z = pd.DataFrame(index=df.index)
        for col in df.columns:
            s = df[col].dropna()
            if len(s) < 5:
                continue
            std = float(s.std())
            if not np.isfinite(std) or std <= 0:
                continue
            z[col] = (df[col] - df[col].mean()) / std
            z[col] = z[col] * factor_orientation.get(col, 1)

        if z.empty:
            return pd.Series(dtype=float)

        return z.mean(axis=1, skipna=True).dropna().rename("3F-COMPOSITE")

    return _factor_fn


@dataclass
class ComboResult:
    factors: Tuple[str, ...]
    mean_ic: float
    ic_tstat: float
    ic_pvalue: float
    ls_total_return_pct: float
    ls_sharpe: float
    ls_total_return_net_pct: float
    ls_sharpe_net: float
    decile_spread_pct: float
    avg_pair_corr: float
    n_eval_points: int
    score: float


@dataclass
class StrategyResult:
    label: str
    metrics: Dict[str, float]
    score: float


_MARKETS: List[Dict] = []


def _load_prices_and_markets() -> Tuple[pd.DataFrame, List[Dict]]:
    markets = fetch_active_markets(limit=BACKTEST_N_MARKETS)
    prices = build_price_matrix(
        markets,
        days=BACKTEST_LOOKBACK_DAYS,
        min_age_days=BACKTEST_MIN_AGE_DAYS,
        max_markets=BACKTEST_MAX_MARKETS,
        bypass_cache=False,
    )
    return prices, markets


def _prepare_market_map(markets: List[Dict]) -> Dict[str, float]:
    return {
        (m.get("question") or "")[:60].strip(): float(m.get("volume24hr") or m.get("volume") or 0.0)
        for m in markets
    }


def _evaluate_combo(
    prices: pd.DataFrame,
    markets: List[Dict],
    factor_results,
    factor_corr: pd.DataFrame,
    combo: Tuple[str, ...],
) -> ComboResult:
    # Factor orientation (sign) is estimated on the FIRST HALF of the sample
    # only; the composite is then evaluated on the second half.  Deriving the
    # sign from full-sample IC and scoring on the same sample guaranteed a
    # positive in-sample bias for every combo.
    split_idx = len(prices) // 2
    split_date = prices.index[split_idx]
    orientation: Dict[str, int] = {}
    for name in combo:
        ic_full = factor_results[name].ic_series
        ic_train = ic_full[ic_full.index < split_date]
        mean_train = float(ic_train.mean()) if len(ic_train.dropna()) else 0.0
        orientation[name] = 1 if mean_train >= 0 else -1

    # Keep a 35-day warmup before the split so the first factor scores
    # (30d min history + 5d forward) land at the split date, not 35d after it.
    eval_prices = prices.iloc[max(split_idx - 35, 0):]

    factor_fn = _make_factor_fn(combo, orientation)
    ic_series = compute_ic_series(eval_prices, factor_fn)
    mean_ic = float(ic_series.mean()) if not ic_series.empty else 0.0
    std_ic = float(ic_series.std()) if len(ic_series.dropna()) > 1 else 0.0
    ic_tstat = mean_ic / (std_ic / np.sqrt(len(ic_series.dropna()))) if std_ic > 0 and len(ic_series.dropna()) > 1 else 0.0
    ic_pvalue = float(2 * (1 - stats.t.cdf(abs(ic_tstat), df=max(len(ic_series.dropna()) - 1, 1)))) if len(ic_series.dropna()) > 2 else 1.0

    decile_returns, portfolio_equity = compute_decile_portfolio(eval_prices, factor_fn)
    ls_total_return_pct = 0.0
    ls_sharpe = 0.0
    ls_total_return_net_pct = 0.0
    ls_sharpe_net = 0.0
    if not portfolio_equity.empty and len(portfolio_equity) > 2:
        ret = portfolio_equity.pct_change().dropna()
        # Rebalances are ~5 trading days apart — annualize from the actual
        # calendar spacing.  sqrt(365) assumed daily returns and inflated
        # Sharpe by ~sqrt(5) ≈ 2.24x.
        spacing = float(np.median(
            np.diff(portfolio_equity.index.values).astype("timedelta64[D]").astype(float)
        ))
        periods_per_year = 365.0 / max(spacing, 1.0)
        ls_total_return_pct = float((portfolio_equity.iloc[-1] - portfolio_equity.iloc[0]) / portfolio_equity.iloc[0] * 100)
        if len(ret) > 1 and ret.std() > 0:
            ls_sharpe = float(ret.mean() / ret.std() * np.sqrt(periods_per_year))
        # Net of trading costs: assume full quintile turnover every rebalance,
        # round-trip cost on both legs (conservative upper bound).
        ret_net = ret - 2.0 * TRANSACTION_COST
        eq_net = (1 + ret_net).cumprod()
        ls_total_return_net_pct = float((eq_net.iloc[-1] - 1) * 100)
        if len(ret_net) > 1 and ret_net.std() > 0:
            ls_sharpe_net = float(ret_net.mean() / ret_net.std() * np.sqrt(periods_per_year))

    if not decile_returns.empty:
        decile_spread_pct = float(decile_returns.loc[N_DECILES] - decile_returns.loc[1])
    else:
        decile_spread_pct = 0.0

    pair_corr_vals = []
    for a, b in itertools.combinations(combo, 2):
        if a in factor_corr.index and b in factor_corr.columns:
            val = factor_corr.loc[a, b]
            if pd.notna(val):
                pair_corr_vals.append(abs(float(val)))
    avg_pair_corr = float(np.mean(pair_corr_vals)) if pair_corr_vals else 0.0

    score = ls_sharpe_net + 0.10 * decile_spread_pct - 0.50 * avg_pair_corr
    return ComboResult(
        factors=combo,
        mean_ic=mean_ic,
        ic_tstat=float(ic_tstat),
        ic_pvalue=ic_pvalue,
        ls_total_return_pct=ls_total_return_pct,
        ls_sharpe=ls_sharpe,
        ls_total_return_net_pct=ls_total_return_net_pct,
        ls_sharpe_net=ls_sharpe_net,
        decile_spread_pct=decile_spread_pct,
        avg_pair_corr=avg_pair_corr,
        n_eval_points=int(ic_series.dropna().shape[0]),
        score=score,
    )


def _screen_combo_static(
    factor_results,
    factor_corr: pd.DataFrame,
    combo: Tuple[str, ...],
) -> float:
    factor_scores = [_factor_quality(factor_results[name]) for name in combo]
    pair_corr_vals = []
    for a, b in itertools.combinations(combo, 2):
        if a in factor_corr.index and b in factor_corr.columns:
            val = factor_corr.loc[a, b]
            if pd.notna(val):
                pair_corr_vals.append(abs(float(val)))
    avg_pair_corr = float(np.mean(pair_corr_vals)) if pair_corr_vals else 0.0
    return float(np.mean(factor_scores) - 50.0 * avg_pair_corr)


def _evaluate_pair_strategy(prices: pd.DataFrame) -> pd.DataFrame:
    grid = list(itertools.product(
        [1.5, 2.0, 2.5],
        [0.25, 0.5],
        [3.0],
        [10, 20],
    ))
    rows = []
    for entry_z, exit_z, stop_z, window in grid:
        params = {
            "entry_z": entry_z,
            "exit_z": exit_z,
            "stop_z": stop_z,
            "window": window,
        }
        result = WalkForwardBacktest(
            prices=prices,
            pair_strategy_kwargs=params,
        ).run()
        m = result.metrics or {}
        rows.append(
            {
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_z": stop_z,
                "window": window,
                "n_trades": m.get("n_trades", 0),
                "total_return_pct": m.get("total_return_pct", 0.0),
                "ann_return_pct": m.get("ann_return_pct", 0.0),
                "sharpe": m.get("sharpe_ratio", 0.0),
                "max_dd_pct": m.get("max_drawdown_pct", 0.0),
                "win_rate_pct": m.get("win_rate_pct", 0.0),
                "score": float(m.get("sharpe_ratio", 0.0)) + 0.05 * float(m.get("total_return_pct", 0.0)) - 0.02 * abs(float(m.get("max_drawdown_pct", 0.0))),
            }
        )
    df = pd.DataFrame(rows).sort_values(["score", "sharpe", "total_return_pct"], ascending=False)
    return df


def _evaluate_portfolio_strategy(prices: pd.DataFrame) -> pd.DataFrame:
    grid = list(itertools.product(
        [0.70, 0.75, 0.80],
        [1.5, 2.0],
        [0.25, 0.5],
        [4.0],
        [30],
    ))
    rows = []
    for min_corr, entry_z, exit_z, stop_z, max_hold_days in grid:
        trades, _ = run_portfolio_arb_walkforward(
            prices,
            train_days=45,
            step_days=30,
            min_corr=min_corr,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold_days=max_hold_days,
            size_usd=POSITION_SIZE_USD,
            tc=TRANSACTION_COST,
        )
        if not trades:
            rows.append(
                {
                    "min_corr": min_corr,
                    "entry_z": entry_z,
                    "exit_z": exit_z,
                    "stop_z": stop_z,
                    "max_hold_days": max_hold_days,
                    "n_trades": 0,
                    "total_pnl_usd": 0.0,
                    "avg_pnl_usd": 0.0,
                    "win_rate_pct": 0.0,
                    "score": float("-inf"),
                }
            )
            continue

        pnls = pd.Series([float(t.pnl_usd) for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        win_rate = float((pnls > 0).mean() * 100) if len(pnls) else 0.0
        rows.append(
            {
                "min_corr": min_corr,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_z": stop_z,
                "max_hold_days": max_hold_days,
                "n_trades": len(trades),
                "total_pnl_usd": float(pnls.sum()),
                "avg_pnl_usd": float(pnls.mean()),
                "win_rate_pct": win_rate,
                "score": float(pnls.sum()) + float(pnls.mean()) * 10 + win_rate * 2,
            }
        )
    df = pd.DataFrame(rows).sort_values(["score", "total_pnl_usd", "win_rate_pct"], ascending=False)
    return df


def main() -> None:
    global _MARKETS
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)

    print("Loading markets and prices...")
    prices, markets = _load_prices_and_markets()
    _MARKETS = markets
    print(f"Price matrix: {prices.shape[0]} days x {prices.shape[1]} markets")

    volume_map = _prepare_market_map(markets)
    print("Running seven-factor engine...")
    factor_results = run_all_factors(prices, markets, volume_map)

    factor_rows = []
    for name, fr in factor_results.items():
        factor_rows.append(
            {
                "factor": name,
                "mean_ic": fr.mean_ic,
                "tstat": fr.ic_tstat,
                "pvalue": fr.ic_pvalue,
                "n_markets_used": fr.n_markets_used,
                "n_ic_points": int(fr.ic_series.dropna().shape[0]),
            }
        )
    factor_df = pd.DataFrame(factor_rows).sort_values("pvalue")
    factor_df["fdr_q"] = _bh_fdr(factor_df["pvalue"])

    factor_corr = compute_factor_correlation({name: fr.scores for name, fr in factor_results.items()})

    print("Searching best 3-factor combination...")
    screen_rows = []
    for combo in itertools.combinations(FACTOR_NAMES, 3):
        screen_rows.append(
            {
                "factors": "+".join(combo),
                "static_score": _screen_combo_static(factor_results, factor_corr, combo),
            }
        )
    screen_df = pd.DataFrame(screen_rows).sort_values("static_score", ascending=False)
    shortlist = [tuple(row.split("+")) for row in screen_df.head(5)["factors"].tolist()]

    combo_rows = []
    for combo in shortlist:
        res = _evaluate_combo(prices, markets, factor_results, factor_corr, combo)
        combo_rows.append(
            {
                "factors": "+".join(combo),
                "mean_ic": res.mean_ic,
                "ic_tstat": res.ic_tstat,
                "ic_pvalue": res.ic_pvalue,
                "ls_total_return_pct": res.ls_total_return_pct,
                "ls_sharpe": res.ls_sharpe,
                "ls_total_return_net_pct": res.ls_total_return_net_pct,
                "ls_sharpe_net": res.ls_sharpe_net,
                "decile_spread_pct": res.decile_spread_pct,
                "avg_pair_corr": res.avg_pair_corr,
                "n_eval_points": res.n_eval_points,
                "score": res.score,
                "static_score": float(screen_df.loc[screen_df["factors"] == "+".join(combo), "static_score"].iloc[0]),
            }
        )
    combo_df = pd.DataFrame(combo_rows).sort_values(["score", "ls_sharpe_net", "mean_ic"], ascending=False)

    print("Optimizing pair trading parameters...")
    pair_df = _evaluate_pair_strategy(prices)

    print("Optimizing portfolio arbitrage parameters...")
    port_df = _evaluate_portfolio_strategy(prices)

    factor_df.to_csv(out_dir / "optimization_factors.csv", index=False)
    screen_df.to_csv(out_dir / "optimization_3factor_screen.csv", index=False)
    combo_df.to_csv(out_dir / "optimization_3factor_combos.csv", index=False)
    pair_df.to_csv(out_dir / "optimization_pair_trading.csv", index=False)
    port_df.to_csv(out_dir / "optimization_portfolio_arb.csv", index=False)

    print("\n=== Factor Summary ===")
    print(factor_df.to_string(index=False))

    print("\n=== Top 10 Three-Factor Combos ===")
    print(combo_df.head(10).to_string(index=False))

    print("\n=== Top 10 Pair Trading Parameter Sets ===")
    print(pair_df.head(10).to_string(index=False))

    print("\n=== Top 10 Portfolio Arb Parameter Sets ===")
    print(port_df.head(10).to_string(index=False))

    best_combo = combo_df.iloc[0]
    best_pair = pair_df.iloc[0]
    best_port = port_df.iloc[0]

    print("\n=== Best Results ===")
    print(
        f"Best 3-factor combo: {best_combo['factors']} | score={best_combo['score']:.4f} | "
        f"ls_sharpe={best_combo['ls_sharpe']:.4f} (net={best_combo['ls_sharpe_net']:.4f}) | "
        f"mean_ic={best_combo['mean_ic']:.4f} (eval on 2nd half, n={int(best_combo['n_eval_points'])})"
    )
    print(
        "Best pair params: "
        f"entry_z={best_pair['entry_z']}, exit_z={best_pair['exit_z']}, stop_z={best_pair['stop_z']}, window={best_pair['window']} | "
        f"sharpe={best_pair['sharpe']:.4f} | total_return_pct={best_pair['total_return_pct']:.2f}%"
    )
    print(
        "Best portfolio params: "
        f"min_corr={best_port['min_corr']}, entry_z={best_port['entry_z']}, exit_z={best_port['exit_z']}, stop_z={best_port['stop_z']}, max_hold_days={best_port['max_hold_days']} | "
        f"total_pnl_usd={best_port['total_pnl_usd']:.2f} | win_rate_pct={best_port['win_rate_pct']:.1f}%"
    )

    summary = {
        "best_3factor_combo": best_combo.to_dict(),
        "best_pair": best_pair.to_dict(),
        "best_portfolio": best_port.to_dict(),
    }
    (out_dir / "optimization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
