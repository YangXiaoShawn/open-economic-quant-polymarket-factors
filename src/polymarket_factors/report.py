"""
Performance report and chart generation for the Polymarket Pair Trading backtest.

Outputs
-------
• Console summary table
• equity_curve.png        – equity curve + drawdown
• pair_signals.png        – z-score signals for each traded pair (top 5)
• trade_dist.png          – P&L distribution histogram
• combined_equity.png     – three-strategy portfolio equity comparison
• combined_metrics.png    – strategy metrics breakdown
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

from backtest import BacktestResult
from strategy import Trade, Side

if TYPE_CHECKING:
    from combined_backtest import CombinedResult

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent
CHART_STYLE = "seaborn-v0_8-darkgrid"


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(result: BacktestResult) -> None:
    m = result.metrics
    sep = "=" * 62

    print(f"\n{sep}")
    print("  POLYMARKET PAIR TRADING — BACKTEST RESULTS")
    print(sep)
    print(f"  Initial Capital     : ${m.get('initial_capital', 0):>12,.2f}")
    print(f"  Final Equity        : ${m.get('final_equity', 0):>12,.2f}")
    print(f"  Total Return        : {m.get('total_return_pct', 0):>+11.2f}%")
    print(f"  Annualised Return   : {m.get('ann_return_pct', 0):>+11.2f}%")
    print(f"  Annualised Volatility: {m.get('ann_volatility_pct', 0):>10.2f}%")
    print(sep)
    print(f"  Sharpe Ratio        : {m.get('sharpe_ratio', 0):>12.4f}")
    print(f"  Sortino Ratio       : {m.get('sortino_ratio', 0):>12.4f}")
    print(f"  Calmar Ratio        : {m.get('calmar_ratio', 0):>12.4f}")
    print(f"  Max Drawdown        : {m.get('max_drawdown_pct', 0):>+11.2f}%")
    print(sep)
    print(f"  Total Trades        : {m.get('n_trades', 0):>12d}")
    print(f"  Win Rate            : {m.get('win_rate_pct', 0):>11.1f}%")
    print(f"  Avg Win             : ${m.get('avg_win_usd', 0):>12.2f}")
    print(f"  Avg Loss            : ${m.get('avg_loss_usd', 0):>12.2f}")
    print(f"  Profit Factor       : {m.get('profit_factor', 0):>12.3f}")
    print(f"  Avg Trade Duration  : {m.get('avg_trade_duration_days', 0):>10.1f} days")
    print(sep)

    exit_reasons = m.get("exit_reasons", {})
    if exit_reasons:
        print("  Exit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<20}: {count}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_equity_curve(result: BacktestResult, save_path: Optional[Path] = None) -> None:
    equity = result.equity_curve
    daily_pnl = result.daily_pnl

    try:
        plt.style.use(CHART_STYLE)
    except Exception:
        pass

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)

    # --- Panel 1: Equity curve ---
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(equity.index, equity.values, color="#2196F3", linewidth=1.8, label="Portfolio Equity")
    ax1.axhline(result.metrics.get("initial_capital", 10000), color="grey",
                linestyle="--", linewidth=1, alpha=0.7, label="Initial Capital")
    ax1.fill_between(equity.index, equity.values,
                     result.metrics.get("initial_capital", 10000),
                     where=equity.values >= result.metrics.get("initial_capital", 10000),
                     alpha=0.15, color="#4CAF50", label="Profit")
    ax1.fill_between(equity.index, equity.values,
                     result.metrics.get("initial_capital", 10000),
                     where=equity.values < result.metrics.get("initial_capital", 10000),
                     alpha=0.15, color="#F44336", label="Loss")
    ax1.set_title("Polymarket Pair Trading — Equity Curve", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (USD)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Annotate key metrics
    m = result.metrics
    info_text = (
        f"Total Return: {m.get('total_return_pct', 0):+.2f}%  "
        f"Sharpe: {m.get('sharpe_ratio', 0):.3f}  "
        f"MaxDD: {m.get('max_drawdown_pct', 0):.2f}%  "
        f"Trades: {m.get('n_trades', 0)}"
    )
    ax1.set_xlabel(info_text, fontsize=9, color="grey")

    # --- Panel 2: Drawdown ---
    ax2 = fig.add_subplot(gs[2, 0])
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.6)
    ax2.set_title("Drawdown (%)", fontsize=10)
    ax2.set_ylabel("Drawdown %")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    path = save_path or OUTPUT_DIR / "equity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity curve saved ->%s", path)
    print(f"  [Chart] Equity curve   ->{path}")


def plot_pair_signals(
    result: BacktestResult,
    max_pairs: int = 5,
    save_path: Optional[Path] = None,
) -> None:
    signals_by_pair = result.signals_by_pair
    if not signals_by_pair:
        return

    pairs = list(signals_by_pair.items())[:max_pairs]
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    try:
        plt.style.use(CHART_STYLE)
    except Exception:
        pass

    for ax, (pair_key, sig_df) in zip(axes, pairs):
        z = sig_df["zscore"].dropna()
        ax.plot(z.index, z.values, color="#607D8B", linewidth=1.2, label="Z-score")
        ax.axhline(2.0,  color="#F44336", linestyle="--", linewidth=1, alpha=0.8, label="+2σ entry")
        ax.axhline(-2.0, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.8, label="-2σ entry")
        ax.axhline(0.5,  color="#FF9800", linestyle=":",  linewidth=1, alpha=0.7, label="±0.5σ exit")
        ax.axhline(-0.5, color="#FF9800", linestyle=":",  linewidth=1, alpha=0.7)
        ax.axhline(0,    color="black",   linestyle="-",  linewidth=0.5, alpha=0.4)

        # Shade entry/exit regions
        long_mask = z < -2.0
        short_mask = z > 2.0
        ax.fill_between(z.index, z.values, -2.0, where=long_mask,
                        alpha=0.2, color="#4CAF50", label="Long spread zone")
        ax.fill_between(z.index, z.values, 2.0, where=short_mask,
                        alpha=0.2, color="#F44336", label="Short spread zone")

        label = pair_key.replace("__", " / ")
        ax.set_title(f"Pair: {label}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Z-score")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle("Pair Trading Signals — Z-score", fontsize=13, fontweight="bold", y=1.01)
    path = save_path or OUTPUT_DIR / "pair_signals.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pair signals saved ->%s", path)
    print(f"  [Chart] Pair signals   ->{path}")


def plot_trade_distribution(
    result: BacktestResult,
    save_path: Optional[Path] = None,
) -> None:
    closed = [t for t in result.trades if t.close_date is not None]
    if not closed:
        return

    pnls = [t.pnl_usd for t in closed]
    durations = [t.duration_days or 0 for t in closed]

    try:
        plt.style.use(CHART_STYLE)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Trade Analysis", fontsize=13, fontweight="bold")

    # P&L histogram
    ax = axes[0]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    ax.hist(wins, bins=20, color="#4CAF50", alpha=0.7, label=f"Wins ({len(wins)})")
    ax.hist(losses, bins=20, color="#F44336", alpha=0.7, label=f"Losses ({len(losses)})")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("P&L Distribution")
    ax.set_xlabel("P&L (USD)")
    ax.set_ylabel("Count")
    ax.legend()

    # Duration histogram
    ax = axes[1]
    ax.hist(durations, bins=20, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.set_title("Trade Duration Distribution")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Count")

    # Exit reasons pie
    ax = axes[2]
    reasons = pd.Series([t.closed_by for t in closed]).value_counts()
    colors = {"signal": "#4CAF50", "stop_loss": "#F44336",
               "eod": "#FF9800", "signal_flip": "#9C27B0"}
    pie_colors = [colors.get(r, "#607D8B") for r in reasons.index]
    ax.pie(reasons.values, labels=reasons.index, autopct="%1.1f%%",
           colors=pie_colors, startangle=90)
    ax.set_title("Exit Reasons")

    path = save_path or OUTPUT_DIR / "trade_dist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trade distribution saved ->%s", path)
    print(f"  [Chart] Trade dist.    ->{path}")


def generate_full_report(result: BacktestResult) -> None:
    print_report(result)
    plot_equity_curve(result)
    plot_pair_signals(result)
    plot_trade_distribution(result)


# ---------------------------------------------------------------------------
# Combined three-strategy report
# ---------------------------------------------------------------------------

def print_combined_report(result: "CombinedResult") -> None:
    sep = "=" * 70

    def _row(label, m, key, fmt=",.2f", prefix="$"):
        val = m.get(key, 0)
        if fmt == "pct":
            return f"  {label:<28}: {val:>+11.2f}%"
        elif fmt == "ratio":
            return f"  {label:<28}: {val:>12.4f}"
        elif fmt == "int":
            return f"  {label:<28}: {val:>12d}"
        else:
            return f"  {label:<28}: {prefix}{val:>12{fmt}}"

    print(f"\n{sep}")
    print("  POLYMARKET TWO-STRATEGY PORTFOLIO -- BACKTEST RESULTS")
    print(sep)

    for label, m in [
        ("COMBINED PORTFOLIO",    result.metrics_combined),
        ("  Pair Trading (40%)",  result.metrics_pair),
        ("  Portfolio Arb (60%)", result.metrics_port_arb),
    ]:
        print(f"\n  {label}")
        print(f"  {'-'*66}")
        print(_row("Initial Capital",    m, "initial_capital"))
        print(_row("Final Equity",       m, "final_equity"))
        print(_row("Total Return",       m, "total_return_pct", "pct"))
        print(_row("Ann. Return",        m, "ann_return_pct",   "pct"))
        print(_row("Ann. Volatility",    m, "ann_volatility_pct", "pct"))
        print(_row("Sharpe Ratio",       m, "sharpe_ratio",     "ratio"))
        print(_row("Max Drawdown",       m, "max_drawdown_pct", "pct"))
        print(_row("Total Trades",       m, "n_trades",         "int"))
        print(_row("Win Rate",           m, "win_rate_pct",     "pct"))
        print(_row("Profit Factor",      m, "profit_factor",    "ratio"))

    print(f"\n{sep}\n")


def plot_combined_equity(
    result: "CombinedResult",
    save_path: Optional[Path] = None,
) -> None:
    try:
        plt.style.use(CHART_STYLE)
    except Exception:
        pass

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    initial = result.metrics_combined.get("initial_capital", 10000)

    # --- Panel 1 (top, spanning): Combined equity ---
    ax1 = fig.add_subplot(gs[0, :])
    eq = result.equity_combined
    ax1.plot(eq.index, eq.values, color="#2196F3", linewidth=2.2, label="Combined Portfolio")
    ax1.plot(result.equity_pair.index,    result.equity_pair.values,
             color="#4CAF50", linewidth=1.4, alpha=0.8, linestyle="--", label="Pair Trading (40%)")
    ax1.plot(result.equity_port_arb.index, result.equity_port_arb.values,
             color="#9C27B0", linewidth=1.4, alpha=0.8, linestyle="--", label="Portfolio Arb (60%)")
    ax1.axhline(initial, color="grey", linestyle=":", linewidth=1, alpha=0.6)
    ax1.fill_between(eq.index, eq.values, initial,
                     where=eq.values >= initial, alpha=0.1, color="#4CAF50")
    ax1.fill_between(eq.index, eq.values, initial,
                     where=eq.values < initial,  alpha=0.1, color="#F44336")
    m = result.metrics_combined
    ax1.set_title(
        f"Two-Strategy Portfolio  |  Return: {m.get('total_return_pct', 0):+.2f}%  "
        f"Sharpe: {m.get('sharpe_ratio', 0):.3f}  MaxDD: {m.get('max_drawdown_pct', 0):.2f}%",
        fontsize=12, fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Value (USD)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # --- Panel 2: Combined drawdown ---
    ax2 = fig.add_subplot(gs[1, :])
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.55, label="Combined DD")
    for eq_s, color, lbl in [
        (result.equity_pair,    "#4CAF50", "Pair Trading"),
        (result.equity_port_arb,"#9C27B0", "Portfolio Arb"),
    ]:
        rm  = eq_s.cummax()
        dds = (eq_s - rm) / rm * 100
        ax2.plot(dds.index, dds.values, color=color, linewidth=1.0, alpha=0.7, label=lbl)
    ax2.set_title("Drawdown Comparison (%)", fontsize=10)
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", fontsize=8, ncol=3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # --- Panel 3 (bottom-left): Strategy return contribution ---
    ax3 = fig.add_subplot(gs[2, 0])
    strategies = ["Pair Trading", "Portfolio Arb", "Combined"]
    returns = [
        result.metrics_pair.get("total_return_pct", 0),
        result.metrics_port_arb.get("total_return_pct", 0),
        result.metrics_combined.get("total_return_pct", 0),
    ]
    bar_colors = ["#4CAF50" if r >= 0 else "#F44336" for r in returns]
    bars = ax3.bar(strategies, returns, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, returns):
        ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.5
        ax3.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{val:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("Total Return by Strategy", fontsize=10)
    ax3.set_ylabel("Return %")

    # --- Panel 4 (bottom-right): Sharpe & win-rate ---
    ax4 = fig.add_subplot(gs[2, 1])
    sharpes   = [result.metrics_pair.get("sharpe_ratio", 0),
                 result.metrics_port_arb.get("sharpe_ratio", 0),
                 result.metrics_combined.get("sharpe_ratio", 0)]
    win_rates = [result.metrics_pair.get("win_rate_pct", 0),
                 result.metrics_port_arb.get("win_rate_pct", 0),
                 result.metrics_combined.get("win_rate_pct", 0)]
    x = np.arange(len(strategies))
    w = 0.35
    ax4.bar(x - w/2, sharpes,   w, label="Sharpe Ratio",  color="#2196F3", alpha=0.85)
    ax4_r = ax4.twinx()
    ax4_r.bar(x + w/2, win_rates, w, label="Win Rate (%)", color="#FF9800", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=15, ha="right", fontsize=8)
    ax4.set_ylabel("Sharpe Ratio", color="#2196F3")
    ax4_r.set_ylabel("Win Rate (%)", color="#FF9800")
    ax4.set_title("Sharpe Ratio & Win Rate", fontsize=10)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    path = save_path or OUTPUT_DIR / "combined_equity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined equity chart saved ->%s", path)
    print(f"  [Chart] Combined equity  ->{path}")


def plot_combined_trades(
    result: "CombinedResult",
    save_path: Optional[Path] = None,
) -> None:
    try:
        plt.style.use(CHART_STYLE)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Three-Strategy Trade Analysis", fontsize=13, fontweight="bold")

    strategy_data = [
        ("Pair Trading",  result.pair_trades,     "#4CAF50"),
        ("Portfolio Arb", result.port_arb_trades, "#9C27B0"),
    ]

    # --- P&L histogram per strategy ---
    ax = axes[0]
    for label, trades, color in strategy_data:
        pnls = [t.pnl_usd for t in trades if getattr(t, "close_date", None) is not None]
        if pnls:
            ax.hist(pnls, bins=25, alpha=0.6, color=color, label=f"{label} ({len(pnls)})")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("P&L Distribution by Strategy")
    ax.set_xlabel("P&L (USD)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # --- Trade count & win rate per strategy ---
    ax = axes[1]
    labels     = [d[0] for d in strategy_data]
    counts     = [len(d[1]) for d in strategy_data]
    win_counts = [sum(1 for t in d[1] if getattr(t, "pnl_usd", 0) > 0) for d in strategy_data]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, counts,     w, color=[d[2] for d in strategy_data], alpha=0.7, label="Total")
    ax.bar(x + w/2, win_counts, w, color=["#81C784", "#CE93D8"],         alpha=0.7, label="Winners")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Trade Count: Total vs Winners")
    ax.set_ylabel("Number of Trades")
    ax.legend(fontsize=9)

    # --- Cumulative P&L over time ---
    ax = axes[2]
    for dpnl, color, label in [
        (result.daily_pnl_pair,     "#4CAF50", "Pair Trading"),
        (result.daily_pnl_port_arb, "#9C27B0", "Portfolio Arb"),
    ]:
        ax.plot(dpnl.cumsum().index, dpnl.cumsum().values,
                color=color, linewidth=1.4, label=label)
    cum_total = result.daily_pnl_combined.cumsum()
    ax.plot(cum_total.index, cum_total.values,
            color="#2196F3", linewidth=2.0, linestyle="--", label="Combined")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Cumulative P&L Over Time")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    path = save_path or OUTPUT_DIR / "combined_trades.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined trades chart saved ->%s", path)
    print(f"  [Chart] Combined trades  ->{path}")


def generate_combined_report(result: "CombinedResult") -> None:
    print_combined_report(result)
    plot_combined_equity(result)
    plot_combined_trades(result)
    if result.signals_by_pair:
        from backtest import BacktestResult as BR
        dummy = BR(
            trades=result.pair_trades,
            equity_curve=result.equity_pair,
            daily_pnl=result.daily_pnl_pair,
            metrics=result.metrics_pair,
            signals_by_pair=result.signals_by_pair,
        )
        plot_pair_signals(dummy, max_pairs=3,
                         save_path=OUTPUT_DIR / "combined_pair_signals.png")
