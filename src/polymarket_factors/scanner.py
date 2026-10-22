"""
Polymarket Arb Scanner — Background signal engine.

Two-tier refresh:
  Fast tier  (ZSCORE_INTERVAL  = 60s)  — recompute z-scores on cached pairs
                                          no HTTP, no cointegration → ~0.1s
  Full tier  (FULL_INTERVAL    = 300s) — fetch new data + full cointegration scan
                                          uses disk cache for prices → ~5-6s

Thread-safe: uses RLock for all state mutations.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import logging
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ENTRY_ZSCORE, EXIT_ZSCORE
from data_fetcher import fetch_active_markets, build_price_matrix, clear_cache
from pair_selector import (
    select_pairs, compute_rolling_zscore, compute_zscores_fast,
    PairInfo,
)
from portfolio_arb import (
    build_baskets_from_prices,
    BASKET_ENTRY_ZSCORE, BASKET_EXIT_ZSCORE, ROLLING_WINDOW,
)

logger = logging.getLogger(__name__)

FULL_INTERVAL    = 300   # full data refresh every 5 minutes
ZSCORE_INTERVAL  = 60    # fast z-score refresh every 60 seconds
COINT_RESCAN     = 6     # re-run cointegration every 6th full-refresh (~30 min)
N_MARKETS        = 2000  # markets to request from Gamma API
LOOKBACK_DAYS    = 90    # days of price history
MIN_AGE_DAYS     = 14    # include markets at least this old
MAX_MARKETS      = 1000  # max markets to fetch prices for


# ─────────────────────────────────────────────────────────────────────
# Signal classifier
# ─────────────────────────────────────────────────────────────────────

def _classify(z: float, entry: float, exit_: float) -> str:
    az = abs(z)
    if az >= entry:
        return "ENTER_SHORT" if z > 0 else "ENTER_LONG"
    if az >= entry * 0.75:
        return "WATCH"
    if az <= exit_:
        return "NEUTRAL"
    return "WATCH"


# ─────────────────────────────────────────────────────────────────────
# Scanner
# ─────────────────────────────────────────────────────────────────────

class MarketScanner:
    """
    Thread-safe two-tier background scanner.

    Usage
    -----
    scanner = MarketScanner()
    scanner.add_callback(fn)         # fn(state: dict) called on every update
    scanner.start()                  # starts daemon thread
    scanner.get_state()              # returns current state snapshot
    scanner.trigger_refresh(force)   # immediate full refresh (force=True clears cache)
    scanner.stop()                   # signals the thread to stop
    """

    def __init__(self) -> None:
        self._lock        = threading.RLock()
        self._state       = _empty_state()
        self._stop        = threading.Event()
        self._callbacks: List[Callable] = []
        # Cache for fast z-score refresh
        self._prices:     Optional[pd.DataFrame] = None
        self._pairs:      List[PairInfo]          = []
        self._pair_cols:  frozenset               = frozenset()  # last markets used for coint
        self._coint_tick  = 0  # counter for cointegration rescan
        self._cycle       = 0  # increments each ZSCORE_INTERVAL tick
        # Prevents concurrent full refreshes (from trigger_refresh vs _loop race)
        self._refresh_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        t = threading.Thread(target=self._loop, daemon=True, name="arb-scanner")
        t.start()
        logger.info("MarketScanner started (full=%ds, zscore=%ds)", FULL_INTERVAL, ZSCORE_INTERVAL)

    def stop(self) -> None:
        self._stop.set()

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def add_callback(self, fn: Callable) -> None:
        self._callbacks.append(fn)

    def trigger_refresh(self, force: bool = False) -> None:
        if force:
            clear_cache()
        def _guarded():
            if not self._refresh_lock.acquire(blocking=False):
                logger.info("trigger_refresh skipped — refresh already in progress")
                return
            try:
                self._full_refresh()
            finally:
                self._refresh_lock.release()
        threading.Thread(target=_guarded, daemon=True, name="arb-refresh").start()

    # ── Internal loop ─────────────────────────────────────────────────

    def _loop(self) -> None:
        full_every = FULL_INTERVAL // ZSCORE_INTERVAL  # how many ticks per full refresh

        while not self._stop.is_set():
            if self._cycle % full_every == 0:
                with self._refresh_lock:
                    self._full_refresh()
            else:
                self._zscore_refresh()
            self._cycle += 1
            self._stop.wait(ZSCORE_INTERVAL)

    def _push(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)
        state = self.get_state()
        for fn in self._callbacks:
            try:
                fn(state)
            except Exception as e:
                logger.debug("Callback error: %s", e)

    # ── Fast tier: z-score only ───────────────────────────────────────

    def _zscore_refresh(self) -> None:
        with self._lock:
            prices = self._prices
            pairs  = list(self._pairs)

        if prices is None or not pairs:
            return  # nothing cached yet

        try:
            pair_results = compute_zscores_fast(prices, pairs)
            pair_sigs    = self._build_pair_signals(pair_results)

            port_sigs, n_baskets = self._portfolio_signals(prices)

            now = datetime.now()
            with self._lock:
                self._state["pair_signals"]       = pair_sigs
                self._state["portfolio_signals"]  = port_sigs
                self._state["last_update"]        = now.isoformat()
                self._state["next_refresh"]       = (now + timedelta(seconds=ZSCORE_INTERVAL)).isoformat()
                s = self._state.get("stats", {})
                s.update({
                    "n_pairs":      len(pair_sigs),
                    "n_pairs_entry": sum(1 for x in pair_sigs if "ENTER" in x["signal"]),
                    "n_port":       len(port_sigs),
                    "n_port_entry": sum(1 for x in port_sigs if "ENTER" in x["signal"]),
                    "n_baskets":    n_baskets,
                    "refresh_type": "zscore",
                })
                self._state["stats"] = s

            self._push()   # broadcast with no extra kwargs (state already updated)
            logger.debug("Fast refresh: %d pair sigs, %d port sigs", len(pair_sigs), len(port_sigs))

        except Exception as exc:
            logger.warning("Fast refresh error: %s", exc)

    def _build_pair_signals(self, pair_results: List[Tuple]) -> List[dict]:
        result = []
        for p, z in pair_results:
            signal = _classify(z, ENTRY_ZSCORE, EXIT_ZSCORE)
            result.append({
                "market_a":        p.market_a[:80],
                "market_b":        p.market_b[:80],
                "zscore":          round(z, 3),
                "hedge_ratio":     round(p.hedge_ratio, 3),
                "correlation":     round(p.correlation, 3),
                "coint_pvalue":    round(p.coint_pvalue, 4),
                "half_life_days":  round(p.half_life_days, 1),
                "spread_std":      round(p.spread_std, 4),
                "signal":          signal,
                "signal_strength": min(1.0, abs(z) / 4.0),
            })
        return sorted(result, key=lambda x: abs(x["zscore"]), reverse=True)

    # ── Full tier: data fetch + cointegration ─────────────────────────

    def _full_refresh(self) -> None:
        try:
            self._push(status="fetching", status_msg="Fetching market list…")
            markets = fetch_active_markets(limit=N_MARKETS)

            self._push(
                status="fetching",
                status_msg=f"Building price matrix ({len(markets)} markets, {LOOKBACK_DAYS}d)…",
            )
            prices = build_price_matrix(
                markets,
                days=LOOKBACK_DAYS,
                min_age_days=MIN_AGE_DAYS,
                max_markets=MAX_MARKETS,
            )

            if prices.empty or prices.shape[1] < 5:
                self._push(status="error", status_msg="Insufficient market data — retrying later")
                return

            # Decide whether to re-run cointegration or reuse cached pairs
            new_cols   = frozenset(prices.columns)
            with self._lock:
                old_cols   = self._pair_cols
                old_pairs  = self._pairs
                coint_tick = self._coint_tick

            cols_changed = len(new_cols.symmetric_difference(old_cols)) > len(new_cols) * 0.05
            need_coint   = (not old_pairs) or cols_changed or (coint_tick % COINT_RESCAN == 0)

            if need_coint:
                reason = "first run" if not old_pairs else ("cols changed" if cols_changed else f"rescan tick {coint_tick}")
                self._push(
                    status="computing",
                    status_msg=f"Cointegration screening ({prices.shape[1]} markets) [{reason}]…",
                )
                pairs = select_pairs(prices, top_n=50)
                with self._lock:
                    self._pairs     = pairs
                    self._pair_cols = new_cols
            else:
                pairs = old_pairs
                logger.info("Skipping cointegration rescan (using %d cached pairs)", len(pairs))

            with self._lock:
                self._coint_tick += 1

            pair_results = compute_zscores_fast(prices, pairs)
            pair_sigs    = self._build_pair_signals(pair_results)

            self._push(status="computing", status_msg="Computing basket deviations…")
            port_sigs, n_baskets = self._portfolio_signals(prices)

            # Update price cache for fast tier
            with self._lock:
                self._prices = prices

            now = datetime.now()
            self._push(
                status     = "live",
                status_msg = f"Live  •  {prices.shape[1]} markets  •  {prices.shape[0]} days",
                last_update   = now.isoformat(),
                next_refresh  = (now + timedelta(seconds=ZSCORE_INTERVAL)).isoformat(),
                pair_signals  = pair_sigs,
                portfolio_signals = port_sigs,
                stats = {
                    "n_markets":        prices.shape[1],
                    "n_pairs":          len(pair_sigs),
                    "n_pairs_entry":    sum(1 for s in pair_sigs if "ENTER" in s["signal"]),
                    "n_baskets":        n_baskets,
                    "n_port":           len(port_sigs),
                    "n_port_entry":     sum(1 for s in port_sigs if "ENTER" in s["signal"]),
                    "date_range":       f"{prices.index[0].date()} – {prices.index[-1].date()}",
                    "full_interval":    FULL_INTERVAL,
                    "zscore_interval":  ZSCORE_INTERVAL,
                    "coint_rescan_min": COINT_RESCAN * FULL_INTERVAL // 60,
                    "refresh_type":     "full" if need_coint else "full_cached",
                },
            )
            logger.info(
                "Full refresh: %d markets, %d pairs (%d entry), %d port (%d entry) coint=%s",
                prices.shape[1],
                len(pair_sigs), sum(1 for s in pair_sigs if "ENTER" in s["signal"]),
                len(port_sigs), sum(1 for s in port_sigs if "ENTER" in s["signal"]),
                need_coint,
            )

        except Exception as exc:
            logger.exception("Full refresh failed")
            self._push(status="error", status_msg=f"Error: {str(exc)[:120]}")

    # ── Portfolio arb signals ─────────────────────────────────────────

    def _portfolio_signals(self, prices: pd.DataFrame) -> Tuple[List[dict], int]:
        baskets = build_baskets_from_prices(prices, min_corr=0.6)
        result  = []

        for bname, bdf in baskets.items():
            df   = bdf.ffill().dropna(how="all")
            if len(df.columns) < 4 or len(df) < 25:
                continue
            bmed = df.median(axis=1)
            devs = df.subtract(bmed, axis=0)

            for mkt in df.columns:
                s = devs[mkt].dropna()
                if len(s) < 20:
                    continue
                mu  = s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2).mean()
                std = s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2).std()
                z   = float(((s - mu) / std.replace(0, np.nan)).iloc[-1])
                if pd.isna(z):
                    continue
                signal = _classify(z, BASKET_ENTRY_ZSCORE, BASKET_EXIT_ZSCORE)
                if signal == "NEUTRAL":
                    continue
                result.append({
                    "basket":          bname,
                    "market":          mkt[:80],
                    "zscore":          round(z, 3),
                    "deviation":       round(float(devs[mkt].iloc[-1]), 4),
                    "price":           round(float(df[mkt].iloc[-1]), 4),
                    "basket_price":    round(float(bmed.iloc[-1]), 4),
                    "signal":          signal,
                    "signal_strength": min(1.0, abs(z) / 4.0),
                })

        return sorted(result, key=lambda x: abs(x["zscore"]), reverse=True), len(baskets)


# ─────────────────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "status":             "initializing",
        "status_msg":         "Scanner starting…",
        "last_update":        None,
        "next_refresh":       None,
        "pair_signals":       [],
        "portfolio_signals":  [],
        "stats":              {},
    }
