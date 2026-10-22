"""
Polymarket Arb Scanner — FastAPI Web Server

Endpoints
---------
GET  /              → scanner dashboard (index.html)
GET  /backtest      → backtest results page (backtest.html)
GET  /api/state     → live scanner state (JSON)
GET  /api/backtest  → combined backtest results (JSON, 6h cached)
POST /api/refresh   → trigger scanner refresh
POST /api/backtest/run → force re-run backtest (clears cache)
WS   /ws            → WebSocket live updates
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from scanner import MarketScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

CACHE_DIR            = Path(__file__).parent / "cache"
BACKTEST_CACHE_FILE  = CACHE_DIR / "backtest_result.pkl"
BACKTEST_CACHE_TTL   = 6 * 3600   # 6 hours
FACTOR_CACHE_FILE    = CACHE_DIR / "factor_result.pkl"
FACTOR_CACHE_TTL     = 12 * 3600  # 12 hours (8 factors now; cal table has own 12h cache)
SW_CACHE_FILE        = CACHE_DIR / "short_winners.pkl"
SW_CACHE_TTL         = 10 * 60    # 10 min — live prices/spreads refresh each scan
SW_SCAN_INTERVAL     = 10 * 60    # background rescan period (seconds)


# ─────────────────────────────────────────────────────────────────────
# WebSocket connection manager
# ─────────────────────────────────────────────────────────────────────

class WSManager:
    def __init__(self) -> None:
        self._clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)
        logger.info("WS connected  (total: %d)", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients = [c for c in self._clients if c is not ws]
        logger.info("WS disconnected (total: %d)", len(self._clients))

    async def broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        msg  = json.dumps(payload, default=str)
        dead = []
        for ws in self._clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ─────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────

ws_manager      = WSManager()
scanner         = MarketScanner()
_loop: asyncio.AbstractEventLoop | None = None
_backtest_lock  = asyncio.Lock()
_factors_lock   = asyncio.Lock()
_sw_lock        = asyncio.Lock()
_sw_task: asyncio.Task | None = None


def _scanner_callback(state: dict) -> None:
    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast({"type": "update", "data": state}),
            _loop,
        )


scanner.add_callback(_scanner_callback)


# ─────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────

def _load_sw_cache() -> Optional[dict]:
    if not SW_CACHE_FILE.exists():
        return None
    age = time.time() - SW_CACHE_FILE.stat().st_mtime
    if age > SW_CACHE_TTL:
        return None
    try:
        return pickle.loads(SW_CACHE_FILE.read_bytes())
    except Exception:
        return None


async def _run_sw_scan() -> dict:
    """Run the short-winners scan in a worker thread, cache + broadcast."""
    from short_winners_scanner import scan_payload
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, scan_payload)
    if "error" not in data:
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            _atomic_pkl_write(SW_CACHE_FILE, data)
        except Exception as exc:
            logger.warning("Short-winners cache write failed: %s", exc)
    await ws_manager.broadcast({"type": "short_winners", "data": data})
    return data


async def _sw_periodic_loop() -> None:
    """Background loop: rescan short-winners every SW_SCAN_INTERVAL seconds."""
    await asyncio.sleep(5)   # let the server finish starting up
    while True:
        try:
            async with _sw_lock:
                data = await _run_sw_scan()
            n = data.get("n_candidates", 0)
            logger.info("Short-winners scan done: %s candidates", n)
        except Exception:
            logger.exception("Short-winners periodic scan failed")
        await asyncio.sleep(SW_SCAN_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop, _sw_task
    _loop = asyncio.get_event_loop()
    scanner.start()
    _sw_task = asyncio.create_task(_sw_periodic_loop())
    yield
    if _sw_task:
        _sw_task.cancel()
    scanner.stop()


app = FastAPI(title="Polymarket Arb Scanner", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ─────────────────────────────────────────────────────────────────────
# Backtest helpers
# ─────────────────────────────────────────────────────────────────────

def _load_backtest_cache() -> Optional[dict]:
    if not BACKTEST_CACHE_FILE.exists():
        return None
    age = time.time() - BACKTEST_CACHE_FILE.stat().st_mtime
    if age > BACKTEST_CACHE_TTL:
        return None
    try:
        return pickle.loads(BACKTEST_CACHE_FILE.read_bytes())
    except Exception:
        return None


def _atomic_pkl_write(path: Path, obj) -> None:
    """Write to a temp file then rename — atomic on NTFS and POSIX."""
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)


def _save_backtest_cache(data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _atomic_pkl_write(BACKTEST_CACHE_FILE, data)
    except Exception as exc:
        logger.warning("Backtest cache write failed: %s", exc)


def _series_to_list(s: pd.Series) -> list:
    return [
        {"date": str(idx.date()), "value": round(float(v), 4)}
        for idx, v in s.items()
        if not pd.isna(v)
    ]


def _trade_rows(trades: list, strategy: str) -> list:
    rows = []
    for t in trades:
        if strategy == "sw":
            name = getattr(t, "market_name", "")[:60]
            mom = getattr(t, "mom30", None)
            detail = f"mom30 {mom * 100:+.0f}%" if mom is not None else ""
        else:
            name   = getattr(t, "market_name", "")[:60]
            detail = getattr(t, "basket_name", "")

        rows.append({
            "strategy":  strategy,
            "name":      name,
            "detail":    detail,
            "direction": str(getattr(t, "direction", getattr(t, "side", ""))),
            "open_date": str(getattr(t, "open_date",  "")),
            "close_date": str(getattr(t, "close_date", "")),
            "pnl_usd":   round(float(getattr(t, "pnl_usd", 0)), 2),
            "closed_by": str(getattr(t, "closed_by", "")),
            "duration":  getattr(t, "duration_days", None),
        })
    return sorted(rows, key=lambda r: r["open_date"], reverse=True)


LOCAL_BT_CACHE   = CACHE_DIR / "bt_prices_local.pkl"
LOCAL_BT_TTL     = 12 * 3600
LOCAL_BT_TOP_N   = 400   # cap for pair-selection runtime (O(n²) cointegration)


def _load_local_backtest_prices() -> Optional[pd.DataFrame]:
    """本地存量数据回测面板：4000 市场 × 574 天 → 按总成交额取前 400。

    预处理与旧 API 管线语义一致（组合回测引擎按此设计）：
    每市场在 close_time 前 7 天截尾（去掉结算收敛段），再 clip [0.03, 0.97]。
    注意不能用"区间外置 NaN"——portfolio_arb 内部 ffill 会把内部 NaN
    变成人工水平线，制造虚假的篮子回归信号（已实测：胜率 9% / PF 7 的假象）。
    """
    if LOCAL_BT_CACHE.exists():
        age = time.time() - LOCAL_BT_CACHE.stat().st_mtime
        if age < LOCAL_BT_TTL:
            try:
                df = pickle.loads(LOCAL_BT_CACHE.read_bytes())
                logger.info("Local backtest panel cache hit: %d x %d", *df.shape)
                return df
            except Exception:
                pass
    try:
        from expanded_factor_research import load_panel
        prices, volume, meta = load_panel()
    except Exception as exc:
        logger.warning("Local panel unavailable (%s) — falling back to API fetch", exc)
        return None

    cols = volume.sum().sort_values(ascending=False).head(LOCAL_BT_TOP_N).index
    df = prices[cols]
    # Tail-trim each market 7 days before its close (resolution buffer)
    cutoff = (meta["close_d"] - pd.Timedelta(days=7)).reindex(cols)
    keep = pd.DataFrame(
        {c: df.index <= cutoff[c] for c in cols}, index=df.index
    )
    # Wide clip only (0.01/0.99): the 7d tail-trim already removes the
    # settlement zone; a narrow clip floor (0.03) caps LONG losses at -40%
    # for p=0.05 entries while leaving 17x upside — inflating profit factor.
    df = df.where(keep).clip(0.01, 0.99)
    df = df.dropna(how="all")

    qmap = meta["question"].fillna("")
    df.columns = [
        f"{(qmap.get(c) or str(c))[:48].strip()} [{c}]"
        for c in df.columns
    ]
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        _atomic_pkl_write(LOCAL_BT_CACHE, df)
    except Exception as exc:
        logger.warning("Local panel cache write failed: %s", exc)
    logger.info("Local backtest panel: %d days x %d markets "
                "(%s → %s)", df.shape[0], df.shape[1],
                df.index[0].date(), df.index[-1].date())
    return df


def _fetch_backtest_prices() -> pd.DataFrame:
    """Backtest prices: local parquet archive first, live API as fallback."""
    local = _load_local_backtest_prices()
    if local is not None and local.shape[1] >= 50:
        return local
    return _fetch_backtest_prices_api()


def _fetch_backtest_prices_api() -> pd.DataFrame:
    """Fetch a longer price history specifically for backtesting."""
    import hashlib
    from config import (
        BACKTEST_LOOKBACK_DAYS, BACKTEST_MIN_AGE_DAYS,
        BACKTEST_N_MARKETS, BACKTEST_MAX_MARKETS,
        MIN_DAILY_VOLUME_USD, RESOLUTION_BUFFER_DAYS, PRICE_CLIP_LOW, PRICE_CLIP_HIGH,
    )
    from data_fetcher import fetch_active_markets, build_price_matrix

    bt_params = {
        "min_age":  BACKTEST_MIN_AGE_DAYS,
        "max_mkts": BACKTEST_MAX_MARKETS,
        "min_vol":  MIN_DAILY_VOLUME_USD,
        "res_buf":  RESOLUTION_BUFFER_DAYS,
        "clip_lo":  PRICE_CLIP_LOW,
        "clip_hi":  PRICE_CLIP_HIGH,
    }
    fp = hashlib.md5(
        json.dumps(bt_params, sort_keys=True).encode()
    ).hexdigest()[:8]
    cache_path = CACHE_DIR / f"bt_prices_{BACKTEST_LOOKBACK_DAYS}d_{fp}.pkl"
    # Use a 12h TTL for backtest prices (changes less often than live prices)
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 12 * 3600:
            try:
                df = pickle.loads(cache_path.read_bytes())
                logger.info("Backtest price cache hit: %d markets x %d days",
                            df.shape[1], df.shape[0])
                return df
            except Exception:
                pass

    logger.info(
        "Fetching %d-day price history for backtest (up to %d markets, min_age=%dd)…",
        BACKTEST_LOOKBACK_DAYS, BACKTEST_MAX_MARKETS, BACKTEST_MIN_AGE_DAYS,
    )
    from data_fetcher import fetch_recently_closed_markets
    active_markets = fetch_active_markets(limit=BACKTEST_N_MARKETS)
    closed_markets = fetch_recently_closed_markets(limit=1000, closed_within_days=365)
    # Combine; deduplicate by question label so closed markets don't shadow active ones
    seen_labels: set = set()
    markets: list = []
    for m in active_markets + closed_markets:
        lbl = (m.get("question") or "")[:60].strip()
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            markets.append(m)
    logger.info(
        "Backtest pool: %d active + %d recently-closed = %d unique markets",
        len(active_markets), len(closed_markets), len(markets),
    )
    df = build_price_matrix(
        markets,
        days         = BACKTEST_LOOKBACK_DAYS,
        min_age_days = BACKTEST_MIN_AGE_DAYS,
        max_markets  = BACKTEST_MAX_MARKETS,
        bypass_cache = True,   # always rebuild from the full active+closed market list
    )
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        _atomic_pkl_write(cache_path, df)
        logger.info("Saved backtest prices: %d x %d", df.shape[0], df.shape[1])
    except Exception as exc:
        logger.warning("Backtest price cache write failed: %s", exc)
    return df


def _compute_backtest_sync(prices: pd.DataFrame) -> dict:
    """Run full combined backtest and return JSON-serialisable dict."""
    from combined_backtest import CombinedBacktest

    logger.info("Running combined backtest on %d markets x %d days…",
                prices.shape[1], prices.shape[0])
    t0 = time.time()

    engine = CombinedBacktest(prices, event_groups=[], initial_cap=10_000)
    result = engine.run()

    elapsed = time.time() - t0
    logger.info("Backtest completed in %.1fs", elapsed)

    m_c = result.metrics_combined
    m_p = result.metrics_sw
    m_a = result.metrics_port_arb

    last_date = prices.index[-1]
    eod_trades = [
        t for t in result.port_arb_trades
        if getattr(t, "closed_by", "") == "eod"
        and t.close_date is not None
        and pd.Timestamp(t.close_date).normalize() >= last_date - pd.Timedelta(days=1)
    ]
    eod_pnl = sum(getattr(t, "pnl_usd", 0.0) for t in eod_trades)

    data = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_sec":  round(elapsed, 1),
        "period": {
            "start":         str(prices.index[0].date()),
            "end":           str(prices.index[-1].date()),
            "days":          len(prices),
            "markets":       prices.shape[1],
            "eod_close_n":   len(eod_trades),
            "eod_close_pnl": round(eod_pnl, 2),
        },
        "combined": {
            "metrics": m_c,
            "equity":  _series_to_list(result.equity_combined),
            "drawdown": _series_to_list(
                (result.equity_combined / result.equity_combined.cummax() - 1) * 100
            ),
        },
        "short_winners": {
            "metrics": m_p,
            "equity":  _series_to_list(result.equity_sw),
            "trades":  _trade_rows(result.sw_trades, "sw"),
        },
        "portfolio_arb": {
            "metrics": m_a,
            "equity":  _series_to_list(result.equity_port_arb),
            "trades":  _trade_rows(result.port_arb_trades, "port"),
        },
        "all_trades": (
            _trade_rows(result.sw_trades, "sw") +
            _trade_rows(result.port_arb_trades, "port")
        ),
    }
    # sort combined trades
    data["all_trades"].sort(key=lambda r: r["open_date"], reverse=True)
    return data


# ─────────────────────────────────────────────────────────────────────
# HTTP endpoints
# ─────────────────────────────────────────────────────────────────────

# HTML pages must never be browser-cached — stale pages miss new panels/scripts.
_NO_CACHE = {"Cache-Control": "no-store, must-revalidate"}


def _page(name: str) -> FileResponse:
    return FileResponse(str(STATIC / name), headers=_NO_CACHE)


@app.get("/")
async def root():
    return _page("index.html")


@app.get("/backtest")
async def backtest_page():
    return _page("backtest.html")


@app.get("/factors")
async def factors_page():
    return _page("factors.html")


@app.get("/threefactor")
async def threefactor_page():
    return _page("threefactor.html")


@app.get("/shortwinners")
async def shortwinners_page():
    return _page("shortwinners.html")


@app.get("/api/short-winners")
async def api_short_winners(force: bool = Query(False)):
    """做空赢家候选（10 分钟缓存；force=true 立即重扫并广播）。"""
    if not force:
        cached = _load_sw_cache()
        if cached is not None:
            return JSONResponse(cached)

    async with _sw_lock:
        if not force:
            cached = _load_sw_cache()
            if cached is not None:
                return JSONResponse(cached)
        try:
            data = await _run_sw_scan()
            status = 500 if "error" in data else 200
            return JSONResponse(data, status_code=status)
        except Exception as exc:
            logger.exception("Short-winners scan failed")
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/state")
async def api_state():
    return JSONResponse(scanner.get_state())


@app.get("/api/backtest")
async def api_backtest(force: bool = Query(False)):
    # Fast path: serve from cache without acquiring the lock
    if not force:
        cached = _load_backtest_cache()
        if cached is not None:
            logger.info("Serving backtest from cache (age %.1fh)",
                        (time.time() - BACKTEST_CACHE_FILE.stat().st_mtime) / 3600)
            return JSONResponse(cached)

    # Only one concurrent computation at a time; second caller double-checks cache
    async with _backtest_lock:
        if not force:
            cached = _load_backtest_cache()
            if cached is not None:
                return JSONResponse(cached)

        loop = asyncio.get_running_loop()
        try:
            def _run():
                prices = _fetch_backtest_prices()
                if prices.empty or prices.shape[1] < 5:
                    raise RuntimeError("Insufficient price data for backtest")
                return _compute_backtest_sync(prices)

            data = await loop.run_in_executor(None, _run)
            _save_backtest_cache(data)
            return JSONResponse(data)
        except Exception as exc:
            logger.exception("Backtest failed")
            return JSONResponse({"error": str(exc)}, status_code=500)


def _load_factor_cache() -> Optional[dict]:
    if not FACTOR_CACHE_FILE.exists():
        return None
    age = time.time() - FACTOR_CACHE_FILE.stat().st_mtime
    if age > FACTOR_CACHE_TTL:
        return None
    try:
        return pickle.loads(FACTOR_CACHE_FILE.read_bytes())
    except Exception:
        return None


@app.get("/api/factors")
async def api_factors(force: bool = Query(False)):
    """Compute and return Phase-1 factor research data (12h cache)."""
    # Fast path: serve from cache without acquiring the lock
    if not force:
        cached = _load_factor_cache()
        if cached is not None:
            logger.info("Serving factor data from cache (age %.1fh)",
                        (time.time() - FACTOR_CACHE_FILE.stat().st_mtime) / 3600)
            return JSONResponse(cached)

    # Only one concurrent computation at a time; second caller double-checks cache
    async with _factors_lock:
        if not force:
            cached = _load_factor_cache()
            if cached is not None:
                return JSONResponse(cached)

        loop = asyncio.get_running_loop()
        try:
            def _run():
                return _compute_factors_sync(force=force)

            data = await loop.run_in_executor(None, _run)
            try:
                CACHE_DIR.mkdir(exist_ok=True)
                _atomic_pkl_write(FACTOR_CACHE_FILE, data)
            except Exception as exc:
                logger.warning("Factor cache write failed: %s", exc)
            return JSONResponse(data)
        except Exception as exc:
            logger.exception("Factor computation failed")
            return JSONResponse({"error": str(exc)}, status_code=500)


def _compute_factors_sync(force: bool = False) -> dict:
    """Run full factor pipeline and return JSON-serialisable dict."""
    import sys, importlib, time as _time
    # Always reload factor_engine to pick up any code changes without server restart
    if "factor_engine" in sys.modules:
        importlib.reload(sys.modules["factor_engine"])
    from config import BACKTEST_LOOKBACK_DAYS, BACKTEST_MIN_AGE_DAYS, BACKTEST_N_MARKETS, BACKTEST_MAX_MARKETS
    from data_fetcher import fetch_active_markets, build_price_matrix, _market_daily_volume, _market_end_date
    from factor_engine import run_all_factors, compute_factor_correlation, build_fm_inputs, run_fama_macbeth, compute_fm_composite_score
    from resolved_data import fetch_resolved_markets, load_calibration_table, calibration_rmse

    t0 = _time.time()

    # --- Prices (reuse backtest cache if warm) ---
    prices = _fetch_backtest_prices()
    if prices.empty or prices.shape[1] < 10:
        raise RuntimeError("Insufficient price data for factor analysis")

    # --- Market metadata for TTR + volume ---
    markets = fetch_active_markets(limit=BACKTEST_N_MARKETS)
    volume_map: dict = {}
    for m in markets:
        label = m.get("question", "")[:60].strip()
        if label:
            volume_map[label] = _market_daily_volume(m)

    # --- Factor computation ---
    factor_results = run_all_factors(prices, markets, volume_map)

    # --- Fama-MacBeth regression (Phase 2) ---
    logger.info("Building FM inputs (joint walk-forward)…")
    scores_by_date, returns_by_date = build_fm_inputs(prices, markets, volume_map)
    try:
        fm_result = run_fama_macbeth(scores_by_date, returns_by_date)
    except Exception as exc:
        logger.warning("FM regression failed: %s", exc)
        fm_result = None

    # FM composite score: beta-weighted z-scored factor signal for each market today
    fm_composite = compute_fm_composite_score(factor_results, fm_result)

    # --- Factor score correlation matrix ---
    score_dict = {name: fr.scores for name, fr in factor_results.items()}
    factor_corr = compute_factor_correlation(score_dict)

    # --- Calibration (PM-CAL) — uses 12h disk cache to avoid 300 CLOB API calls ---
    resolved  = fetch_resolved_markets(limit=2000)
    cal_table = load_calibration_table(resolved, n_bins=10, max_markets=300, force=force)
    rmse      = calibration_rmse(cal_table)

    elapsed = _time.time() - t0

    def _ser(s: pd.Series) -> list:
        return [{"date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                 "value": round(float(v), 6)}
                for idx, v in s.items() if not pd.isna(v)]

    def _factor_to_dict(fr) -> dict:
        # Use explicit float()/int()/bool() to convert numpy scalars → Python natives
        result = {
            "name":          fr.name,
            "description":   fr.description_zh,
            "n_markets":     int(fr.n_markets_used),
            "mean_ic":       round(float(fr.mean_ic), 4),
            "ic_tstat":      round(float(fr.ic_tstat), 3),
            "ic_pvalue":     round(float(fr.ic_pvalue), 4),
            "significant":   bool(abs(float(fr.ic_tstat)) >= 2.0),
            "scores":        [
                {"market": str(k)[:80], "score": round(float(v), 6)}
                for k, v in fr.scores.dropna().sort_values(ascending=False).items()
            ],
            "ic_series":     _ser(fr.ic_series),
            "decile_returns": [
                {"decile": int(d), "return_pct": round(float(v), 3)}
                for d, v in fr.decile_returns.items()
            ],
            "portfolio_equity": _ser(fr.portfolio_equity),
        }
        # Multi-window MOM scan (attached only to PM-MOM result)
        mw = getattr(fr, "mom_windows", None)
        if mw:
            result["mom_windows"] = {
                str(w): {
                    "mean_ic":    round(float(vals[0]), 4),
                    "ic_tstat":   round(float(vals[1]), 3),
                    "ic_pvalue":  round(float(vals[2]), 4),
                    "significant": bool(abs(float(vals[1])) >= 2.0),
                }
                for w, vals in mw.items()
            }
        return result

    def _fm_to_dict(fm, composite: pd.Series) -> dict:
        if fm is None:
            return {}
        from scipy.stats import t as _t
        n_periods = len(fm.beta_time_series)

        # Per-factor IC direction from univariate results (for sign-consistency check)
        ic_dir = {name: round(float(fr.mean_ic), 4) for name, fr in factor_results.items()}

        factor_detail = {}
        for k in fm.factor_names:
            prem  = float(fm.factor_premia[k])
            tstat = float(fm.t_statistics[k])
            pval  = float(fm.p_values[k])
            ic    = ic_dir.get(k)
            fm_sign = 1 if prem > 0 else -1
            ic_sign = (1 if ic > 0 else -1) if ic is not None else 0
            factor_detail[k] = {
                "premium_bp":    round(prem * 10000, 2),
                "t_statistic":   round(tstat, 3),
                "p_value":       round(pval, 4),
                "significant":   bool(abs(tstat) >= 2.0),
                "ic_mean":       ic,
                "sign_consistent": bool(fm_sign == ic_sign) if ic_sign != 0 else None,
            }

        # Beta + R² time series
        beta_rows = []
        for idx, row in fm.beta_time_series.iterrows():
            entry = {"date": str(idx.date())}
            for k, v in row.items():
                if not pd.isna(v):
                    entry[k] = round(float(v), 6)
            beta_rows.append(entry)

        # FM composite score: top/bottom 20
        comp_sorted = composite.sort_values(ascending=False)
        composite_top    = [{"market": str(m)[:80], "score_bp": round(float(v), 1)}
                            for m, v in comp_sorted.head(20).items()]
        composite_bottom = [{"market": str(m)[:80], "score_bp": round(float(v), 1)}
                            for m, v in comp_sorted.tail(20).items()]

        alpha_pval = round(float(
            2 * (1 - _t.cdf(abs(float(fm.alpha_tstat)), df=max(n_periods - 1, 1)))
        ), 4)

        return {
            "factor_names":    fm.factor_names,
            "factors":         factor_detail,
            "mean_r_squared":  round(float(fm.mean_r_squared), 4),
            "alpha_bp":        round(float(fm.alpha) * 10000, 2),
            "alpha_tstat":     round(float(fm.alpha_tstat), 3),
            "alpha_pvalue":    alpha_pval,
            "n_periods":       n_periods,
            "beta_series":     beta_rows,
            "composite_top":   composite_top,
            "composite_bottom": composite_bottom,
        }

    calibration_rows = []
    if not cal_table.empty:
        for _, row in cal_table.iterrows():
            se = row.get("std_error")
            calibration_rows.append({
                "price_low":      float(row["price_low"]),
                "price_high":     float(row["price_high"]),
                "price_midpoint": float(row["price_midpoint"]),
                "actual_rate":    float(row["actual_rate"]),
                "count":          int(row["count"]),
                "std_error":      float(se) if se is not None and se == se else None,
            })

    factor_corr_out = {}
    if not factor_corr.empty:
        for row_name, row_data in factor_corr.iterrows():
            factor_corr_out[str(row_name)] = {
                str(col): round(float(v), 3)
                for col, v in row_data.items()
            }

    return {
        "generated_at": datetime.now().isoformat(),
        "elapsed_sec":  round(elapsed, 1),
        "period": {
            "start":   str(prices.index[0].date()),
            "end":     str(prices.index[-1].date()),
            "days":    len(prices),
            "markets": prices.shape[1],
        },
        "factors": {name: _factor_to_dict(fr) for name, fr in factor_results.items()},
        "factor_correlation": factor_corr_out,
        "calibration": {
            "rmse":       round(rmse, 4) if not pd.isna(rmse) else None,
            "n_resolved": len(resolved),
            "table":      calibration_rows,
        },
        "fama_macbeth": _fm_to_dict(fm_result, fm_composite),
    }


@app.get("/api/research-summary")
async def api_research_summary():
    """扩样研究 + 策略原型 + lead-lag 终版结论（直接读取 logs/ 产出文件）。"""
    logs = Path(__file__).parent / "logs"

    def _csv_rows(name: str, first_col: Optional[str] = None):
        p = logs / name
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p)
            if first_col and df.columns[0].startswith("Unnamed"):
                df = df.rename(columns={df.columns[0]: first_col})
            return json.loads(df.to_json(orient="records"))
        except Exception:
            return None

    def _json_file(name: str):
        p = logs / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    return JSONResponse({
        "expanded_factors": _csv_rows("expanded_factor_summary.csv"),
        "expanded_fm":      _csv_rows("expanded_fm_results.csv", first_col="factor"),
        "combo":            _json_file("expanded_combo.json"),
        "momrev":           _json_file("momrev_strategy_summary.json"),
        "leadlag":          _csv_rows("leadlag_summary.csv"),
        "factor_longterm":  _json_file("factor_longterm.json"),
        "threefactor_search": _csv_rows("threefactor_search.csv"),
        "best_threefactor": _json_file("best_threefactor.json"),
    })


@app.post("/api/refresh")
async def api_refresh(force: bool = Query(False)):
    scanner.trigger_refresh(force=force)
    msg = "Force refresh triggered (cache cleared)" if force else "Refresh triggered"
    return {"ok": True, "msg": msg}


# ─────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        await ws.send_text(json.dumps(
            {"type": "init", "data": scanner.get_state()}, default=str
        ))

        while True:
            try:
                raw  = await asyncio.wait_for(ws.receive_text(), timeout=25.0)
                msg  = json.loads(raw)
                kind = msg.get("type", "")

                if kind == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                elif kind == "refresh":
                    scanner.trigger_refresh()
                    await ws.send_text(json.dumps({"type": "ack", "msg": "Refresh started"}))

            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "heartbeat"}))

    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception as e:
        logger.warning("WS error: %s", e)
        ws_manager.disconnect(ws)
