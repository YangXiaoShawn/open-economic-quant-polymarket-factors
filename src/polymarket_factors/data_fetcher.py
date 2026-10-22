"""
Fetch real Polymarket market data.

Data flow:
  1. Gamma API  →  market metadata + clobTokenIds
  2. CLOB API   →  daily price history per YES-token (parallel, cached)
  3. Build wide price matrix (dates × markets) for the strategy

Cache:
  Price matrix is saved to cache/prices_<date>.parquet with a 4-hour TTL.
  On warm restarts the cache is loaded instantly (~0.1s vs ~60s for live fetch).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    POLYMARKET_GAMMA_API,
    POLYMARKET_CLOB_API,
    DEFAULT_LOOKBACK_DAYS,
    MIN_DATA_POINTS,
    MIN_DAILY_VOLUME_USD,
    RESOLUTION_BUFFER_DAYS,
    PRICE_CLIP_HIGH,
    PRICE_CLIP_LOW,
)

logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = f"{POLYMARKET_GAMMA_API}/markets"
CLOB_HISTORY_URL  = f"{POLYMARKET_CLOB_API}/prices-history"
REQUEST_TIMEOUT   = 15
MAX_WORKERS       = 40      # parallel HTTP workers
CACHE_TTL_HOURS   = 4       # reuse cached data for up to 4 hours
CACHE_DIR         = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# HTTP session (connection pooling)
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update({"User-Agent": "polymarket-arb-scanner/1.0"})
# Retry with exponential backoff on 429 (rate-limit) and transient 5xx errors.
# backoff_factor=1 → waits 0s, 2s, 4s between retries.
_retry = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(
    pool_connections=MAX_WORKERS,
    pool_maxsize=MAX_WORKERS * 2,
    max_retries=_retry,
)
_session.mount("https://", _adapter)
_session.mount("http://",  _adapter)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _config_fingerprint() -> str:
    """8-char MD5 of the parameters that affect price matrix contents."""
    import hashlib
    params = {
        "min_vol":     MIN_DAILY_VOLUME_USD,
        "res_buf":     RESOLUTION_BUFFER_DAYS,
        "clip_lo":     PRICE_CLIP_LOW,
        "clip_hi":     PRICE_CLIP_HIGH,
    }
    return hashlib.md5(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:8]


def _cache_path(days: int) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    today = datetime.now(timezone.utc).date()
    fp = _config_fingerprint()
    return CACHE_DIR / f"prices_{today}_{days}d_{fp}.pkl"


def _load_cache(days: int) -> Optional[pd.DataFrame]:
    path = _cache_path(days)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > CACHE_TTL_HOURS:
        return None
    try:
        df = pickle.loads(path.read_bytes())
        logger.info("Cache hit: %s (%.1fh old, %d markets)", path.name, age_hours, len(df.columns))
        return df
    except Exception as exc:
        logger.warning("Cache read failed: %s", exc)
        return None


def _atomic_pkl_write(path: Path, obj) -> None:
    """Write to a temp file then rename — atomic on NTFS and POSIX."""
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)


def _save_cache(df: pd.DataFrame, days: int) -> None:
    try:
        path = _cache_path(days)
        _atomic_pkl_write(path, df)
        logger.info("Saved price matrix to cache: %s", path.name)
    except Exception as exc:
        logger.warning("Cache write failed: %s", exc)


def clear_cache() -> None:
    """Remove all cached price files (call to force a fresh fetch)."""
    for f in CACHE_DIR.glob("prices_*.pkl"):
        f.unlink(missing_ok=True)
    logger.info("Cache cleared")


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

def fetch_active_markets(limit: int = 500) -> List[Dict]:
    """
    Return active markets from the Gamma API, ordered by total volume.
    Only includes markets that have clobTokenIds (needed for CLOB history).
    """
    markets: List[Dict] = []
    batch_size = 100
    for offset in range(0, limit, batch_size):
        try:
            resp = _session.get(
                GAMMA_MARKETS_URL,
                params={
                    "active":    "true",
                    "closed":    "false",
                    "limit":     batch_size,
                    "offset":    offset,
                    "order":     "volume",
                    "ascending": "false",
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            markets.extend(batch)
        except Exception as exc:
            logger.warning("Gamma API batch offset=%d failed: %s", offset, exc)
            break

    filtered = [m for m in markets if _extract_yes_token(m)]
    logger.info("Fetched %d active markets (%d with CLOB tokens)", len(markets), len(filtered))
    return filtered


def fetch_recently_closed_markets(limit: int = 1000, closed_within_days: int = 180) -> List[Dict]:
    """
    Return recently closed markets from the Gamma API with CLOB token IDs.
    Used alongside active markets to build a larger, stable backtest price matrix.
    Only includes markets that closed within `closed_within_days` days — older markets
    have prices outside the 365-day backtest window and contribute nothing.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=closed_within_days)
    markets: List[Dict] = []
    batch_size = 100
    for offset in range(0, limit, batch_size):
        try:
            resp = _session.get(
                GAMMA_MARKETS_URL,
                params={
                    "closed":    "true",
                    "limit":     batch_size,
                    "offset":    offset,
                    "order":     "volume",
                    "ascending": "false",
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for m in batch:
                end_raw = m.get("endDate") or m.get("resolutionDate") or ""
                try:
                    end_ts = datetime.fromisoformat(end_raw[:19]).replace(tzinfo=timezone.utc)
                    if end_ts >= cutoff:
                        markets.append(m)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Closed markets batch offset=%d failed: %s", offset, exc)
            break

    filtered = [m for m in markets if _extract_yes_token(m)]
    logger.info(
        "Fetched %d recently-closed markets (closed within %dd, %d with CLOB tokens)",
        len(markets), closed_within_days, len(filtered),
    )
    return filtered


def _extract_yes_token(market: Dict) -> Optional[str]:
    """Return the YES-token ID string (index 0 of clobTokenIds), or None."""
    raw = market.get("clobTokenIds", "[]")
    try:
        ids = json.loads(raw)
        return str(ids[0]) if ids else None
    except Exception:
        return None


def _market_age_days(market: Dict) -> int:
    start_str = market.get("startDate", "")[:19]
    try:
        return max(0, (datetime.now() - datetime.fromisoformat(start_str)).days)
    except Exception:
        return 0


def _market_daily_volume(market: Dict) -> float:
    """Estimate average daily USD volume.  Prefers volume24hr; falls back to total/age."""
    v24 = market.get("volume24hr")
    if v24:
        try:
            val = float(v24)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass
    # Fallback: lifetime total / active period
    # For closed/resolved markets use (endDate - startDate) as the active period so that
    # the per-day average isn't diluted by months of inactivity after resolution.
    for field in ("volume", "volumeClob"):
        raw = market.get(field)
        if raw:
            try:
                total = float(raw)
                if total > 0:
                    # Try to get actual active period (startDate → endDate)
                    start_str = market.get("startDate", "")[:19]
                    end_str   = (market.get("endDate") or market.get("resolutionDate") or "")[:19]
                    active_days = 0
                    if start_str and end_str:
                        try:
                            t_start = datetime.fromisoformat(start_str)
                            t_end   = datetime.fromisoformat(end_str)
                            active_days = max((t_end - t_start).days, 1)
                        except Exception:
                            pass
                    if active_days == 0:
                        active_days = max(_market_age_days(market), 1)
                    return total / active_days
            except (ValueError, TypeError):
                pass
    return 0.0


def _market_end_date(market: Dict) -> Optional[str]:
    """Return market end/resolution date string (YYYY-MM-DD) or None."""
    for field in ("endDate", "resolutionDate"):
        raw = market.get(field)
        if raw and len(raw) >= 10:
            return raw[:10]
    return None


# ---------------------------------------------------------------------------
# Price history (single market)
# ---------------------------------------------------------------------------

def fetch_daily_prices(
    token_id: str,
    days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Fetch daily YES-price history from the CLOB API.

    Uses interval=max + fidelity=1440 (daily candles).

    end_date (YYYY-MM-DD): if provided, data within RESOLUTION_BUFFER_DAYS of the
    market's resolution date is excluded to prevent look-ahead from price convergence.
    Prices are clipped to [PRICE_CLIP_LOW, PRICE_CLIP_HIGH] to remove extreme
    near-resolution noise regardless.
    """
    try:
        resp = _session.get(
            CLOB_HISTORY_URL,
            params={"market": token_id, "interval": "max", "fidelity": 1440},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        history = payload.get("history", [])
    except Exception as exc:
        logger.debug("CLOB history failed for token %s: %s", token_id[:20], exc)
        return None

    if len(history) < MIN_DATA_POINTS:
        return None

    records = [
        (pd.Timestamp(pt["t"], unit="s", tz="UTC").normalize(), float(pt["p"]))
        for pt in history
    ]
    series = pd.Series(
        {ts: price for ts, price in records},
        dtype=float,
        name=token_id,
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()

    # Apply lookback window cutoff
    cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=days)
    series = series[series.index >= cutoff]

    # Exclude data near resolution date (prevents look-ahead from price convergence)
    if end_date:
        try:
            end_ts     = pd.Timestamp(end_date[:10]).tz_localize("UTC")
            buffer_ts  = end_ts - pd.Timedelta(days=RESOLUTION_BUFFER_DAYS)
            series     = series[series.index <= buffer_ts]
        except Exception:
            pass

    if len(series) < MIN_DATA_POINTS:
        return None

    # Clip prices to remove extreme convergence noise (near-resolution markets)
    series = series.clip(lower=PRICE_CLIP_LOW, upper=PRICE_CLIP_HIGH)

    return series


# ---------------------------------------------------------------------------
# Price matrix builder (parallel fetch + cache)
# ---------------------------------------------------------------------------

def _fetch_one(args: Tuple) -> Tuple[str, Optional[pd.Series]]:
    """Worker: fetch a single market's price history. Returns (label, series)."""
    label, token, days, end_date = args
    series = fetch_daily_prices(token, days=days, end_date=end_date)
    return label, series


def build_price_matrix(
    markets: List[Dict],
    days: int = DEFAULT_LOOKBACK_DAYS,
    min_age_days: int = 30,
    max_markets: int = 500,
    bypass_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch price histories in parallel and return a wide DataFrame.
    Uses on-disk cache (TTL=4h) to avoid repeated network calls.

    Columns : market question (truncated to 60 chars)
    Index   : UTC dates
    bypass_cache: skip the read cache (still writes the result)
    """
    # --- Try cache first ---
    if not bypass_cache:
        cached = _load_cache(days)
        if cached is not None:
            return cached

    # --- Filter candidates by age ---
    candidates = [
        m for m in markets
        if _market_age_days(m) >= min_age_days
    ][:max_markets]

    logger.info(
        "Parallel fetch: %d candidates (age>=%dd, %d workers) ...",
        len(candidates), min_age_days, MAX_WORKERS,
    )

    # Build (label, token, days, end_date) tuples — ensure unique labels up front
    tasks: List[Tuple[str, str, int, Optional[str]]] = []
    seen: Dict[str, int] = {}
    skipped_vol = 0
    for mkt in candidates:
        token = _extract_yes_token(mkt)
        if not token:
            continue
        # Volume filter: skip illiquid markets (unexecutable in practice)
        if MIN_DAILY_VOLUME_USD > 0 and _market_daily_volume(mkt) < MIN_DAILY_VOLUME_USD:
            skipped_vol += 1
            continue
        base     = mkt.get("question", token)[:60].strip()
        count    = seen.get(base, 0)
        label    = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        end_date = _market_end_date(mkt)
        tasks.append((label, token, days, end_date))
    if skipped_vol:
        logger.info("Volume filter: skipped %d markets below $%.0f/day", skipped_vol, MIN_DAILY_VOLUME_USD)

    rows: Dict[str, pd.Series] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            label, series = future.result()
            done += 1
            if series is not None:
                rows[label] = series
            if done % 50 == 0:
                logger.info("  ... %d/%d fetched, %d usable", done, len(tasks), len(rows))

    if len(rows) < 5:
        raise RuntimeError(
            f"只抓到 {len(rows)} 个有效市场（最少需要 5 个）。"
            "请检查网络连接或 API 限额。不允许使用合成数据降级。"
        )

    df = pd.DataFrame(rows)
    df = df.dropna(thresh=MIN_DATA_POINTS)
    logger.info("Price matrix: %d days x %d markets", len(df), len(df.columns))
    _save_cache(df, days)
    return df


# ---------------------------------------------------------------------------
# Synthetic fallback (offline / test)
# ---------------------------------------------------------------------------

def _synthetic_price_matrix(
    n_markets: int = 20,
    days: int = DEFAULT_LOOKBACK_DAYS,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic synthetic binary-market price paths. FOR UNIT TESTS ONLY."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        end=datetime.now(timezone.utc).date(),
        periods=days,
        freq="D",
        tz="UTC",
    )
    prices: Dict[str, np.ndarray] = {}

    pair_labels = [
        ("ELECTION_A_YES", "ELECTION_A_NO"),
        ("SPORTS_TEAM1_WIN", "SPORTS_TEAM2_WIN"),
        ("CRYPTO_BTC_ABOVE", "CRYPTO_BTC_BELOW"),
        ("MACRO_RATE_HIKE", "MACRO_RATE_CUT"),
        ("TECH_STOCK_UP", "TECH_STOCK_DOWN"),
    ]
    for label_a, label_b in pair_labels:
        common  = np.cumsum(rng.normal(0, 0.01, days))
        noise_a = np.cumsum(rng.normal(0, 0.005, days))
        noise_b = np.cumsum(rng.normal(0, 0.005, days))
        prices[label_a] = np.clip(0.5 + common + noise_a, 0.02, 0.98)
        prices[label_b] = np.clip(0.5 - common + noise_b, 0.02, 0.98)

    for i in range(n_markets - len(pair_labels) * 2):
        path = 0.5 + np.cumsum(rng.normal(0, 0.012, days))
        prices[f"RANDOM_MARKET_{i}"] = np.clip(path, 0.02, 0.98)

    df = pd.DataFrame(prices, index=dates)
    logger.info("Synthetic price matrix: %d days x %d markets", len(df), len(df.columns))
    return df
