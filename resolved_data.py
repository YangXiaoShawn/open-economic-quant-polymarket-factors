"""
已结算市场数据 — 用于校准分析 (PM-CAL)

从 Gamma API 抓取 closed=true 的已结算市场，构建价格分位 vs 实际结算率校准表。
校准研究的核心问题：Polymarket 价格是否是良好的概率校准？

Longshot Bias：低概率事件（价格 < 0.15）往往被高估；
Favorite Bias ：高概率事件（价格 > 0.85）往往被低估。
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_fetcher import _session, _extract_yes_token, fetch_daily_prices, CACHE_DIR
from config import POLYMARKET_GAMMA_API

logger = logging.getLogger(__name__)


def _atomic_pkl_write(path: Path, obj) -> None:
    """Write to a temp file then rename — atomic on NTFS and POSIX."""
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)


GAMMA_MARKETS_URL     = f"{POLYMARKET_GAMMA_API}/markets"
RESOLVED_CACHE_FILE   = CACHE_DIR / "resolved_markets.pkl"
RESOLVED_CACHE_TTL    = 24 * 3600  # 24h — 已结算市场不频繁变动
CAL_TABLE_CACHE_FILE  = CACHE_DIR / "cal_table.pkl"
CAL_TABLE_CACHE_TTL   = 12 * 3600  # 12h — 校准表依赖 CLOB 历史价格，抓取慢，单独缓存
REQUEST_TIMEOUT       = 15
MIN_RESOLVED_MARKETS  = 20         # 校准曲线需要的最少市场数


# ---------------------------------------------------------------------------
# Outcome parsing
# ---------------------------------------------------------------------------

def _parse_outcome(market: Dict) -> Optional[int]:
    """
    从市场元数据解析 YES-token 结算结果。

    Gamma API 的 outcomePrices 字段格式为 JSON 字符串数组，
    index 0 对应 YES-token 的最终价格：
      "1" = YES 事件发生（YES-token 价值 $1）
      "0" = NO 事件发生（YES-token 价值 $0）

    备选：旧格式的 outcome / resolutionValue 字段。
    """
    import json as _json
    # Primary: outcomePrices[0] — final settlement price of YES token
    raw_prices = market.get("outcomePrices")
    if raw_prices:
        try:
            prices = _json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            yes_price = str(prices[0]).strip()
            if yes_price == "1":
                return 1
            if yes_price == "0":
                return 0
        except Exception:
            pass

    # Fallback: outcome / resolutionValue text fields
    for field in ("outcome", "resolutionValue"):
        raw = market.get(field)
        if raw is None:
            continue
        s = str(raw).strip().lower()
        if s in ("yes", "1", "true", "win", "correct"):
            return 1
        if s in ("no", "0", "false", "lose", "incorrect"):
            return 0
    return None


def _market_resolution_date(market: Dict) -> Optional[str]:
    """Return YYYY-MM-DD of market resolution (closedTime preferred, then endDate)."""
    for field in ("closedTime", "endDate"):
        raw = market.get(field)
        if raw and len(str(raw)) >= 10:
            return str(raw)[:10]
    return None


# ---------------------------------------------------------------------------
# Fetch resolved markets
# ---------------------------------------------------------------------------

def fetch_resolved_markets(limit: int = 2000) -> List[Dict]:
    """
    从 Gamma API 抓取已结算市场列表（closed=true），按成交量降序排列。
    保留有可解析 outcome 字段的市场（YES/NO 已知）。
    使用 24h 磁盘缓存，避免重复请求。
    """
    # --- Try disk cache ---
    if RESOLVED_CACHE_FILE.exists():
        age = time.time() - RESOLVED_CACHE_FILE.stat().st_mtime
        if age < RESOLVED_CACHE_TTL:
            try:
                markets = pickle.loads(RESOLVED_CACHE_FILE.read_bytes())
                logger.info(
                    "Resolved cache hit: %d markets (%.1fh old)",
                    len(markets), age / 3600,
                )
                return markets
            except Exception:
                pass

    logger.info("Fetching resolved markets from Gamma API (limit=%d)…", limit)
    markets: List[Dict] = []
    batch_size = 100

    for offset in range(0, limit, batch_size):
        try:
            resp = _session.get(
                GAMMA_MARKETS_URL,
                params={
                    "closed":    "true",
                    # NOTE: do NOT filter active=false — the Gamma API returns
                    # active=True for many genuinely resolved markets.
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
            logger.warning("Resolved markets batch offset=%d failed: %s", offset, exc)
            break

    # Keep only markets with parseable outcome
    valid = [m for m in markets if _parse_outcome(m) is not None]
    logger.info(
        "Fetched %d resolved markets, %d with known outcome",
        len(markets), len(valid),
    )

    # Save cache
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        _atomic_pkl_write(RESOLVED_CACHE_FILE, valid)
    except Exception as exc:
        logger.warning("Resolved cache write failed: %s", exc)

    return valid


# ---------------------------------------------------------------------------
# Calibration table builder
# ---------------------------------------------------------------------------

_CLOB_PRICES_URL = None  # filled lazily below

def _fetch_pre_settlement_price(market: Dict) -> Optional[float]:
    """
    从 CLOB API 取已结算市场的结算前预测价格。

    直接请求 YES-token 的全量日线价格历史（fidelity=1440），
    取最后一个非结算价格（0.01 < p < 0.99），代表市场"预测"。
    不使用 fetch_daily_prices()（它会应用 7 天 resolution buffer，
    会截掉大部分短期赛事市场的有效数据）。
    """
    from config import POLYMARKET_CLOB_API as _CLOB_API
    token_id = _extract_yes_token(market)
    if not token_id:
        return None
    try:
        resp = _session.get(
            f"{_CLOB_API}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": 1440},
            timeout=15,
        )
        resp.raise_for_status()
        hist = resp.json().get("history", [])
        if not hist:
            return None
        # Filter to pre-settlement prices only (exclude 0/1 settlement)
        pre_settle = [float(h["p"]) for h in hist if 0.01 < float(h["p"]) < 0.99]
        if pre_settle:
            return pre_settle[-1]
    except Exception:
        pass
    return None


def build_calibration_table(
    resolved_markets: List[Dict],
    n_bins: int = 10,
    max_markets: int = 300,
) -> pd.DataFrame:
    """
    构建校准表：结算前价格分位 → 实际 YES 结算率。

    对每个已结算市场：
      1. 解析 outcomePrices 得到 outcome (0 or 1)
      2. 调用 CLOB API 取结算前日线价格（末值，含 7 天 resolution buffer）
      3. 按价格等宽分 n_bins 个区间，统计各区间的 YES 率

    使用线程池并行抓取，最多 max_markets 个市场。

    Returns
    -------
    pd.DataFrame  columns: price_low, price_high, price_midpoint,
                           actual_rate, count, std_error
    """
    import concurrent.futures

    # Limit to top markets by volume
    sample = resolved_markets[:max_markets]

    def _process_one(m: Dict) -> Optional[Tuple[float, int]]:
        outcome = _parse_outcome(m)
        if outcome is None:
            return None
        price = _fetch_pre_settlement_price(m)
        if price is None:
            return None
        return (price, outcome)

    records: List[Tuple[float, int]] = []
    logger.info("Fetching pre-settlement prices for %d resolved markets…", len(sample))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for result in pool.map(_process_one, sample):
            if result is not None:
                records.append(result)
    logger.info("Got pre-settlement price for %d / %d markets", len(records), len(sample))

    if len(records) < MIN_RESOLVED_MARKETS:
        logger.warning(
            "Only %d records with price+outcome — calibration table may be sparse",
            len(records),
        )

    if not records:
        return pd.DataFrame(columns=[
            "price_low", "price_high", "price_midpoint",
            "actual_rate", "count", "std_error",
        ])

    df = pd.DataFrame(records, columns=["price", "outcome"])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["price"], bins=bins, include_lowest=True)

    rows = []
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask = df["bin"] == df["bin"].cat.categories[i]
        sub  = df[mask]
        n    = len(sub)
        if n == 0:
            continue
        rate = sub["outcome"].mean()
        se   = np.sqrt(rate * (1 - rate) / n) if n > 1 else np.nan
        rows.append({
            "price_low":       round(low,  3),
            "price_high":      round(high, 3),
            "price_midpoint":  round((low + high) / 2, 3),
            "actual_rate":     round(rate, 4),
            "count":           n,
            "std_error":       round(se, 4) if not np.isnan(se) else None,
        })

    result = pd.DataFrame(rows)
    logger.info(
        "Calibration table: %d bins, %d total records, avg count/bin=%.0f",
        len(result), len(df), len(df) / max(len(result), 1),
    )
    return result


def calibration_rmse(cal_table: pd.DataFrame) -> float:
    """
    校准均方根误差 RMSE = sqrt(mean((predicted - actual)²))。
    完美校准时 RMSE = 0；随机乱猜约 0.29。
    实际市场通常 0.05 – 0.15。
    """
    if cal_table.empty:
        return float("nan")
    diff = cal_table["price_midpoint"] - cal_table["actual_rate"]
    return float(np.sqrt((diff ** 2).mean()))


def load_calibration_table(
    resolved_markets: List[Dict],
    n_bins: int = 10,
    max_markets: int = 300,
    force: bool = False,
) -> pd.DataFrame:
    """
    带磁盘缓存的校准表加载器（12h TTL）。

    校准表构建需要对 300 个已结算市场调用 CLOB API 获取历史价格（≈15-30s），
    但结果在 12h 内基本稳定（已结算市场不会改变），因此独立缓存以避免重复调用。
    """
    if not force and CAL_TABLE_CACHE_FILE.exists():
        age = time.time() - CAL_TABLE_CACHE_FILE.stat().st_mtime
        if age < CAL_TABLE_CACHE_TTL:
            try:
                table = pickle.loads(CAL_TABLE_CACHE_FILE.read_bytes())
                logger.info(
                    "Cal table cache hit: %d bins (%.1fh old)",
                    len(table), age / 3600,
                )
                return table
            except Exception:
                pass

    table = build_calibration_table(resolved_markets, n_bins=n_bins, max_markets=max_markets)

    try:
        CACHE_DIR.mkdir(exist_ok=True)
        _atomic_pkl_write(CAL_TABLE_CACHE_FILE, table)
        logger.info("Cal table cache written (%d bins)", len(table))
    except Exception as exc:
        logger.warning("Cal table cache write failed: %s", exc)

    return table
