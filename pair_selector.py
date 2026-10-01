"""
Pair selection for Polymarket markets.

Pipeline:
  1. Vectorized correlation matrix  – filter 89%+ of pairs instantly (O(n²) numpy)
  2. Cointegration test (parallel)  – Engle-Granger ADF on high-corr residuals only
  3. Half-life filter               – reject HL < MIN_HALF_LIFE_DAYS (spurious binary)
                                      and HL > MAX_HALF_LIFE_DAYS (too slow to trade)
  4. Return the best pairs sorted by composite score
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConstantInputWarning.*")
warnings.filterwarnings("ignore", message=".*An input array is constant.*")
warnings.filterwarnings("ignore", message=".*perfectly colinear.*")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from config import (
    COINTEGRATION_PVALUE,
    MIN_CORRELATION,
    MAX_HALF_LIFE_DAYS,
    MIN_HALF_LIFE_DAYS,
    MIN_DATA_POINTS,
)

# Minimum times the spread must cross its own mean in the training window.
# Filters out trending/directional pairs that are spuriously cointegrated
# (e.g., a market that sharply resolves, skewing the ADF residual).
MIN_SPREAD_MEAN_CROSSINGS = 10

logger = logging.getLogger(__name__)

COINT_WORKERS = 64   # parallel threads for cointegration tests


@dataclass
class PairInfo:
    market_a:       str
    market_b:       str
    correlation:    float
    coint_pvalue:   float
    hedge_ratio:    float
    half_life_days: float
    spread_mean:    float
    spread_std:     float
    score: float = field(init=False)

    def __post_init__(self) -> None:
        self.score = (
            (1 - self.coint_pvalue)
            * abs(self.correlation)
            / (1 + self.half_life_days / 10)
        )

    def __repr__(self) -> str:
        return (
            f"PairInfo({self.market_a[:40]} / {self.market_b[:40]} | "
            f"corr={self.correlation:.3f}, p={self.coint_pvalue:.4f}, "
            f"HL={self.half_life_days:.1f}d)"
        )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """OLS hedge ratio using numpy lstsq — ~10x faster than statsmodels OLS."""
    X = np.column_stack([np.ones(len(x)), x])
    betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(betas[1])


def _half_life(spread: np.ndarray) -> float:
    """AR(1) half-life using numpy lstsq — ~10x faster than statsmodels OLS."""
    delta = np.diff(spread)
    lag   = spread[:-1]
    X     = np.column_stack([np.ones(len(lag)), lag])
    betas, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
    lam = betas[1]
    if lam >= 0:
        return float("inf")
    return -np.log(2) / lam


# ---------------------------------------------------------------------------
# Worker: full test for one high-correlation pair
# ---------------------------------------------------------------------------

def _test_pair(
    col_a: str,
    col_b: str,
    a: np.ndarray,
    b: np.ndarray,
    corr: float,
) -> Optional[PairInfo]:
    """Cointegration + half-life test. Returns PairInfo or None."""
    # Cointegration (Engle-Granger)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pvalue, _ = coint(a, b)
    except Exception:
        return None
    if pvalue > COINTEGRATION_PVALUE:
        return None

    # OLS hedge ratio
    try:
        beta = _ols_hedge_ratio(a, b)
    except Exception:
        return None

    spread_arr = a - beta * b

    # Half-life filter
    try:
        hl = _half_life(spread_arr)
    except Exception:
        return None

    if not (MIN_HALF_LIFE_DAYS <= hl <= MAX_HALF_LIFE_DAYS):
        return None

    # Mean-crossing filter: spread must cross its mean at least N times.
    # Trending/directional pairs (spurious ADF artifacts) cross rarely.
    centered = spread_arr - np.mean(spread_arr)
    signs = np.sign(centered)
    signs[signs == 0] = 1  # treat exact-zero as positive
    crossings = int(np.sum(np.diff(signs) != 0))
    if crossings < MIN_SPREAD_MEAN_CROSSINGS:
        return None

    return PairInfo(
        market_a=col_a,
        market_b=col_b,
        correlation=corr,
        coint_pvalue=pvalue,
        hedge_ratio=beta,
        half_life_days=hl,
        spread_mean=float(np.mean(spread_arr)),
        spread_std=float(np.std(spread_arr)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_pairs(
    prices: pd.DataFrame,
    top_n: int = 30,
) -> List[PairInfo]:
    """
    Screen all column-pairs and return the best cointegrated pairs.

    Two-stage pipeline:
      Stage 1 — vectorized correlation matrix (O(n²) numpy, ~0.005s for n=200)
               filters out ~90% of pairs before any expensive stats are run.
      Stage 2 — parallel Engle-Granger cointegration on remaining candidates.
    """
    # limit=3: forward-fill gaps up to 3 days only; prevents resolved markets
    # (price stuck at 0/1) from being artificially treated as cointegrated pairs.
    prices  = prices.dropna(axis=1, thresh=MIN_DATA_POINTS).ffill(limit=3)
    columns = list(prices.columns)
    n       = len(columns)
    n_total = n * (n - 1) // 2

    # ── Stage 1: vectorized correlation matrix ────────────────────────
    corr_matrix = prices.corr().values          # numpy array, shape (n, n)
    rows, cols  = np.where(np.triu(np.abs(corr_matrix) >= MIN_CORRELATION, k=1))
    n_coint     = len(rows)
    logger.info(
        "Stage 1: %d/%d pairs pass corr>=%.2f (%.0f%% filtered) from %d markets",
        n_coint, n_total, MIN_CORRELATION,
        100 * (1 - n_coint / max(n_total, 1)),
        n,
    )

    # ── Stage 2: parallel cointegration on high-corr pairs ───────────
    # Pre-extract aligned arrays to avoid repeated pandas slicing in threads.
    # Require 2×MIN_DATA_POINTS of shared history for a reliable ADF test.
    MIN_JOINT = max(MIN_DATA_POINTS * 2, 40)
    tasks: List[Tuple] = []
    for r, c in zip(rows, cols):
        col_a, col_b = columns[r], columns[c]
        joint = prices[[col_a, col_b]].dropna()
        if len(joint) < MIN_JOINT:
            continue
        tasks.append((col_a, col_b, joint[col_a].values, joint[col_b].values,
                      float(corr_matrix[r, c])))

    valid_pairs: List[PairInfo] = []
    with ThreadPoolExecutor(max_workers=COINT_WORKERS) as pool:
        futures = {pool.submit(_test_pair, *t): t[:2] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                valid_pairs.append(result)

    valid_pairs.sort(key=lambda p: p.score, reverse=True)
    kept = valid_pairs[:top_n]
    logger.info(
        "Stage 2: %d/%d pairs pass coint+HL (HL %.0f–%.0fd); returning top %d",
        len(valid_pairs), n_coint,
        MIN_HALF_LIFE_DAYS, MAX_HALF_LIFE_DAYS,
        len(kept),
    )
    return kept


def compute_rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    mu    = spread.rolling(window, min_periods=window // 2).mean()
    sigma = spread.rolling(window, min_periods=window // 2).std()
    return (spread - mu) / sigma.replace(0, np.nan)


def compute_zscores_fast(
    prices: pd.DataFrame,
    pairs: List[PairInfo],
    window: int = 20,
) -> List[Tuple[PairInfo, float]]:
    """
    Recompute only the latest z-score for existing pairs (no cointegration retest).
    Used for the fast 60s refresh tier.
    Returns list of (pair, zscore) tuples.
    """
    results = []
    for p in pairs:
        try:
            joint  = prices[[p.market_a, p.market_b]].dropna()
            if len(joint) < window:
                continue
            spread = joint[p.market_a] - p.hedge_ratio * joint[p.market_b]
            zs     = compute_rolling_zscore(spread, window)
            z      = float(zs.iloc[-1])
            if not pd.isna(z):
                results.append((p, z))
        except Exception:
            pass
    return results


def recompute_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int,
) -> pd.Series:
    ratios = pd.Series(index=price_a.index, dtype=float)
    for i in range(window, len(price_a) + 1):
        try:
            ratios.iloc[i - 1] = _ols_hedge_ratio(
                price_a.iloc[i - window:i].values,
                price_b.iloc[i - window:i].values,
            )
        except Exception:
            pass
    return ratios
