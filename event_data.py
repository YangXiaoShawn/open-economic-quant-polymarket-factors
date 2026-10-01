"""
Event-level data fetcher for Polymarket portfolio and complete-event arbitrage.

Provides:
  - Multi-outcome event groups (e.g., election with N candidates)
  - Historical YES price matrices per event
  - Sum-deviation time series for arbitrage signal generation
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from config import POLYMARKET_GAMMA_API, POLYMARKET_CLOB_API, MIN_DATA_POINTS

logger = logging.getLogger(__name__)

GAMMA_EVENTS_URL = f"{POLYMARKET_GAMMA_API}/events"
CLOB_HISTORY_URL = f"{POLYMARKET_CLOB_API}/prices-history"
REQUEST_TIMEOUT  = 12
SLEEP_SEC        = 0.12


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class EventGroup:
    """
    A group of mutually exclusive prediction markets belonging to one event.
    E.g., all candidates in an election, or all teams in a tournament.

    The sum of all YES prices should theoretically equal 1.0 for truly
    mutually exclusive & exhaustive outcomes.
    """

    def __init__(
        self,
        event_id: str,
        title: str,
        markets: List[Dict],
    ) -> None:
        self.event_id = event_id
        self.title = title
        self.markets = markets            # list of market dicts from Gamma API
        self.price_matrix: Optional[pd.DataFrame] = None   # dates x candidates
        self.sum_series: Optional[pd.Series]   = None      # daily sum of YES prices

    @property
    def n_markets(self) -> int:
        return len(self.markets)

    def __repr__(self) -> str:
        pts = len(self.price_matrix) if self.price_matrix is not None else 0
        return f"EventGroup({self.title[:40]!r}, n={self.n_markets}, days={pts})"


# ---------------------------------------------------------------------------
# Event discovery
# ---------------------------------------------------------------------------

def fetch_event_groups(
    min_markets: int = 3,
    min_total_volume: float = 50_000,
    limit: int = 100,
) -> List[EventGroup]:
    """
    Return EventGroup objects for large multi-outcome events.

    Only includes events where:
      - ≥ min_markets candidates exist
      - Total cumulative volume ≥ min_total_volume
    """
    groups: List[EventGroup] = []
    try:
        resp = requests.get(
            GAMMA_EVENTS_URL,
            params={"limit": limit, "order": "volume", "ascending": "false"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch events: %s", exc)
        return groups

    for ev in events:
        mkts = ev.get("markets", [])
        if not isinstance(mkts, list) or len(mkts) < min_markets:
            continue

        total_vol = sum(float(m.get("volume") or 0) for m in mkts)
        if total_vol < min_total_volume:
            continue

        # Keep only markets with CLOB token IDs
        viable = [m for m in mkts if _yes_token(m)]
        if len(viable) < min_markets:
            continue

        groups.append(
            EventGroup(
                event_id=str(ev.get("id", "")),
                title=ev.get("title", ""),
                markets=viable,
            )
        )

    logger.info("Found %d qualifying event groups", len(groups))
    return groups


def _yes_token(market: Dict) -> Optional[str]:
    raw = market.get("clobTokenIds", "[]")
    try:
        ids = json.loads(raw)
        return str(ids[0]) if ids else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Historical price fetching
# ---------------------------------------------------------------------------

def fetch_event_price_history(
    group: EventGroup,
    days: int = 365,
    min_pts: int = MIN_DATA_POINTS,
    max_candidates: int = 20,
) -> EventGroup:
    """
    Populate group.price_matrix and group.sum_series with historical data.

    Fetches YES prices for each candidate and aligns on a shared date index.
    """
    series_dict: Dict[str, pd.Series] = {}
    candidates = group.markets[:max_candidates]

    for mkt in candidates:
        token = _yes_token(mkt)
        if not token:
            continue

        try:
            resp = requests.get(
                CLOB_HISTORY_URL,
                params={"market": token, "interval": "max", "fidelity": 1440},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            history = resp.json().get("history", [])
        except Exception:
            time.sleep(SLEEP_SEC)
            continue

        if len(history) < min_pts:
            time.sleep(SLEEP_SEC)
            continue

        label = mkt.get("question", token)[:50]
        records = {
            pd.Timestamp(pt["t"], unit="s", tz="UTC").normalize(): float(pt["p"])
            for pt in history
        }
        s = pd.Series(records, dtype=float, name=label)
        s = s[~s.index.duplicated(keep="last")].sort_index()

        # Restrict to lookback window
        cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=days)
        s = s[s.index >= cutoff]

        if len(s) >= min_pts:
            series_dict[label] = s

        time.sleep(SLEEP_SEC)

    if len(series_dict) < 2:
        return group

    df = pd.DataFrame(series_dict).sort_index()
    group.price_matrix = df
    group.sum_series = df.sum(axis=1)   # sum of YES prices per day
    logger.debug("Event '%s': %d candidates, %d days", group.title[:40], len(series_dict), len(df))
    return group


def load_all_event_data(
    days: int = 365,
    min_markets: int = 3,
    min_volume: float = 50_000,
) -> List[EventGroup]:
    """Convenience: fetch all event groups and populate their price histories."""
    groups = fetch_event_groups(
        min_markets=min_markets,
        min_total_volume=min_volume,
    )
    enriched = []
    for i, g in enumerate(groups, 1):
        logger.info("[%d/%d] Fetching history for: %s", i, len(groups), g.title[:50])
        g = fetch_event_price_history(g, days=days)
        if g.price_matrix is not None and len(g.price_matrix) >= MIN_DATA_POINTS:
            enriched.append(g)
    logger.info("Loaded %d event groups with usable history", len(enriched))
    return enriched
