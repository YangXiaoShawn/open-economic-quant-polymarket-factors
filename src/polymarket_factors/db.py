import os
from pathlib import Path

import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET threads=4")

_HERE = Path(__file__).resolve().parent
# Local copies of the two public datasets (see README.md, "Data"); override with env vars.
USERS_BASE = os.getenv("POLYMARKET_USERS_DIR", str(_HERE / "data" / "polymarket-users")).replace("\\", "/")
V1_BASE    = os.getenv("POLYMARKET_V1_DIR", str(_HERE / "data" / "Polymarket-v1")).replace("\\", "/")


def query(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def _fmt_ids(ids: list) -> str:
    """Format an id list for SQL IN (): numbers bare, strings single-quoted."""
    return ",".join(
        str(i) if isinstance(i, (int, float)) else "'" + str(i).replace("'", "''") + "'"
        for i in ids
    )


# ── polymarket-users flat tables (safe to load whole) ────────

def get_events() -> pd.DataFrame:
    return query(f"SELECT * FROM '{USERS_BASE}/events.parquet'")

def get_markets(columns: str = "*") -> pd.DataFrame:
    return query(f"SELECT {columns} FROM '{USERS_BASE}/markets.parquet'")

def get_predictions() -> pd.DataFrame:
    return query(f"SELECT * FROM '{USERS_BASE}/predictions.parquet'")

def get_ohlcv_1d(market_ids: list | None = None) -> pd.DataFrame:
    where = f"WHERE market_id IN ({_fmt_ids(market_ids)})" if market_ids else ""
    return query(f"SELECT * FROM '{USERS_BASE}/ohlcv_1d.parquet' {where}")

def get_pnl_change_monthly() -> pd.DataFrame:
    return query(f"SELECT * FROM '{USERS_BASE}/pnl_change_monthly.parquet'")

def get_user_pnl_summary(columns: str = "*") -> pd.DataFrame:
    """59 cols, 2.48M rows — use columns= to limit memory."""
    return query(f"SELECT {columns} FROM '{USERS_BASE}/user_pnl_summary.parquet'")

def get_user_features(columns: str = "*") -> pd.DataFrame:
    """87 cols, 2.48M rows — use columns= to limit memory."""
    return query(f"SELECT {columns} FROM '{USERS_BASE}/user_features.parquet'")


# ── polymarket-users partitioned tables (must filter by date) ─

def get_trades(start: str, end: str, columns: str = "*") -> pd.DataFrame:
    """
    Load trades for a date range.
    start/end: 'YYYY-MM-DD'  (end is exclusive)
    columns: comma-separated column names, or '*'

    Example:
        get_trades('2024-01-01', '2024-04-01', 'trade_id,timestamp,price,quantity,category')
    """
    return query(f"""
        SELECT {columns}
        FROM read_parquet('{USERS_BASE}/trades/**/*.parquet', hive_partitioning=true)
        WHERE "timestamp" >= '{start}' AND "timestamp" < '{end}'
    """)

def get_pnl_daily(start: str, end: str) -> pd.DataFrame:
    """Load daily PnL snapshots for a date range. start/end: 'YYYY-MM-DD'"""
    return query(f"""
        SELECT *
        FROM read_parquet('{USERS_BASE}/pnl_daily/**/*.parquet', hive_partitioning=true)
        WHERE day >= '{start}' AND day < '{end}'
    """)

def get_pnl_daily_resolved(start: str, end: str) -> pd.DataFrame:
    return query(f"""
        SELECT *
        FROM read_parquet('{USERS_BASE}/pnl_daily_resolved/**/*.parquet', hive_partitioning=true)
        WHERE day >= '{start}' AND day < '{end}'
    """)

def get_pnl_category_daily(start: str, end: str, category: str | None = None) -> pd.DataFrame:
    where = f"WHERE day >= '{start}' AND day < '{end}'"
    if category:
        where += f" AND category = '{category}'"
    return query(f"""
        SELECT *
        FROM read_parquet('{USERS_BASE}/pnl_category_daily/**/*.parquet', hive_partitioning=true)
        {where}
    """)

def get_pnl_change_daily(start: str, end: str) -> pd.DataFrame:
    return query(f"""
        SELECT *
        FROM read_parquet('{USERS_BASE}/pnl_change_daily/**/*.parquet', hive_partitioning=true)
        WHERE day >= '{start}' AND day < '{end}'
    """)

def get_ohlcv_1h(start: str, end: str, market_ids: list | None = None) -> pd.DataFrame:
    where = f"WHERE timestamp >= '{start}' AND timestamp < '{end}'"
    if market_ids:
        where += f" AND market_id IN ({_fmt_ids(market_ids)})"
    return query(f"""
        SELECT *
        FROM read_parquet('{USERS_BASE}/ohlcv_1h/**/*.parquet', hive_partitioning=true)
        {where}
    """)

def get_ohlcv_5m(start: str, end: str, market_ids: list | None = None) -> pd.DataFrame:
    where = f"WHERE timestamp >= '{start}' AND timestamp < '{end}'"
    if market_ids:
        where += f" AND market_id IN ({_fmt_ids(market_ids)})"
    return query(f"""
        SELECT *
        FROM read_parquet('{USERS_BASE}/ohlcv_5m/**/*.parquet', hive_partitioning=true)
        {where}
    """)


# ── Polymarket-v1 ─────────────────────────────────────────────

def get_ctf(table: str) -> pd.DataFrame:
    """table: merges | preparations | redemptions | resolutions | splits"""
    return query(f"SELECT * FROM '{V1_BASE}/CTF/{table}.parquet'")

def get_order_filled(start_month: str, end_month: str) -> pd.DataFrame:
    """
    Load OrderFilled for a range of months.
    start_month / end_month: 'YYYY_MM'  (both inclusive)

    Example:
        get_order_filled('2024_01', '2024_03')
    """
    return query(f"""
        SELECT *, filename
        FROM read_parquet('{V1_BASE}/OrderFilled/*.parquet', filename=true)
        WHERE regexp_extract(filename, '[0-9]{{4}}_[0-9]{{2}}') BETWEEN '{start_month}' AND '{end_month}'
    """)

def get_daily_aligned(start: str, end: str) -> pd.DataFrame:
    """
    Load daily_aligned for a date range.
    start/end: 'YYYY-MM-DD'  (end is exclusive)
    Files are named YYYY-MM-DD.parquet
    """
    return query(f"""
        SELECT *
        FROM read_parquet('{V1_BASE}/daily_aligned/*.parquet', filename=true)
        WHERE regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}') >= '{start}'
          AND regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}') < '{end}'
    """)
