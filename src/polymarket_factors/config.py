"""Central configuration for Polymarket Pair Trading Strategy."""

import os
from pathlib import Path

# Load .env (if present) so API endpoints and secrets can be overridden without
# editing source code.  python-dotenv is a no-op when the file doesn't exist.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Polymarket API endpoints — override via .env if using a proxy
POLYMARKET_GAMMA_API = os.getenv("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com")
POLYMARKET_CLOB_API  = os.getenv("POLYMARKET_CLOB_API",  "https://clob.polymarket.com")

# Data fetch settings
DEFAULT_LOOKBACK_DAYS  = 365        # 1-year history window
BACKTEST_LOOKBACK_DAYS = 365        # days of history used for backtest
MIN_DATA_POINTS = 20                # minimum daily observations per market

# Pair selection thresholds
COINTEGRATION_PVALUE = 0.05         # ADF/Engle-Granger p-value threshold (tighter = fewer spurious pairs)
MIN_CORRELATION = 0.65              # minimum absolute Pearson correlation
MIN_HALF_LIFE_DAYS = 1              # prediction-market pairs naturally revert in 1-2d; accept them
MAX_HALF_LIFE_DAYS = 30             # max mean-reversion half-life accepted

# Signal thresholds (z-score of spread)
ENTRY_ZSCORE    = 2.0               # open a position when |z| exceeds this
EXIT_ZSCORE     = 0.5               # close position when |z| falls below this
STOP_LOSS_ZSCORE = 3.0              # emergency stop-loss

# Position sizing
POSITION_SIZE_USD = 100             # notional per leg (USD)
MAX_OPEN_PAIRS    = 5               # maximum concurrent pair positions

# Backtesting parameters
INITIAL_CAPITAL   = 10_000          # USD
TRANSACTION_COST  = 0.015           # 1.5% PER-SIDE cost (spread + slippage); strategies apply
                                    # it at open AND close → 3% round-trip per leg

# Rolling window for z-score calculation
ZSCORE_WINDOW = 20                  # days

# Walk-forward window sizes
TRAIN_DAYS = 45                     # training window for pair selection
STEP_DAYS  = 30                     # slide step size (pair trading)
PAIR_STEP_DAYS = 45                 # longer test windows for pair trades (less EOD force-close)

# Minimum market liquidity filter
MIN_DAILY_VOLUME_USD      = 500     # min avg daily volume per market (USD); unexecutable below this

# Backtest-specific fetch settings
BACKTEST_MIN_AGE_DAYS     = 14      # min market age for backtest inclusion (was 365//4 = 91)
BACKTEST_N_MARKETS        = 2000    # markets to request from Gamma API for backtest
BACKTEST_MAX_MARKETS      = 800     # markets to include in backtest price matrix

# Price realism filters
RESOLUTION_BUFFER_DAYS    = 7       # exclude final N days before market resolution date
PRICE_CLIP_HIGH           = 0.97    # clip prices above this (near-resolution convergence)
PRICE_CLIP_LOW            = 0.03    # clip prices below this
