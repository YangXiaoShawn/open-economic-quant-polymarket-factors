# Prediction-Market Factor Research: Polymarket

**Question.** Do behavioral biases (overreaction, liquidity discounts, longshot
preference) leave predictable cross-sectional structure in prediction-market prices, once
the mechanical artifacts that dominate prediction-market backtests are removed?

**Answer.** Yes, modestly. Five artifacts inflate a 30-day momentum IC eightfold
(−0.44 → −0.054). After they are fixed, short-horizon **reversal**, **volatility** and
**illiquidity** survive out of sample and beyond price-level controls (price-neutral
|IC| 0.032–0.072 on the held-out second half). Capacity is tiny: about $100–200 per
position.

Evidence grade: **backtest** (association). The full report is
[`REPORT_EN.md`](REPORT_EN.md) (English) and [`REPORT.md`](REPORT.md) (Chinese),
dated 2026-06-11; figures are in `report_figs_en/` and `report_figs/`.

## Data

The 4,000-market × 574-day panel (2024-09-01 to 2026-03-28, including resolved markets,
so there is no survivorship bias) is built from two public academic datasets, both
released under CC BY 4.0. Please cite them if you reuse this work:

- **Polymarket Users** — Akey, P., Grégoire, V., Harvie, N. and Martineau, C. (2026),
  *Who Wins and Who Loses in Prediction Markets? Evidence from Polymarket*, working
  paper ([SSRN 6443103](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6443103));
  data at Hugging Face `vgregoire/polymarket-users`. Markets, trades and daily OHLCV
  used for the factor study.
- **Polymarket-v1 Database** — Qin, B. and Yang, R. (2026),
  [arXiv:2606.04217](https://arxiv.org/abs/2606.04217). On-chain `OrderFilled` and
  CTF lifecycle layers, read through `db.py`.

There are 107 evaluation periods with a 5-day rebalance and an average tradeable
cross-section of 288 markets. An earlier 125-market sample pulled from Polymarket's
public Gamma and CLOB APIs (`data_fetcher.py`, `resolved_data.py`) is where the first
factor combination was selected; it failed to replicate on the large panel (report
Section 3.3). No data are redistributed here: download the two datasets and point
`POLYMARKET_USERS_DIR` and `POLYMARKET_V1_DIR` at them (defaults: `data/polymarket-users`
and `data/Polymarket-v1`).

## Why prediction markets need different conventions

- The return unit is **Δp in probability points**, not a percentage: one cent is +20% at
  p = 0.05 and +2% at p = 0.50.
- **Shorting YES means buying NO** at cost 1 − p; using p as the short-leg denominator
  inflates short returns on low-priced markets by up to 19×.
- Prices are bounded in [0, 1], so any factor correlated with the price level rides on
  mechanical asymmetry. Every IC is therefore also reported **price-neutral**, with
  forward Δp residualized on [1, p, p²].

## The five artifacts

| Fix | Artifact it removes |
|---|---|
| 1. No price clipping | clipping to [0.03, 0.97] censors forward returns and creates mechanical "reversal" |
| 2. Δp returns + tradeability filter | percentage changes explode on low bases; untradeable quotes enter the sample |
| 3. Skip a day before entry | the formation-day close sits in both the score and the return base |
| 4. Settlement-inclusive exits | requiring a later quote silently drops early-settled losers |
| 5. Executable entry | stale forward-filled quotes allow fills at pre-news prices |

As the fixes are applied in turn, the 30-day momentum IC moves
−0.44 → −0.34 → −0.105 → −0.096 → **−0.054** (final; report Section 2).

## Out-of-sample factor results

Factor signs are set on the first half (2024-10 to 2025-06); all statistics below are
from the second half (2025-06-28 to 2026-03-20, 54 periods).

| Factor | Raw IC | Price-neutral IC | Verdict |
|---|---|---|---|
| PM-VOL (volatility, reversed) | −0.067 (t = −7.0) | −0.069 (t = −6.7) | genuine |
| PM-LIQ (illiquidity, reversed) | −0.060 (t = −6.6) | −0.072 (t = −8.2) | genuine |
| PM-MOM (30-day momentum, reversed) | −0.054 (t = −5.5) | −0.032 (t = −3.4) | genuine reversal |
| PM-TTR (time to resolution, reversed) | −0.007 (n.s.) | −0.028 (t = −3.5) | weak but robust |
| PM-SPREAD (spread proxy) | +0.047 (t = 5.7) | +0.026 (t = 2.7) | mostly price level |
| PM-EXTR (price extremity) | +0.044 (t = 4.9) | +0.021 (t = 2.9) | mostly price level |
| PM-DRIFT (directional drift) | −0.014 (n.s.) | −0.001 (n.s.) | artifact |

Fama–MacBeth regressions with p and p² controls (Newey–West, 6 lags, 107 periods)
agree: MOM −140 bp per 5 days per σ (t = −8.3), VOL −209 bp (t = −6.4), LIQ −67 bp
(t = −3.5), TTR −102 bp (t = −4.7); EXTR, SPREAD and DRIFT collapse to zero.

- Reversal is concentrated in **Sports** (|IC| 0.060, t = 3.7) and **Politics**
  (0.051, t = 3.4) and absent in Crypto and Finance markets.
- An exhaustive search over 35 three-factor combinations ranks **MOM + LIQ + TTR**
  first: price-neutral IC 0.063 (t = 8.4), long–short spread +4.31 probability points
  per 5 days (t = 6.0). Factor signs come from the first half, but the ranking itself is
  computed on the second half (`threefactor_search.py`), so this is the best of 35 on the
  evaluation sample, **not an out-of-sample result**.
- **Negative result:** within multi-leg events (690 markets, 75 events, 175 days),
  sibling-market momentum has no next-24-hour predictability (IC 0.004, t = 0.3);
  sibling information is priced within the hour.

## Costs and strategies

Effective spreads are rebuilt from actual fills: the median is **0.4¢** across 2,058
active markets (p75 = 1¢), charged as s/p on longs and s/(1 − p) on shorts.

On the 107 periods with these costs, the reversal **short leg** (buy NO on the largest
30-day gainers, price band [0.15, 0.85], Sports and Politics) nets +20.6% per period
(t = 9.1, 84% win rate); the long leg earns +6.2% (t = 2.8).

The report also combines this short leg 50/50 with a basket-deviation arbitrage strategy
(Sharpe 6.30, maximum drawdown −4.0% over 574 days). **Read that number as in-sample:**
the 50/50 weight was chosen after the fact and has not been validated out of sample,
daily-close fills ignore intraday slippage, the arbitrage leg's profit factor (22.5)
depends on a few winners, and candidate markets trade only $1.6k–12k a day, so results
hold at roughly **$100–200 per position**.

## Reproduce

```bash
pip install -r requirements.txt
python expanded_factor_research.py           # factor study
python threefactor_search.py                 # 35-combination search
python momrev_strategy.py                    # short-winners strategy
python lead_lag_hourly.py                    # hourly lead-lag test
python make_report_figs_en.py                # figures for REPORT_EN.md
python -m pytest tests/ -v
```

`app.py` is a local FastAPI front end (`python -m uvicorn app:app --port 8000`) with the
live scanner, backtest, factor and three-factor pages described in
[`README_app.md`](README_app.md). Copy `.env.example` to `.env` to override the default
public API endpoints; no credentials are needed for public market data.

Research period: June 2026.
