# Status

Last updated: 2026-09-30

## Current evidence

- **Evidence grade**: Backtest association (`PUBLIC_RESEARCH_SNAPSHOT`).
- **Panel**: 4,000 Polymarket markets across 574 days (2024-09-01 to 2026-03-28, including resolved markets), 107 evaluation periods (5-day rebalance), held-out second half of 54 periods (2025-06-28 to 2026-03-20).
- **Artifact funnel**: Five sequential fixes reduce apparent 30-day momentum IC from -0.440 to -0.054 (price-neutral -0.032, t = -3.4).
- **Out-of-sample factors**: Illiquidity (`PM-LIQ`, price-neutral IC -0.072, t = -8.2), volatility (`PM-VOL`, -0.069, t = -6.7), reversal (`PM-MOM`, -0.032, t = -3.4), and time-to-resolution (`PM-TTR`, -0.028, t = -3.5) survive price-level controls.
- **Negative findings**: Sibling-market 24h lead-lag has no predictability (IC 0.004, t = 0.3); pair trading fails out of sample.
- **Limitations**: Capacity is approximately $100-$200 per position; daily-close fills omit intraday slippage; the 50/50 combined strategy weight was chosen in sample.
