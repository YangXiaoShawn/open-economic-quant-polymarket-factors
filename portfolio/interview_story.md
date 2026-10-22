# Interview Story: Prediction-Market Factor Research

- **Question**: Which cross-sectional signals in prediction markets survive after removing bounded-price, stale-quote, and early-settlement artifacts?
- **Diagnostic**: Five successive artifact fixes reduce apparent 30-day momentum IC from -0.440 to -0.054 (price-neutral -0.032, t = -3.4).
- **Finding**: Short-horizon reversal (PM-MOM), volatility (PM-VOL, price-neutral IC -0.069), and illiquidity (PM-LIQ, price-neutral IC -0.072) survive out of sample on 4,000 Polymarket markets.
- **Boundary**: Backtest association only; position capacity is roughly $100-$200 per market.
