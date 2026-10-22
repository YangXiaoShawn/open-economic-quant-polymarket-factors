# Methodology & Artifact Elimination

Prediction-market prices represent bounded probabilities $p \in (0, 1)$. Standard equity backtest conventions introduce severe mechanical biases:

1. **No price clipping**: Clipping to $[0.03, 0.97]$ censors forward returns and manufactures artificial reversal.
2. **Probability-point returns ($\Delta p$) and tradeability filter**: Percentage returns explode near zero; formation days require $0.05 < p < 0.95$ with non-zero volume.
3. **Skip-day execution**: Entering at $t+1$ close separates the formation-day close from the return base.
4. **Settlement-inclusive exits**: Markets resolving within the holding window exit at their final traded price rather than being dropped.
5. **Executable entry**: Entry requires active trading on day $t+1$ within the tradeable band.
6. **Price-neutral residualization**: Forward $\Delta p$ is cross-sectionally residualized on $[1, p, p^2]$ to isolate alpha from boundary asymmetry.
