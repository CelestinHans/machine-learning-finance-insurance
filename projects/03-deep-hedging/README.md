# Project 3: Deep Hedging (Black-Scholes and Heston)

## Summary
This notebook implements a **deep hedging** framework to hedge a European call option on simulated paths in two settings:
- a complete market (**Black-Scholes**) with comparison to the analytical delta hedge;
- an incomplete market (**Heston**) with training under a **CVaR** risk objective.

The goal is to learn dynamic hedging strategies and analyze the resulting hedging loss distributions.

## Key concepts

### Deep hedging (general idea)
The hedging strategy is parameterized by a neural model that decides, at each rebalancing date, how much of the underlying to hold given the available information.  
The model is trained end-to-end to minimize a risk criterion on the terminal hedging loss.

### Complete vs incomplete markets
- In **Black-Scholes**, replication is theoretically achievable via the analytical **delta hedge**.
- In **Heston**, the market is incomplete (stochastic volatility), so we directly optimize a risk criterion (here **CVaR**) instead of expecting perfect replication.

## What this notebook covers

### 1) Black-Scholes
- Exact simulation of price paths on a discrete time grid.
- Deep hedger construction (one sub-network per rebalancing date).
- Training with a squared terminal hedging error objective.
- Grid search over architecture choices (depth, activation, hidden dimension, epochs).
- Out-of-sample evaluation via histograms, mean, and standard deviation of losses.
- Analytical benchmark using the Black-Scholes delta strategy.
- Visual comparison of learned strategies vs analytical strategy as a function of spot.

### 2) Heston (incomplete market)
- Simulation of stochastic variance (non-central chi-square type transitions) and price paths.
- Deep hedging with joint price/variance information.
- **CVaR** optimization with a jointly learned auxiliary threshold variable.
- Training at two risk levels: **alpha = 0.5** and **alpha = 0.99**.
- Comparison of hedging losses and implied hedging price across alpha values.

## Main results

### Black-Scholes
- Best configuration by test loss dispersion: `hidden_dim=32`, `Tanh` activation, `Deep` model, `100` epochs.
- Deep hedging losses (selected model): mean close to `0` and standard deviation `~0.0091`.
- Analytical delta benchmark: mean `-9.3e-05`, standard deviation `~0.0089`.

Interpretation: in the Black-Scholes setting, deep hedging learns a policy very close to the analytical hedge, with slightly higher but comparable loss dispersion.

### Heston
- Estimated (CVaR-based) price:
  - `alpha = 0.5`: `p ≈ 0.0923`
  - `alpha = 0.99`: `p ≈ 0.1417`
- Empirical hedging loss statistics:
  - `alpha = 0.5`: mean `0.0007`, std `0.0162`
  - `alpha = 0.99`: mean `-0.0516`, std `0.0275`

Interpretation: increasing `alpha` makes the objective more sensitive to tail scenarios, which strongly changes the price/risk trade-off and the loss distribution.

## Conclusion
This project shows that deep hedging:
- effectively reproduces delta-hedging behavior in Black-Scholes;
- remains useful in incomplete markets (Heston), where CVaR provides explicit control over tail-risk aversion.

In practice, `alpha` is a key control parameter: it determines how much protection is sought against extreme losses and directly impacts the learned hedging price.
