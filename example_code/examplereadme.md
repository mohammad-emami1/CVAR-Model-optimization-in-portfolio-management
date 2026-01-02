# CVaR Portfolio Optimization with Filtered Historical Simulation (FHS)

## 1. Overview
This project implements a Conditional Value at Risk (CVaR) portfolio optimization using **Filtered Historical Simulation (FHS)** for scenario generation. By combining GARCH volatility modeling with historical standardized residuals, the model creates realistic future return scenarios that capture current market regimes.

**Key Features**
- **Volatility Modeling:** GARCH(1,1) filters to obtain standardized residuals.
- **Scenario Generation:** FHS preserves cross-sectional asset dependence and empirical tail behavior.
- **Optimization:** Linear Programming (LP) via the Rockafellar & Uryasev formulation.
- **Risk Management:** Constraints for diversification caps, budget, and risk budgets (CVaR).
- **Backtesting Utility:** Includes a synthetic "BACK" asset to test the optimizer's ability to avoid high-risk/low-return instruments.

## 2. Mathematical Formulation

### Filtered Historical Simulation (FHS)
FHS uses standardized residuals from a GARCH model and rescales them by the most recent volatility estimate to produce scenarios consistent with current conditions.

**GARCH(1,1) Fit:** For each asset $i$, we model returns as:

$$
r_{i,t} = \mu_i + \sigma_{i,t} \varepsilon_{i,t}, \quad \varepsilon_{i,t} \sim \text{i.i.d. } N(0,1)
$$

$$
\sigma_{i,t}^2 = \omega_i + \alpha_i \varepsilon_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2
$$

**Standardized Residuals:** 

$$
\hat{\varepsilon}_{i,t} = \frac{r_{i,t} - \hat{\mu}_i}{\hat{\sigma}_{i,t}}
$$

**Scenario Generation:** We sample a random historical time index $\tau_j$ and apply it across all assets to maintain the correlation structure:

$$
\tilde{r}_{j,i} = \hat{\sigma}_{i,T} \cdot \hat{\varepsilon}_{i,\tau_j} \cdot H
$$

where $H$ is the forecast horizon (e.g., 10 days).

### CVaR Linear Program
Following the Rockafellar & Uryasev approach, we linearize the CVaR objective by introducing auxiliary variables $t$ (VaR) and $z_j$ (excess loss).

**Objective: Maximize expected portfolio return**

$$
\max_{w \in \mathbb{R}^n} \sum_{i=1}^n \mu_i w_i
$$

**Subject to:**

**CVaR Constraint:**

$$
t + \frac{1}{(1-\beta) J} \sum_{j=1}^J z_j \le \alpha
$$

**Excess Loss Constraints:**

$$
z_j \ge - \sum_{i=1}^n \tilde{r}_{j,i} w_i, \quad z_j \ge 0
$$

**Budget & Diversification:**

$$
\sum_i w_i = 1, \quad 0 \le w_i \le \gamma
$$

## 3. Implementation Details

### CVaR Sign and Units
- **Loss Definition:** $L = -R$, negative return.
- **Interpretation:** $\alpha = 0.05$ limits the average loss in the worst 5% of cases to 5% of portfolio value.
- **Example:** If optimization returns $t = 0.03$, in 95% of scenarios, loss will not exceed 3%, and the average loss beyond that is capped at 5%.

### Data & Modeling Rationale
- **yfinance:** Fetches historical prices.
- **Synthetic "BACK" Asset:** Mean return -0.05% and high volatility to test optimizer robustness.
- **Scaling:** Returns multiplied by 100 for numerical stability in the `arch` library.
- **Solver:** PuLP with CBC. Large universes (>500 assets) may require commercial solvers (e.g., Gurobi).

## 4. How to Run

### Requirements
```bash
pip install numpy pandas yfinance pulp arch
```
Example Usage
```
python

from cvar_optimizer import optimize_portfolio

# Define your universe of assets
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "IBM"]

# Optimize portfolio with CVaR-FHS
# alpha_risk = 0.05 corresponds to limiting the average loss in the worst 5% of scenarios
weights, realized_cvar = optimize_portfolio(
    assets=stocks,
    start_date="2023-01-01",
    end_date="2025-01-01",
    alpha_risk=0.05,
    beta_conf=0.95,
    gamma=0.25,
    n_scenarios=2000
)

print("Optimized Portfolio Weights:")
print(weights)
print(f"Realized CVaR over scenarios: {realized_cvar:.4f}")
```
### Notes on Parameters

- **alpha_risk**: Tail risk level for the CVaR constraint. For example, `0.05` limits the average loss in the worst 5% of scenarios.  
- **beta_conf**: Confidence level for VaR. For example, `0.95` corresponds to a 95% confidence that losses will not exceed the VaR threshold.  
- **gamma**: Maximum weight allocation per asset to enforce diversification.  
- **n_scenarios**: Number of Filtered Historical Simulation (FHS) scenarios generated for the optimization.

## 5. Assumptions & Limitations

- **GARCH Assumptions**: Assumes volatility clustering is the primary driver of risk and may not capture extreme regime shifts not present in historical residuals.  
- **Square-Root Scaling**: Scaling daily volatility to a multi-day horizon assumes returns are independent and not autocorrelated.  
- **Historical Dependence**: FHS preserves historical correlations but cannot predict correlation breakdowns during market crises.  
- **Static Optimization**: This is a single-period optimization; it does not account for transaction costs or multi-period rebalancing.  
- **Solver Considerations**: The default CBC solver works well for small to medium asset universes (<500 assets). For larger universes or faster performance, a commercial solver like Gurobi is recommended.
