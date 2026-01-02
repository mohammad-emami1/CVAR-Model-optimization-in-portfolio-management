This is a comprehensive and technically sound README for a financial engineering project. I have corrected the LaTeX rendering, filled in the missing section on CVaR sign conventions, and polished the formatting to ensure it is clear and professional.

CVaR Portfolio Optimization with Filtered Historical Simulation (FHS)



1. Overview
This project implements a Conditional Value at Risk (CVaR) portfolio optimization using Filtered Historical Simulation (FHS) for scenario generation. By combining GARCH volatility modeling with historical standardized residuals, the model creates realistic future return scenarios that capture current market regimes.

Key Features
Volatility Modeling: GARCH(1,1) filters to obtain standardized residuals.

Scenario Generation: FHS preserves cross-sectional asset dependence and empirical tail behavior.

Optimization: Linear Programming (LP) via the Rockafellar & Uryasev formulation.

Risk Management: Constraints for diversification caps, budget, and risk budgets (CVaR).

Backtesting Utility: Includes a synthetic "BACK" asset to test the optimizer's ability to avoid high-risk/low-return instruments.

2. Mathematical Formulation
Filtered Historical Simulation (FHS)
FHS uses standardized residuals from a GARCH model and rescales them by the most recent volatility estimate to produce scenarios consistent with current conditions.

GARCH(1,1) Fit: For each asset i, we model returns as:

r 
i,t
​
 =μ 
i
​
 +σ 
i,t
​
 ε 
i,t
​
 ,ε 
i,t
​
 ∼i.i.d.(0,1)
σ 
i,t
2
​
 =ω 
i
​
 +α 
i
​
 ε 
i,t−1
2
​
 +β 
i
​
 σ 
i,t−1
2
​
 
Standardized Residuals: We extract  
ε
^
  
i,t
​
 =(r 
i,t
​
 − 
μ
^
​
  
i
​
 )/ 
σ
^
  
i,t
​
 .

Scenario Generation: We sample a random historical time index τ 
j
​
  and apply it across all assets to maintain the correlation structure:

r
~
  
j,i
​
 = 
σ
^
  
i,T
​
 ⋅ 
ε
^
  
i,τ 
j
​
 
​
 ⋅ 
H

​
 
where H is the forecast horizon (e.g., 10 days).

CVaR Linear Program
Following the Rockafellar & Uryasev approach, we linearize the CVaR objective by introducing auxiliary variables t (representing VaR) and z 
j
​
  (representing excess loss).

Objective: Maximize expected portfolio return:

w∈R 
n
 
max
​
  
i=1
∑
n
​
 μ 
i
​
 w 
i
​
 
Subject to:

CVaR Constraint:

t+ 
(1−β)J
1
​
  
j=1
∑
J
​
 z 
j
​
 ≤α
Excess Loss Constraints:

z 
j
​
 ≥− 
i=1
∑
n
​
  
r
~
  
j,i
​
 w 
i
​
 −t,z 
j
​
 ≥0
Budget & Diversification:

∑w 
i
​
 =1,0≤w 
i
​
 ≤γ
3. Implementation Details
CVaR Sign and Units
In this implementation, Loss (L) is defined as negative return (L=−R).

An α value of 0.05 means you are limiting the average loss in the worst 5% of cases (the tail) to 5% of the portfolio value.

If the optimization returns a t (VaR) of 0.03, it implies that in 95% of scenarios, the loss will not exceed 3%, but the average loss beyond that threshold is capped at 5% by the α constraint.

Data & Modeling Rationale
yfinance: Used for seamless historical price fetching.

Synthetic "BACK" Asset: Injected with a mean return of -0.05% and high volatility to ensure the optimizer correctly identifies and avoids "wealth destroyers."

Scaling: Returns are scaled by 100 for the GARCH fit to improve numerical stability in the arch library.

Solver: Uses PuLP with the CBC solver. For large-scale universes (>500 assets), a commercial solver like Gurobi is recommended.

4. How to Run
Requirements
Bash

pip install numpy pandas yfinance pulp arch
Example Usage
Python

from cvar_optimizer import optimize_portfolio

stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA","NVDA", "META", "NFLX", "INTC", "IBM"]

# alpha_risk=0.05 implies a 5% CVaR limit
weights, realized_cvar = optimize_portfolio(
    stocks,
    start_date="2023-01-01",
    end_date="2025-01-01",
    alpha_risk=0.05, 
    beta_conf=0.95,
    gamma=0.25,
    n_scenarios=2000
)

print(f"Optimized Weights:\n{weights}")
print(f"Realized CVaR: {realized_cvar:.4f}")
5. Assumptions & Limitations
GARCH Adequacy: Assumes volatility clustering is the primary driver of risk; does not account for sudden regime shifts (e.g., "black swan" events not present in the historical residuals).

Square-Root Scaling: Scaling daily volatility to H days via  
H

​
  assumes returns are independent over time (no autocorrelation).

Historical Dependence: While FHS captures historical correlations, it cannot predict "correlation breakdown" during extreme market stress.

Static Optimization: This is a single-period model and does not account for transaction costs or multi-period rebalancing logic.

Would you like me to generate a Python backtesting script to compare this CVaR-FHS approach against a standard Mean-Variance optimization?
