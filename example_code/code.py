import yfinance as yf
import pandas as pd
import numpy as np
import pulp
from arch import arch_model

def optimize_portfolio(
    tickers,
    start_date,
    end_date,
    alpha_risk=0.05,
    beta_conf=0.95,
    gamma=0.2,
    n_scenarios=2000
):

    # =========================
    # 1. Data Acquisition
    # =========================
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()

    # -------------------------
    # Add synthetic "bad" BACK stock
    # -------------------------
    np.random.seed(42)
    last_price = 100
    n_days = data.shape[0]
    back_returns = np.random.normal(-0.2, 0.1, size=n_days)  # negative drift, high vol
    back_prices = last_price * np.cumprod(1 + back_returns)
    data['BACK'] = back_prices
    returns['BACK'] = data['BACK'].pct_change().dropna()
    
    tickers = tickers + ['BACK']
    n = len(tickers)

    # =========================
    # 2. Filtered Historical Simulation
    # =========================
    Z = []
    sigma = []

    for col in returns.columns:
        am = arch_model(returns[col] * 100, vol="Garch", p=1, q=1)
        res = am.fit(disp="off")
        sigma.append(res.conditional_volatility.iloc[-1] / 100)
        Z.append(res.std_resid.dropna().values)

    Z = np.vstack(Z).T
    sigma = np.array(sigma)

    idx = np.random.choice(len(Z), size=n_scenarios)
    scenarios = Z[idx] * sigma
    scenarios *= np.sqrt(10)  # 10-day horizon

    # -------------------------
    # Inject moderate stress scenario
    # -------------------------
    #stress = np.array([-0.03]*10 + [-0.15])  # 10 stocks moderate -3%, BACK -15%
   # scenarios = np.vstack([scenarios, stress])
    J = scenarios.shape[0]

    mu = scenarios.mean(axis=0)

    # =========================
    # 3. LP Model
    # =========================
    model = pulp.LpProblem("CVaR_Portfolio", pulp.LpMaximize)

    # Decision variables with small positive floor
    w = pulp.LpVariable.dicts("w", range(n), lowBound=1e-6)
    t = pulp.LpVariable("VaR")
    z = pulp.LpVariable.dicts("z", range(J), lowBound=0)

    # Objective: maximize expected return
    model += pulp.lpSum(mu[i] * w[i] for i in range(n))
    # -------------------------
    # 1. Budget constraint
    # -------------------------
    model += pulp.lpSum(w[i] for i in range(n)) == 1, "Budget"

    # -------------------------
    # 2. Diversification cap
    # -------------------------
    for i in range(n):
        model += w[i] <= gamma, f"Diversification_cap_{i}"

    # -------------------------
    # 3. CVaR constraint
    # -------------------------
    # Portfolio loss in scenario j: L_j = -sum(scenarios[j, i] * w[i])
    # Define z_j >= L_j - t, for all scenarios
    # Then: t + (1 / ((1 - beta_conf) * J)) * sum(z_j) <= alpha_risk_scaled

    portfolio_value = 1.0  # normalized
    alpha_risk_scaled = alpha_risk * portfolio_value

    # Loss constraints for each scenario
    for j in range(J):
        model += z[j] >= -pulp.lpSum(scenarios[j, i] * w[i] for i in range(n)) - t, f"Loss_scenario_{j}"

    # CVaR constraint
    model += t + (1 / ((1 - beta_conf) * J)) * pulp.lpSum(z[j] for j in range(J)) <= alpha_risk_scaled, "CVaR"

    # -------------------------
    # 4. Prevent negative expected return (optional, recommended for BACK)
    # -------------------------
    mu_real = returns.mean().values  # historical mean
    for i in range(n):
        if mu_real[i] < 0:
            model += w[i] == 0
    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    weights = np.array([w[i].value() for i in range(n)])
    weights_series = pd.Series(weights, index=tickers)

    # =========================
    # 4. Validate Model
    # =========================
    portfolio_returns = returns @ weights
    realized_cvar = -portfolio_returns.quantile(alpha_risk)

    print("Predicted CVaR constraint (alpha):", alpha_risk)
    print("Realized CVaR on portfolio:", realized_cvar)
    print("\nPortfolio Weights:")
    print(weights_series)

    return weights_series, realized_cvar

# =========================
# Example with 10 stocks + BACK
# =========================
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
          "NVDA", "META", "NFLX", "INTC", "IBM"]

weights, realized_cvar = optimize_portfolio(
    stocks,
    "2023-01-01",
    "2025-01-01"
)
