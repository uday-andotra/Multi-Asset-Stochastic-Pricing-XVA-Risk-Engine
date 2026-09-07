# Multi-Asset Stochastic Pricing & XVA Risk Engine

An enterprise-grade, portfolio-level quantitative risk and pricing engine built in Python to simulate multi-asset paths, bootstrap macroeconomic yield curves, and compute Counterparty Credit Risk (CCR) and Valuation Adjustments (XVA).

## 🚀 Key Features & Architecture

The engine is structured into a modular, high-performance architecture:

### Phase 1: High-Performance Simulation Engine (Stochastic Calculus & HPC)
*   **Geometric Brownian Motion (GBM):** Simulated using the Euler-Maruyama discretization method.
*   **Heston Model:** Incorporates stochastic volatility via the Cox-Ingersoll-Ross (CIR) variance process.
*   **Vasicek Model:** Simulates mean-reverting interest rate paths.
*   **Merton Jump-Diffusion:** Simulates market crashes and fat-tailed distributions using a Poisson jump process.
*   **Cholesky Decomposition:** Correlates multi-asset baskets and interest rates using real-world historical correlation matrices.
*   **GPU Acceleration:** Vectorized computations utilizing PyTorch/CuPy for high-performance Monte Carlo path generation.

### Phase 2: Dynamic Yield Curve Bootstrapper
*   **Zero-Coupon Rate Extraction:** Strips rates from simulated market instrument data.
*   **Dual-Curve Multi-Curve Framework:** Separates cash flow projection curves from OIS/SOFR discounting curves.
*   **Nelson-Siegel-Svensson (NSS):** Implements parametric macroeconomic curve fitting as an advanced alternative to basic cubic splines.

### Phase 3: Counterparty Credit Risk & XVA Module
*   **Exposure Profiles:** Computes cross-sectional Expected Exposure (EE) and 95% Potential Future Exposure (PFE).
*   **Credit Value Adjustment (CVA):** Integrates Probability of Default (PD) and Loss Given Default (LGD) to quantify counterparty credit risk.
*   **Wrong-Way Risk (WWR):** Dynamically correlates counterparty hazard rates with portfolio asset exposures.
*   **Funding Value Adjustment (FVA):** Calculates the ongoing cost of posting cash collateral.

### Interactive Dashboard Frontend
*   **Streamlit UI (`app.py`):** Provides a web-based interactive dashboard featuring real-time logging status containers, dynamic parameter configuration, and automated Matplotlib visualizations.

---

## 🛠️ Tech Stack
*   **Language:** Python
*   **Core Libraries:** NumPy, SciPy, PyTorch / CuPy (GPU Acceleration)
*   **Data & Visualization:** Matplotlib, `yfinance`
*   **Frontend/Dashboard:** Streamlit

---

## 📦 Project Structure
```text
├── core/
│   ├── monte_carlo.py       # Simulation engine (GBM, Heston, Vasicek, Jumps, Cholesky)
│   ├── yield_curve.py       # Bootstrapping & NSS curve fitting
│   └── xva_calculator.py    # CCR, EE, PFE, CVA, FVA, and Wrong-Way Risk
├── engine_backend.py        # Orchestration pipeline & optimizers
├── app.py                   # Streamlit interactive dashboard frontend
└── README.md
```

---

## 🚦 Getting Started & Installation

1. Clone the repository:
```bash
git clone [https://github.com/YOUR_USERNAME/Multi-Asset-Stochastic-Pricing-XVA-Risk-Engine.git](https://github.com/YOUR_USERNAME/Multi-Asset-Stochastic-Pricing-XVA-Risk-Engine.git)
cd Multi-Asset-Stochastic-Pricing-XVA-Risk-Engine

```


2. Install dependencies:
```bash
pip install numpy scipy torch matplotlib streamlit yfinance

```


3. Run the interactive Streamlit dashboard:
```bash
streamlit run app.py

```
