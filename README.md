# Multi-asset Monte Carlo and XVA study engine

Educational Python engine: simulate a small equity basket, build a discount curve,
and compute exposure and unilateral CVA / FVA on a toy option-style payoff.

It is a study desk. It is not a production CCR system, not a netting-set
valuer, and not investment advice.

This repository is a cleaned public snapshot of local work.

## What it computes

**Paths.** Correlated log-Euler steps. Asset Browns are built from a
**correlation** Cholesky. Instantaneous variance follows a CIR layer
(Heston-style) with asset–variance correlation fixed at −0.7. Optional
Merton jumps with a compensator. Crisis / calm scenarios scale vols and
push off-diagonal correlations toward 1 or toward 0. The stressed
covariance is repaired if a stress breaks positive-definiteness.

The CIR layer is a variance multiplier on unit Browns. It is not a
calibrated Heston surface.

**Curve.** Instruments come from `data/curve_instruments.csv`
(columns `curve,kind,maturity,rate`). Two named tapes can live in that
file: `discount` and `projection`. Each is stripped then optionally NSS-fitted.
XVA on this option payoff uses the **discount** tape only. The projection
tape is loaded so a later linear instrument can project floats. That is two
independent strips, not a full OIS/LIBOR dual-curve engine.

Optional Vasicek short-rate paths (`use_vasicek_discount=True`) replace
deterministic discount factors `DF(t) = exp(-y(t) * t)` with a pathwise
integral of the short rate `r`. Default is off (deterministic NSS).

**Book.** Simulated spots are turned into a long-only weighted portfolio
(`sum_i units_i * S_i`). Exposure is not marked name-by-name.

**Exposure.** For each path and time, mark a call or put on that portfolio
value and keep the positive part:

- `E(t) = max(V(t), 0)`
- `EE(t) = mean of E(t) across paths`
- `PFE95(t) = 95th percentile of E(t)`

Strike is `initial_value * strike_multiplier`. Baseline vs crash scenarios
switch call vs put.

**CVA (unilateral).** Constant recovery. Intensity can depend on exposure:

- `lambda(t) = lambda0 * exp(alpha * E(t) / S0)`
- `PD(t) = exp(-Lambda(t-dt)) - exp(-Lambda(t))`
- `CVA = mean over paths of sum_t LGD * E(t) * DF(t) * PD(t)`

`alpha = 0` is no wrong-way risk. `alpha > 0` is a reduced-form WWR knob,
not a structural model of the counterparty’s asset.

**FVA.** Funding spread times discounted positive exposure, discrete sum.
No collateral schedule, no CSA, no DVA.

**Allocator.** A cheap differential-evolution + SLSQP pass reweights the
book to cut a score of peak PFE + CVA + FVA. It is a demo knob, not a
desk optimiser.

## What it is not

- Calibrated Vasicek term structure (the OU paths are a discounting knob)
- Full dual-curve OIS vs projection with basis, collateral, and CSA
- KVA / regulatory capital
- Netting, margin, or close-out set
- Calibrated Heston or jump surface
- A bank XVA library

Default device is CPU. The simulator uses PyTorch and will take CUDA or
MPS if present. CuPy is not used.

## Run

```bash
git clone https://github.com/uday-andotra/Multi-Asset-Stochastic-Pricing-XVA-Risk-Engine.git
cd Multi-Asset-Stochastic-Pricing-XVA-Risk-Engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

```bash
PYTHONPATH=. pytest -q
```

Tests do not download prices.

## Layout

```
core/monte_carlo.py      # paths
core/yield_curve.py      # bootstrap + NSS
core/xva_calculator.py   # EE, PFE, CVA, FVA
engine_backend.py        # yfinance hist, weights, plots
app.py                   # Streamlit
tests/
```

## How to read a run

EE should sit below 95% PFE on every date. CVA should rise if you lift
λ₀, LGD, or α, and fall if you lift recovery. FVA should rise in the
funding spread. If a crisis correlation stress is on, PFE usually fattens
faster than EE.

The allocator can cut that score by concentrating the book. Treat the
weights as an illustration of the objective, not a recommendation.
