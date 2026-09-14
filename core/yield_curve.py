"""File-driven strips: deposits then par swaps, plus an optional NSS fit.

Two named tapes (discount, projection) can live in one CSV.
This payoff only *uses* the discount strip. The projection strip is stored
so a later linear instrument can project floats. It is not a full
post-2008 OIS/LIBOR dual-curve engine.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class YieldCurveBootstrapper:
    def __init__(self):
        self.maturities = []
        self.discount_factors = []
        self.zero_rates = []

    def add_cash_rate(self, maturity, rate):
        df = 1.0 / (1.0 + rate * maturity)
        zero_rate = -np.log(df) / maturity
        self.maturities.append(maturity)
        self.discount_factors.append(df)
        self.zero_rates.append(zero_rate)

    def add_swap_rate(self, maturity, swap_rate, payment_freq=1.0):
        dt = payment_freq
        payment_dates = np.arange(dt, maturity + dt, dt)
        sum_discount_factors = 0.0

        for t in payment_dates[:-1]:
            interp_zero = np.interp(t, self.maturities, self.zero_rates)
            sum_discount_factors += np.exp(-interp_zero * t)

        df_N = (1.0 - swap_rate * dt * sum_discount_factors) / (1.0 + swap_rate * dt)
        zero_rate = -np.log(df_N) / maturity
        self.maturities.append(maturity)
        self.discount_factors.append(df_N)
        self.zero_rates.append(zero_rate)

    def get_curve(self):
        return np.array(self.maturities), np.array(self.zero_rates)


class NelsonSiegelSvensson:
    def __init__(self):
        self.params = None

    def nss_yield(self, t, beta0, beta1, beta2, beta3, lambda1, lambda2):
        t = np.where(t == 0, 1e-8, t)
        term1 = (1 - np.exp(-t / lambda1)) / (t / lambda1)
        term2 = term1 - np.exp(-t / lambda1)
        term3 = ((1 - np.exp(-t / lambda2)) / (t / lambda2)) - np.exp(-t / lambda2)
        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

    def objective_function(self, params, t_market, y_market):
        beta0, beta1, beta2, beta3, lambda1, lambda2 = params
        if lambda1 <= 0 or lambda2 <= 0 or beta0 <= 0:
            return 1e10
        y_model = self.nss_yield(t_market, beta0, beta1, beta2, beta3, lambda1, lambda2)
        return np.sum((y_model - y_market) ** 2)

    def fit(self, t_market, y_market):
        initial_guess = [np.mean(y_market), -0.01, 0.01, 0.01, 1.0, 1.0]
        result = minimize(
            self.objective_function,
            initial_guess,
            args=(t_market, y_market),
            method="Nelder-Mead",
            options={"maxiter": 5000},
        )
        if result.success:
            self.params = result.x

    def get_curve(self, t_array):
        if self.params is None:
            raise ValueError("Model must be fitted first.")
        return self.nss_yield(t_array, *self.params)


def load_instrument_tape(path, curve_name="discount"):
    """CSV columns: curve, kind, maturity, rate. kind in {cash, swap}."""
    df = pd.read_csv(path)
    df = df.loc[df["curve"].str.lower() == curve_name.lower()].copy()
    df = df.sort_values("maturity")
    boot = YieldCurveBootstrapper()
    for row in df.itertuples(index=False):
        kind = str(row.kind).lower()
        if kind == "cash":
            boot.add_cash_rate(float(row.maturity), float(row.rate))
        elif kind == "swap":
            boot.add_swap_rate(float(row.maturity), float(row.rate))
        else:
            raise ValueError(f"unknown instrument kind: {row.kind}")
    return boot


def fit_nss(bootstrapper):
    t, y = bootstrapper.get_curve()
    nss = NelsonSiegelSvensson()
    nss.fit(t, y)
    return nss, t, y


def default_curve_csv():
    return Path(__file__).resolve().parents[1] / "data" / "curve_instruments.csv"
