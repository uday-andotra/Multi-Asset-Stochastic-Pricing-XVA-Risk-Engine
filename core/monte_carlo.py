"""Correlated log-Euler paths with a CIR variance layer and optional jumps.

Asset Browns come from a correlation Cholesky. Instantaneous variance
is the only source of diffusion scale. That avoids multiplying a
covariance-Cholesky increment by sqrt(v) again.
"""

import torch
import numpy as np


def simulate_advanced_asset_paths_gpu(
    tickers, S0, mu, cov_matrix, scenario, T, dt, num_paths, device="cpu"
):
    num_assets = len(tickers)
    num_steps = int(T / dt)

    cov_np = np.array(cov_matrix, dtype=np.float64)
    vols = np.sqrt(np.clip(np.diagonal(cov_np), a_min=1e-8, a_max=None))

    outer_vols = np.outer(vols, vols)
    corr_np = np.clip(cov_np / outer_vols, -1.0, 1.0)
    np.fill_diagonal(corr_np, 1.0)

    stressed_vols = vols * scenario["vol_modifier"]

    corr_stress = scenario["correlation_stress"]
    if corr_stress != 0.0:
        if corr_stress > 0:
            corr_np = corr_np + corr_stress * (1.0 - corr_np)
        else:
            corr_np = corr_np * (1.0 + corr_stress)
        np.fill_diagonal(corr_np, 1.0)
        corr_np = np.clip(corr_np, -1.0, 1.0)
        np.fill_diagonal(corr_np, 1.0)

    min_eig = np.min(np.real(np.linalg.eigvalsh(corr_np)))
    if min_eig < 1e-8:
        corr_np = corr_np + np.eye(num_assets) * (abs(min_eig) + 1e-6)
        d = np.sqrt(np.clip(np.diagonal(corr_np), 1e-12, None))
        corr_np = corr_np / np.outer(d, d)
        np.fill_diagonal(corr_np, 1.0)

    stressed_mu = mu * scenario["drift_modifier"]

    S0_tensor = torch.tensor(S0, dtype=torch.float32, device=device)
    mu_tensor = torch.tensor(stressed_mu, dtype=torch.float32, device=device)
    corr_tensor = torch.tensor(corr_np, dtype=torch.float32, device=device)
    vol0 = torch.tensor(stressed_vols, dtype=torch.float32, device=device)
    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

    L = torch.linalg.cholesky(corr_tensor)

    log_paths = torch.zeros((num_steps + 1, num_paths, num_assets), dtype=torch.float32, device=device)
    log_paths[0] = torch.log(S0_tensor)

    # CIR variance. theta is the stressed sample variance, not a fitted Heston theta.
    v0 = (vol0 ** 2).unsqueeze(0).repeat(num_paths, 1)
    v_paths = v0.clone()
    kappa = 2.0
    theta = v0.mean(dim=0)
    xi = 0.3
    rho = -0.7

    lambda_j = scenario["jump_intensity_lambda"]
    mu_j = scenario["jump_mean"]
    sig_j = scenario["jump_vol"]
    jump_k = torch.exp(torch.tensor(mu_j + 0.5 * sig_j**2, device=device)) - 1.0 if lambda_j > 0 else 0.0

    for t in range(num_steps):
        Z_asset_indep = torch.randn((num_paths, num_assets), device=device)
        Z_var_indep = torch.randn((num_paths, num_assets), device=device)

        Z_S = torch.matmul(Z_asset_indep, L.T)
        dW_S = Z_S * torch.sqrt(dt_tensor)
        Z_v_corr = rho * Z_S + np.sqrt(1.0 - rho**2) * Z_var_indep
        dW_v = Z_v_corr * torch.sqrt(dt_tensor)

        v_prev = v_paths
        v_next = (
            v_prev
            + kappa * (theta - v_prev) * dt_tensor
            + xi * torch.sqrt(torch.clamp(v_prev, min=0.0)) * dW_v
        )
        v_paths = torch.clamp(v_next, min=1e-6)

        effective_drift = mu_tensor - 0.5 * v_prev - lambda_j * jump_k

        jump_log_returns = 0.0
        if lambda_j > 0:
            jump_prob = lambda_j * dt
            jump_occurs = (torch.rand((num_paths, num_assets), device=device) < jump_prob).float()
            normal_jumps = torch.randn((num_paths, num_assets), device=device) * sig_j + mu_j
            jump_log_returns = jump_occurs * normal_jumps

        increment = (
            effective_drift * dt_tensor
            + torch.sqrt(torch.clamp(v_prev, min=1e-6)) * dW_S
            + jump_log_returns
        )
        log_paths[t + 1] = log_paths[t] + increment

    paths = torch.exp(log_paths)
    return paths


def simulate_vasicek_rates(
    n_steps,
    n_paths,
    dt,
    r0=0.06,
    a=0.5,
    b=0.06,
    sigma_r=0.01,
    device="cpu",
):
    """Ornstein-Uhlenbeck short rate. Independent of the equity Browns.

    Used only as an optional pathwise discount factor.
    Not a calibrated term-structure model and not dual-curve OIS.
    """
    dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
    r = torch.full((n_paths,), float(r0), dtype=torch.float32, device=device)
    out = torch.zeros((n_steps + 1, n_paths), dtype=torch.float32, device=device)
    out[0] = r
    for t in range(n_steps):
        z = torch.randn(n_paths, device=device)
        r = r + a * (b - r) * dt_t + sigma_r * torch.sqrt(dt_t) * z
        out[t + 1] = r
    return out
