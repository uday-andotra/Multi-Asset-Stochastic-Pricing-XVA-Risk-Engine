import torch
import numpy as np

def simulate_advanced_asset_paths_gpu(
    tickers, S0, mu, cov_matrix, scenario, T, dt, num_paths, device="cpu"
):
    num_assets = len(tickers)
    num_steps = int(T / dt)
    
    # 1. Rigorous Correlation & Volatility Stressing
    cov_np = np.array(cov_matrix, dtype=np.float64)
    vols = np.sqrt(np.clip(np.diagonal(cov_np), a_min=1e-8, a_max=None))
    
    # Extract correlation matrix R from covariance
    outer_vols = np.outer(vols, vols)
    corr_np = np.clip(cov_np / outer_vols, -1.0, 1.0)
    np.fill_diagonal(corr_np, 1.0)
    
    # Apply scenario volatility modifier
    stressed_vols = vols * scenario["vol_modifier"]
    
    # Apply scenario correlation stress
    corr_stress = scenario["correlation_stress"]
    if corr_stress != 0.0:
        if corr_stress > 0:
            # Crisis convergence: push all off-diagonal correlations toward 1.0
            corr_np = corr_np + corr_stress * (1.0 - corr_np)
        else:
            # Bullish diversification: push all off-diagonal correlations toward 0.0 or lower
            corr_np = corr_np * (1.0 + corr_stress)
        np.fill_diagonal(corr_np, 1.0)
        
    # Reconstruct stressed covariance matrix: Sigma = D * R * D
    stressed_cov_np = corr_np * np.outer(stressed_vols, stressed_vols)
    
    # Ensure positive-definiteness via spectral eigenvalue clipping
    min_eig = np.min(np.real(np.linalg.eigvalsh(stressed_cov_np)))
    if min_eig < 1e-5:
        stressed_cov_np += np.eye(num_assets) * (abs(min_eig) + 1e-4)
        
    stressed_mu = mu * scenario["drift_modifier"]
    
    S0_tensor = torch.tensor(S0, dtype=torch.float32, device=device)
    mu_tensor = torch.tensor(stressed_mu, dtype=torch.float32, device=device)
    cov_tensor = torch.tensor(stressed_cov_np, dtype=torch.float32, device=device)
    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
    
    L = torch.linalg.cholesky(cov_tensor)
    
    # 2. Log-Normal Path Initialization
    log_paths = torch.zeros((num_steps + 1, num_paths, num_assets), dtype=torch.float32, device=device)
    log_paths[0] = torch.log(S0_tensor)
    
    # 3. Heston Parameters & Initialization
    v0 = torch.diag(cov_tensor).unsqueeze(0).repeat(num_paths, 1)
    v_paths = v0.clone()
    kappa = 2.0  
    theta = v0.mean(dim=0) 
    xi = 0.3 
    rho = -0.7  # Asset-variance correlation
    
    # 4. Merton Jump Parameters & Compensator
    lambda_j = scenario["jump_intensity_lambda"]
    mu_j = scenario["jump_mean"]
    sig_j = scenario["jump_vol"]
    jump_k = torch.exp(torch.tensor(mu_j + 0.5 * sig_j**2, device=device)) - 1.0 if lambda_j > 0 else 0.0
    
    for t in range(num_steps):
        Z_asset_indep = torch.randn((num_paths, num_assets), device=device)
        Z_var_indep = torch.randn((num_paths, num_assets), device=device)
        
        dW_S = torch.matmul(Z_asset_indep, L.T) * torch.sqrt(dt_tensor)
        
        asset_vols = torch.sqrt(torch.clamp(torch.diag(cov_tensor), min=1e-6))
        Z_S_unit = dW_S / (asset_vols * torch.sqrt(dt_tensor))
        Z_v_corr = rho * Z_S_unit + np.sqrt(1.0 - rho**2) * Z_var_indep
        dW_v = Z_v_corr * torch.sqrt(dt_tensor)
        
        v_prev = v_paths
        v_next = v_prev + kappa * (theta - v_prev) * dt_tensor + xi * torch.sqrt(torch.clamp(v_prev, min=0.0)) * dW_v
        v_paths = torch.clamp(v_next, min=1e-6)
        
        effective_drift = mu_tensor - 0.5 * v_prev - lambda_j * jump_k
        
        jump_log_returns = 0.0
        if lambda_j > 0:
            jump_prob = lambda_j * dt
            jump_occurs = (torch.rand((num_paths, num_assets), device=device) < jump_prob).float()
            normal_jumps = torch.randn((num_paths, num_assets), device=device) * sig_j + mu_j
            jump_log_returns = jump_occurs * normal_jumps
            
        increment = effective_drift * dt_tensor + torch.sqrt(torch.clamp(v_prev, min=1e-6)) * dW_S + jump_log_returns
        log_paths[t+1] = log_paths[t] + increment
        
    paths = torch.exp(log_paths)
    return paths