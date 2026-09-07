import torch
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

from core.monte_carlo import simulate_advanced_asset_paths_gpu
from core.yield_curve import YieldCurveBootstrapper, NelsonSiegelSvensson
from core.xva_calculator import calculate_xva_metrics_gpu

def get_scenario_config(scenario_name="baseline"):
    scenarios = {
        "baseline": {
            "drift_modifier": 1.2,
            "vol_modifier": 1.0,
            "jump_intensity_lambda": 0.2,
            "jump_mean": 0.01,
            "jump_vol": 0.08,
            "correlation_stress": 0.0, 
            "derivative_type": "call",
            "strike_multiplier": 1.00  
        },
        "bullish": {
            "drift_modifier": 1.8,
            "vol_modifier": 0.7,       
            "jump_intensity_lambda": 0.1, 
            "jump_mean": 0.03,         
            "jump_vol": 0.05,
            "correlation_stress": -0.2, # Enhanced diversification
            "derivative_type": "call",
            "strike_multiplier": 1.00
        },
        "market_crash": {
            "drift_modifier": -1.5,    
            "vol_modifier": 2.0,       
            "jump_intensity_lambda": 3.0, 
            "jump_mean": -0.10,        
            "jump_vol": 0.20,
            "correlation_stress": 0.5, # Breakdown of diversification (correlations surge)
            "derivative_type": "put",  
            "strike_multiplier": 1.00  
        }
    }
    return scenarios.get(scenario_name.lower(), scenarios["baseline"])

def run_risk_engine(tickers, initial_capital, custom_weights=None, scenario_name="baseline", progress_callback=None):
    def update_status(msg):
        if progress_callback:
            progress_callback(msg)

    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")

    scenario = get_scenario_config(scenario_name)

    update_status("📥 Fetching historical market data via yfinance...")
    raw_data = yf.download(tickers, period="1y", progress=False)['Close']
    data = raw_data.ffill().bfill().dropna(axis=1, how='all')
    
    S0 = data.iloc[-1].values.tolist()
    active_tickers = data.columns.tolist()
    num_assets = len(active_tickers)
    
    log_returns = np.log(data / data.shift(1)).dropna()
    mu = (log_returns.mean() * 252).values
    cov_matrix = (log_returns.cov() * 252).values

    maturity_years = 2.0
    dt = 1/252
    simulation_paths = 3000

    torch.manual_seed(42)
    np.random.seed(42)
    
    update_status(f"⚙️ Phase 1: Running GPU-accelerated Monte Carlo simulation ({simulation_paths} paths)...")
    paths = simulate_advanced_asset_paths_gpu(
        tickers=active_tickers, S0=S0, mu=mu, cov_matrix=cov_matrix, 
        scenario=scenario, T=maturity_years, dt=dt, num_paths=simulation_paths, device=device
    )

    update_status("⚙️ Phase 2: Bootstrapping yield curve & fitting NSS model...")
    bootstrapper = YieldCurveBootstrapper()
    bootstrapper.add_cash_rate(0.25, 0.065)
    bootstrapper.add_cash_rate(0.5, 0.067)
    bootstrapper.add_swap_rate(1.0, 0.069)
    bootstrapper.add_swap_rate(2.0, 0.071)
    bootstrapper.add_swap_rate(3.0, 0.073)
    bootstrapper.add_swap_rate(5.0, 0.075)
    bootstrapper.add_swap_rate(10.0, 0.077)
    bootstrapper.add_swap_rate(30.0, 0.080)
    
    t_raw, y_raw = bootstrapper.get_curve()
    nss = NelsonSiegelSvensson()
    nss.fit(t_raw, y_raw)
    
    num_steps = int(maturity_years / dt)
    t_array = torch.linspace(0, maturity_years, num_steps + 1, device=device)
    discount_rates = torch.tensor(nss.get_curve(t_array.cpu().numpy()), dtype=torch.float32, device=device)

    S0_tensor = torch.tensor(S0, dtype=torch.float32, device=device)
    
    if custom_weights and len(custom_weights) == num_assets:
        encoded_weights_np = np.array(custom_weights)
        encoded_weights_np = encoded_weights_np / np.sum(encoded_weights_np)
    else:
        raw_random_weights = np.random.rand(num_assets)
        encoded_weights_np = raw_random_weights / np.sum(raw_random_weights)
    
    encoded_weights = torch.tensor(encoded_weights_np, dtype=torch.float32, device=device)
    encoded_units = (initial_capital * encoded_weights) / S0_tensor
    
    encoded_portfolio_paths = torch.sum(paths * encoded_units, dim=2)
    encoded_initial_value = encoded_portfolio_paths[0, 0].item()
    
    update_status("⚙️ Phase 3: Calculating baseline XVA metrics (EE, PFE, CVA, FVA)...")
    enc_EE, enc_PFE_95, enc_CVA, enc_FVA = calculate_xva_metrics_gpu(
        paths=encoded_portfolio_paths, initial_value=encoded_initial_value, scenario=scenario,
        dt=dt, t_array=t_array, discount_rates=discount_rates,
        recovery_rate=0.40, base_hazard_rate=0.03,
        wwr_alpha=0.1, funding_spread=0.01
    )

    def objective_function(weights_np):
        weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
        weights = torch.clamp(weights, min=0.0) 
        if torch.sum(weights) > 0:
            weights = weights / torch.sum(weights)  
        else:
            return 1e6 
            
        allocated_capital = initial_capital * weights
        units = allocated_capital / S0_tensor
        
        portfolio_paths = torch.sum(paths * units, dim=2)
        current_initial_value = portfolio_paths[0, 0].item()
        
        EE, PFE_95, CVA, FVA = calculate_xva_metrics_gpu(
            paths=portfolio_paths, initial_value=current_initial_value, scenario=scenario,
            dt=dt, t_array=t_array, discount_rates=discount_rates,
            recovery_rate=0.40, base_hazard_rate=0.03,
            wwr_alpha=0.1, funding_spread=0.01
        )
        
        risk_score = PFE_95.max().item() + (CVA * 15.0) + (FVA * 8.0)
        
        # Scenario-aware growth penalty: only enforce upward growth in non-crash scenarios
        if scenario_name.lower() != "market_crash":
            expected_terminal_value = torch.mean(portfolio_paths[-1, :]).item()
            growth_penalty = max(0.0, current_initial_value - expected_terminal_value) * 5.0
            return risk_score + growth_penalty
        else:
            return risk_score

    bounds = [(0.0, 0.25) for _ in range(num_assets)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    update_status("🔍 Running global optimization using differential_evolution...")
    global_result = differential_evolution(
        objective_function, 
        bounds=bounds, 
        strategy='best1bin', 
        maxiter=15, 
        popsize=5, 
        seed=42
    )
    
    global_guess = global_result.x / np.sum(global_result.x)

    update_status("🎯 Running local optimization using SLSQP...")
    local_result = minimize(
        objective_function, 
        global_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options={'maxiter': 150, 'ftol': 1e-6, 'eps': 1e-3}
    )
    
    optimal_weights_np = local_result.x / np.sum(local_result.x)

    optimal_weights = torch.tensor(optimal_weights_np, dtype=torch.float32, device=device)
    optimal_units = (initial_capital * optimal_weights) / S0_tensor
    optimal_portfolio_paths = torch.sum(paths * optimal_units, dim=2)
    
    optimal_initial_value = optimal_portfolio_paths[0, 0].item()

    opt_EE, opt_PFE_95, opt_CVA, opt_FVA = calculate_xva_metrics_gpu(
        paths=optimal_portfolio_paths, initial_value=optimal_initial_value, scenario=scenario,
        dt=dt, t_array=t_array, discount_rates=discount_rates,
        recovery_rate=0.40, base_hazard_rate=0.03,
        wwr_alpha=0.1, funding_spread=0.01
    )

    update_status("📈 Generating final risk exposure and yield curve figures...")
    time_axis = t_array.cpu().numpy()
    
    encoded_strike = encoded_initial_value * scenario["strike_multiplier"]
    optimal_strike = optimal_initial_value * scenario["strike_multiplier"]
    
    if scenario["derivative_type"] == "call":
        baseline_exposure = torch.clamp(encoded_portfolio_paths - encoded_strike, min=0.0).cpu().numpy()
        optimal_exposure = torch.clamp(optimal_portfolio_paths - optimal_strike, min=0.0).cpu().numpy()
    else:
        baseline_exposure = torch.clamp(encoded_strike - encoded_portfolio_paths, min=0.0).cpu().numpy()
        optimal_exposure = torch.clamp(optimal_strike - optimal_portfolio_paths, min=0.0).cpu().numpy()
        
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 10))
    fig1.suptitle(f'SLIDE 1: Baseline Portfolio Risk Profile ({scenario_name.upper()})', fontsize=14, fontweight='bold')
    axes1[0].plot(time_axis, encoded_portfolio_paths[:, :15].cpu().numpy(), color='red', alpha=0.25, linewidth=0.8)
    axes1[0].axhline(y=encoded_initial_value, color='black', linestyle='--', label='Initial Value')
    axes1[0].set_title('Phase 1: Baseline Total Portfolio Value Paths (₹)')
    axes1[0].set_ylabel('Portfolio Value (INR)')
    axes1[0].legend()
    axes1[0].grid(True)

    axes1[1].plot(time_axis, baseline_exposure[:, :20], color='gray', alpha=0.1) 
    axes1[1].plot(time_axis, enc_EE.cpu().numpy(), color='darkblue', linewidth=2, label='Baseline Expected Exposure (EE)')
    axes1[1].plot(time_axis, enc_PFE_95.cpu().numpy(), color='darkred', linewidth=2, linestyle='--', label='Baseline 95% PFE')
    axes1[1].set_title(f'Phase 3: Aggregate Counterparty Risk Exposure ({scenario["derivative_type"].upper()} Option)')
    axes1[1].set_xlabel('Time (Years)')
    axes1[1].set_ylabel('Exposure (INR)')
    axes1[1].legend()
    axes1[1].grid(True)
    plt.tight_layout()

    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10))
    fig2.suptitle(f'SLIDE 2: Hybrid Optimized Portfolio Risk Profile ({scenario_name.upper()})', fontsize=14, fontweight='bold')
    axes2[0].plot(time_axis, optimal_portfolio_paths[:, :15].cpu().numpy(), color='blue', alpha=0.25, linewidth=0.8)
    axes2[0].axhline(y=optimal_initial_value, color='black', linestyle='--', label='Initial Value')
    axes2[0].set_title('Phase 1: Optimized Total Portfolio Value Paths (₹)')
    axes2[0].set_ylabel('Portfolio Value (INR)')
    axes2[0].legend()
    axes2[0].grid(True)

    axes2[1].plot(time_axis, optimal_exposure[:, :20], color='gray', alpha=0.1) 
    axes2[1].plot(time_axis, opt_EE.cpu().numpy(), color='darkblue', linewidth=2, label='Optimized Expected Exposure (EE)')
    axes2[1].plot(time_axis, opt_PFE_95.cpu().numpy(), color='darkorange', linewidth=2, linestyle='--', label='Optimized 95% PFE')
    axes2[1].set_title(f'Phase 3: Optimized Counterparty Risk Exposure ({scenario["derivative_type"].upper()} Option)')
    axes2[1].set_xlabel('Time (Years)')
    axes2[1].set_ylabel('Exposure (INR)')
    axes2[1].legend()
    axes2[1].grid(True)
    plt.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.suptitle('SLIDE 3: Macroeconomic Discounting Environment', fontsize=14, fontweight='bold')
    dense_t = np.linspace(0.1, 30.0, 100)
    dense_y = nss.get_curve(dense_t) * 100
    ax3.plot(dense_t, dense_y, label='NSS Fitted Curve', color='blue', linewidth=2)
    ax3.scatter(t_raw, y_raw * 100, color='red', label='Market Rates', zorder=5)
    ax3.set_title('Phase 2: Nelson-Siegel-Svensson Yield Curve')
    ax3.set_xlabel('Maturity (Years)')
    ax3.set_ylabel('Zero Rate (%)')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()

    metrics = {
        "baseline": {"PFE": enc_PFE_95.max().item(), "EE": enc_EE.max().item(), "CVA": enc_CVA, "FVA": enc_FVA},
        "optimized": {"PFE": opt_PFE_95.max().item(), "EE": opt_EE.max().item(), "CVA": opt_CVA, "FVA": opt_FVA}
    }
    allocation = list(zip(active_tickers, encoded_weights_np, optimal_weights_np))

    update_status("✅ Simulation & Optimization Complete!")
    return metrics, allocation, (fig1, fig2, fig3)