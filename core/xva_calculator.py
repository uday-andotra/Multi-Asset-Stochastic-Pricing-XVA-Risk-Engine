import torch

def calculate_xva_metrics_gpu(
    paths, initial_value, scenario, dt, t_array, discount_rates, 
    recovery_rate=0.40, base_hazard_rate=0.03, wwr_alpha=0.1, funding_spread=0.01
):
    strike_multiplier = scenario.get("strike_multiplier", 1.00)
    derivative_type = scenario.get("derivative_type", "call")
    strike_price = initial_value * strike_multiplier
    
    if derivative_type == "call":
        V = paths - strike_price 
    elif derivative_type == "put":
        V = strike_price - paths
    else:
        V = paths
        
    E = torch.clamp(V, min=0.0) 
    
    EE = torch.mean(E, dim=1)
    PFE_95 = torch.quantile(E, 0.95, dim=1)
    
    normalized_exposure = E / max(initial_value, 1.0)
    wwr_exponent = torch.clamp(wwr_alpha * normalized_exposure, max=20.0)
    lambda_t = base_hazard_rate * torch.exp(wwr_exponent)
    
    integral_lambda = torch.cumsum(lambda_t * dt, dim=0)
    Q = torch.exp(-integral_lambda)
    
    Q_shifted = torch.cat([torch.ones((1, Q.shape[1]), device=Q.device), Q[:-1, :]], dim=0)
    PD = Q_shifted - Q
    
    df = torch.exp(-discount_rates * t_array).unsqueeze(1)
    
    LGD = 1.0 - recovery_rate
    pathwise_CVA = torch.sum(LGD * E * df * PD, dim=0)
    final_CVA = torch.mean(pathwise_CVA).item()
    
    pathwise_FVA = torch.sum(funding_spread * E * df * dt, dim=0)
    final_FVA = torch.mean(pathwise_FVA).item()
    
    return EE, PFE_95, final_CVA, final_FVA