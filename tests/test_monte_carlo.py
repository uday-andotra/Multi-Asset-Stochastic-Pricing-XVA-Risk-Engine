import numpy as np
import torch
from core.monte_carlo import simulate_advanced_asset_paths_gpu


def _scenario(jumps=False, corr=0.0, vol=1.0):
    return {
        "drift_modifier": 1.0,
        "vol_modifier": vol,
        "jump_intensity_lambda": 2.0 if jumps else 0.0,
        "jump_mean": -0.05,
        "jump_vol": 0.1,
        "correlation_stress": corr,
        "derivative_type": "call",
        "strike_multiplier": 1.0,
    }


def test_paths_start_at_spot():
    torch.manual_seed(0)
    S0 = [100.0, 50.0]
    mu = np.array([0.05, 0.04])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    paths = simulate_advanced_asset_paths_gpu(
        tickers=["A", "B"],
        S0=S0,
        mu=mu,
        cov_matrix=cov,
        scenario=_scenario(),
        T=5 / 252,
        dt=1 / 252,
        num_paths=32,
        device="cpu",
    )
    assert paths.shape == (6, 32, 2)
    np.testing.assert_allclose(paths[0, 0].cpu().numpy(), S0, rtol=1e-5)


def test_paths_stay_positive():
    torch.manual_seed(1)
    paths = simulate_advanced_asset_paths_gpu(
        tickers=["A"],
        S0=[100.0],
        mu=np.array([0.0]),
        cov_matrix=np.array([[0.04]]),
        scenario=_scenario(jumps=True),
        T=0.2,
        dt=1 / 252,
        num_paths=64,
        device="cpu",
    )
    assert torch.all(paths > 0)
