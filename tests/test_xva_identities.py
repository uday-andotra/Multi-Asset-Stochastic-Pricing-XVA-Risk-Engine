import torch
from core.xva_calculator import calculate_xva_metrics_gpu


def _toy(paths, **kw):
    n_steps = paths.shape[0]
    dt = 1 / 252
    t = torch.linspace(0, (n_steps - 1) * dt, n_steps)
    r = torch.zeros(n_steps)
    return calculate_xva_metrics_gpu(
        paths,
        initial_value=100.0,
        scenario={"strike_multiplier": 1.0, "derivative_type": "call"},
        dt=dt,
        t_array=t,
        discount_rates=r,
        **kw,
    )


def test_ee_below_pfe():
    torch.manual_seed(0)
    paths = 100 + torch.randn(20, 500) * 5
    ee, pfe, _, _ = _toy(paths)
    assert torch.all(ee <= pfe + 1e-5)


def test_cva_zero_if_lgd_zero():
    torch.manual_seed(1)
    paths = 100 + torch.abs(torch.randn(20, 200))
    _, _, cva, _ = _toy(paths, recovery_rate=1.0)
    assert abs(cva) < 1e-8


def test_cva_rises_with_hazard():
    paths = torch.linspace(100, 130, 30).unsqueeze(1).repeat(1, 100)
    _, _, cva_lo, _ = _toy(paths, base_hazard_rate=0.01, wwr_alpha=0.0)
    _, _, cva_hi, _ = _toy(paths, base_hazard_rate=0.10, wwr_alpha=0.0)
    assert cva_hi > cva_lo


def test_cva_rises_with_wwr_alpha():
    paths = torch.linspace(100, 140, 40).unsqueeze(1).repeat(1, 80)
    _, _, cva_flat, _ = _toy(paths, wwr_alpha=0.0)
    _, _, cva_wwr, _ = _toy(paths, wwr_alpha=0.5)
    assert cva_wwr > cva_flat


def test_fva_rises_with_spread():
    paths = torch.linspace(100, 120, 30).unsqueeze(1).repeat(1, 80)
    _, _, _, fva_lo = _toy(paths, funding_spread=0.0)
    _, _, _, fva_hi = _toy(paths, funding_spread=0.05)
    assert abs(fva_lo) < 1e-10
    assert fva_hi > fva_lo
