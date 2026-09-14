from pathlib import Path

import torch

from core.monte_carlo import simulate_vasicek_rates
from core.yield_curve import fit_nss, load_instrument_tape
from core.xva_calculator import calculate_xva_metrics_gpu

CSV = Path(__file__).resolve().parents[1] / "data" / "curve_instruments.csv"


def test_discount_and_projection_strips_load():
    d = load_instrument_tape(CSV, "discount")
    p = load_instrument_tape(CSV, "projection")
    td, yd = d.get_curve()
    tp, yp = p.get_curve()
    assert len(td) >= 2 and len(tp) >= 2
    # projection tape is quoted richer than discount in the demo file
    assert yp[-1] > yd[-1]


def test_nss_fits_discount_tape():
    nss, t, y = fit_nss(load_instrument_tape(CSV, "discount"))
    assert nss.params is not None
    fitted = nss.get_curve(t)
    assert fitted.shape == y.shape


def test_vasicek_starts_at_r0():
    torch.manual_seed(0)
    r = simulate_vasicek_rates(n_steps=10, n_paths=32, dt=1 / 252, r0=0.05, b=0.05)
    assert r.shape == (11, 32)
    assert torch.allclose(r[0], torch.full((32,), 0.05))


def test_pathwise_df_changes_cva():
    torch.manual_seed(0)
    paths = torch.linspace(100, 130, 20).unsqueeze(1).repeat(1, 40)
    n_steps = paths.shape[0]
    dt = 1 / 252
    t = torch.linspace(0, (n_steps - 1) * dt, n_steps)
    flat = torch.zeros(n_steps)
    scenario = {"strike_multiplier": 1.0, "derivative_type": "call"}
    _, _, cva_flat, _ = calculate_xva_metrics_gpu(
        paths, 100.0, scenario, dt, t, flat, wwr_alpha=0.0
    )
    high_r = torch.full((n_steps, 40), 0.20)
    _, _, cva_high_r, _ = calculate_xva_metrics_gpu(
        paths, 100.0, scenario, dt, t, flat, wwr_alpha=0.0, short_rates=high_r
    )
    assert cva_high_r < cva_flat
