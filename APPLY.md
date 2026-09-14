# How to apply this overlay

Un-archive the GitHub repo first. Then copy these files on top of
`Multi-Asset-Stochastic-Pricing-XVA-Risk-Engine`.

```bash
cd Multi-Asset-Stochastic-Pricing-XVA-Risk-Engine
# backup first
cp README.md README.old.md
cp -r core core.bak

cp -R /path/to/xva-fix/. .
# keep your existing app.py and engine_backend.py unless you also copy those

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
streamlit run app.py
```

What changed

- `README.md` — matches the code. No enterprise / dual-curve / Vasicek claims.
- `core/monte_carlo.py` — correlation Cholesky + vol only from `sqrt(v)`. Stops double-counting sample vol.
- `core/xva_calculator.py` — same maths, documented.
- `core/yield_curve.py` — same maths, documented.
- `engine_backend.py` — status text no longer says GPU when the box is a CPU.
- `app.py` — title and caption match the study-desk tone.
- `tests/` — identities you can defend on a call.
- `requirements.txt` — adds pytest.

Leave `Document/` as it is.
