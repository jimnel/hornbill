# hornbill
PyTorch based quantum simulator

All quantities are PyTorch tensors, so any parameter (gate angles, Hamiltonian
couplings, ...) can be differentiated with autograd.

## Setup

```bash
uv sync
uv run pytest
uv run python examples/gradient_descent.py
```

## Layout

- `src/hornbill/estimation.py` – Bayesian parameter estimation: `GridPosterior` (grid posterior updated from measurement shots), `choose_setting` (adaptive choice of e.g. evolution time by expected posterior variance), `sample_outcomes`
- `src/hornbill/evolution.py` – `time_evolution(hamiltonian, t)` returns exp(−iHt), differentiable in H and t
- `src/hornbill/gates.py` – fixed gates (`X`, `H`, `CNOT`, ...) and differentiable parametric gates (`rx`, `u3`, `rzz`, `controlled`, ...); angles can be batched tensors
- `src/hornbill/metrics.py` – `unitary_distance` (phase-invariant metric) and `unitary_infidelity` (smooth loss) between unitaries
- `src/hornbill/optimize.py` – `gradient_descent(loss_fn, params, lr, steps, tol)`
- `examples/gradient_descent.py` – a classical quadratic and single-qubit ground-state search
- `tests/` – pytest suite
