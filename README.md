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

- `src/hornbill/optimize.py` – `gradient_descent(loss_fn, params, lr, steps, tol)`
- `examples/gradient_descent.py` – a classical quadratic and single-qubit ground-state search
- `tests/` – pytest suite
