# hornbill

A differentiable quantum simulator built on PyTorch.

Everything is a PyTorch tensor: gate angles, Hamiltonian couplings, pulse amplitudes, even evolution times and
atom spacings. So any quantity can be differentiated with autograd and optimised by gradient descent. Most functions
also accept batched parameters, so a whole sweep is computed in one call.

## Setup

```bash
uv sync
uv run pytest
```

The notebooks also need Jupyter and matplotlib, which are not package dependencies:

```bash
uv run --with jupyterlab --with matplotlib jupyter lab
```

## Quick start

Simulate a circuit on a state vector:

```python
import hornbill
from hornbill import operators as ops

sv = hornbill.StateVector(2).apply(ops.H, [0]).apply(ops.CNOT, [0, 1])
sv.probabilities()  # Bell state: [0.5, 0, 0, 0.5]
```

Optimise a laser pulse on two Rydberg atoms to implement a CZ gate:

```python
ham = hornbill.RydbergHamiltonian(distance=0.7)
pulse = hornbill.Pulse(n=50, time=10.0)  # 50 steps of (Rabi, detuning) amplitudes

def loss(params):
    return hornbill.unitary_infidelity(pulse.unitary(ham.controls, ham.drift), ops.CZ)

history = hornbill.gradient_descent(loss, [pulse.params], lr=1.0, steps=600)
```

## Modules

| Module | Contents |
| --- | --- |
| `operators` | Fixed operators (`I`, `X`, `Y`, `Z`, `H`, `S`, `T`, `CNOT`, `CZ`, `SWAP`); differentiable parametric gates (`rx`, `ry`, `rz`, `rxx`, `ryy`, `rzz`, `phase`, `u3`, `pauli_rotation`, `controlled`, `crx`, ...); batched `kron` for multi-qubit operators |
| `states` | `StateVector(n)` with `apply(operator, qubits)` and `probabilities()`; `zero_state` |
| `evolution` | `time_evolution(hamiltonian, t)`: exp(−iHt), differentiable in both H and t |
| `hamiltonians` | `RydbergHamiltonian(distance, c6)`: two-atom Rydberg Hamiltonian Ω/2 ΣX − Δ Σn + C6/r⁶ n⊗n, callable as `ham(rabi, detuning)` |
| `pulses` | `Pulse(n, time, n_controls)`: piecewise-constant control amplitudes (trainable `params`) with `hamiltonians`, `step_unitaries` and `unitary` for given control and drift operators |
| `metrics` | `unitary_distance` (global-phase-invariant metric) and `unitary_infidelity` (smooth loss) |
| `optimize` | `gradient_descent(loss_fn, params, lr, steps, tol)` |
| `estimation` | Bayesian parameter estimation: `GridPosterior` (grid posterior updated from measurement shots), `choose_setting` (adaptive experiment design by expected posterior variance), `sample_outcomes` |

All of these are importable from `hornbill` directly (e.g. `hornbill.Pulse`), except the operators, which live in
`hornbill.operators`.

## Notebooks

Rendered versions are at https://jimnel.github.io/hornbill/. They are re-executed whenever the notebooks or the
library change on `main`.

- `notebooks/neutral_atom.ipynb`: Rydberg spectrum, blockade, and optimising a CZ pulse with `Pulse` and `RydbergHamiltonian`
- `notebooks/optimisation.ipynb`: fitting a single-qubit Hamiltonian so that its time evolution equals a target gate
- `notebooks/bayesian_estimation.ipynb`: estimating a phase rate from Ramsey shots, with fixed and adaptive evolution times
- `notebooks/idea.ipynb`: scratch work

## Conventions

- Units with ħ = 1.
- Qubit 0 is the most significant: the two-qubit basis is |00⟩, |01⟩, |10⟩, |11⟩ = |q0 q1⟩, and `kron(A, B)` puts `A` on qubit 0.
- Fixed operators are `complex128`. Parametric gates follow the precision of their angles (float32 → complex64, float64 → complex128).
