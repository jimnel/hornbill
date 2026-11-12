"""Minimal gradient descent examples.

1. A classical quadratic, to show the optimiser on its own.
2. The ground-state energy of a single qubit with H = X + Z, found by
   differentiating <psi(theta, phi)| H |psi(theta, phi)> with respect to the
   state parameters. The exact answer is -sqrt(2).

Run with:  uv run python examples/gradient_descent.py
"""

import math

import torch

from hornbill import gradient_descent

DTYPE = torch.complex128

# --- 1. Quadratic: f(x, y) = (x - 3)^2 + 2 (y + 1)^2, minimum at (3, -1) -----
x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
y = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

history = gradient_descent(lambda x, y: (x - 3) ** 2 + 2 * (y + 1) ** 2, [x, y], lr=0.1, steps=200)
print(f"quadratic: x={x.item():.6f}, y={y.item():.6f}, loss={history[-1]:.2e}")

# --- 2. Single-qubit energy minimisation -------------------------------------
X = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
Z = torch.tensor([[1, 0], [0, -1]], dtype=DTYPE)
H = X + Z


def state(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """|psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return torch.stack(
        [torch.cos(theta / 2).to(DTYPE), torch.exp(1j * phi) * torch.sin(theta / 2)]
    )


def energy(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    psi = state(theta, phi)
    return torch.vdot(psi, H @ psi).real


theta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
phi = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

history = gradient_descent(energy, [theta, phi], lr=0.2, steps=500, tol=1e-12)
print(
    f"qubit:     E={history[-1]:.8f} (exact {-math.sqrt(2):.8f}) "
    f"after {len(history)} steps, theta={theta.item():.4f}, phi={phi.item():.4f}"
)
