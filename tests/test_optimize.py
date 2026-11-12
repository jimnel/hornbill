import pytest
import torch

from hornbill import gradient_descent


def test_quadratic_converges_to_minimum():
    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    history = gradient_descent(lambda x, y: (x - 3) ** 2 + 2 * (y + 1) ** 2, [x, y], lr=0.1, steps=200)

    assert x.item() == pytest.approx(3.0, abs=1e-8)
    assert y.item() == pytest.approx(-1.0, abs=1e-8)
    assert history[-1] < history[0]


def test_qubit_ground_state_energy():
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    H = X + Z

    def energy(theta, phi):
        psi = torch.stack([torch.cos(theta / 2).to(H.dtype), torch.exp(1j * phi) * torch.sin(theta / 2)])
        return torch.vdot(psi, H @ psi).real

    theta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    history = gradient_descent(energy, [theta, phi], lr=0.2, steps=500, tol=1e-12)

    assert history[-1] == pytest.approx(-(2**0.5), abs=1e-8)


def test_tol_stops_early():
    x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    history = gradient_descent(lambda x: x**2, [x], lr=0.1, steps=10_000, tol=1e-10)
    assert len(history) < 10_000


def test_rejects_params_without_grad():
    x = torch.tensor(1.0)
    with pytest.raises(ValueError, match="requires_grad"):
        gradient_descent(lambda x: x**2, [x])
