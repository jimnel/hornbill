import math

import pytest
import torch

from hornbill import gates
from hornbill.evolution import time_evolution


def random_hermitian(dim, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, dim, dtype=torch.complex128, generator=gen)
    return (a + a.mH) / 2


def test_zero_time_is_identity():
    torch.testing.assert_close(time_evolution(random_hermitian(4, 0), 0.0), torch.eye(4, dtype=torch.complex128))


def test_result_is_unitary():
    u = time_evolution(random_hermitian(4, 1), 0.83)
    torch.testing.assert_close(u @ u.mH, torch.eye(4, dtype=torch.complex128))


def test_matches_pauli_rotations():
    theta = 0.7
    # RX(theta) = exp(-i theta X / 2), i.e. H = X / 2 evolved for time theta
    torch.testing.assert_close(time_evolution(gates.X / 2, theta), gates.rx(theta))
    torch.testing.assert_close(time_evolution(gates.Z, theta / 2), gates.rz(theta))
    torch.testing.assert_close(time_evolution(torch.kron(gates.Z, gates.Z) / 2, theta), gates.rzz(theta))


def test_diagonal_hamiltonian_gives_phases():
    energies = torch.tensor([0.0, 1.0, -2.5], dtype=torch.float64)
    t = 1.3
    expected = torch.diag(torch.polar(torch.ones(3, dtype=torch.float64), -energies * t))
    torch.testing.assert_close(time_evolution(torch.diag(energies), t), expected)


def test_composition_in_time():
    h = random_hermitian(3, 2)
    torch.testing.assert_close(time_evolution(h, 0.4) @ time_evolution(h, 0.9), time_evolution(h, 1.3))


def test_batched_times():
    h = random_hermitian(2, 3)
    ts = torch.linspace(0, 2, 5, dtype=torch.float64)
    out = time_evolution(h, ts)
    assert out.shape == (5, 2, 2)
    for t, u in zip(ts, out):
        torch.testing.assert_close(u, time_evolution(h, t.item()))


def test_batched_hamiltonians():
    hs = torch.stack([random_hermitian(2, s) for s in range(4)])
    assert time_evolution(hs, 0.5).shape == (4, 2, 2)


def test_dtypes():
    real_h = torch.tensor([[1.0, 0.5], [0.5, -1.0]], dtype=torch.float64)
    assert time_evolution(real_h, 1.0).dtype == torch.complex128
    h32 = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
    assert time_evolution(h32, torch.tensor(1.0, dtype=torch.float32)).dtype == torch.complex64


def test_derivative_in_time_is_schrodinger_equation():
    """dU/dt = -i H U."""
    h = random_hermitian(3, 4)
    t = torch.tensor(0.6, dtype=torch.float64)
    du_dt = torch.autograd.functional.jacobian(lambda s: torch.view_as_real(time_evolution(h, s)), t)
    du_dt = torch.view_as_complex(du_dt)
    torch.testing.assert_close(du_dt, -1j * h @ time_evolution(h, t))


def test_gradcheck_through_hamiltonian_parameters_and_time():
    def u(params, t):
        h = params[0] * gates.X + params[1] * gates.Y + params[2] * gates.Z
        return time_evolution(h, t)

    params = torch.tensor([0.3, -0.8, 0.5], dtype=torch.float64, requires_grad=True)
    t = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(u, (params, t))


@pytest.mark.parametrize("h", [torch.zeros(2, 3), torch.zeros(2)])
def test_rejects_non_square(h):
    with pytest.raises(ValueError):
        time_evolution(h, 1.0)


def test_rejects_complex_time():
    with pytest.raises(TypeError):
        time_evolution(gates.X, torch.tensor(1.0 + 1.0j))
