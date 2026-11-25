import pytest
import torch

from hornbill import Pulse, RydbergHamiltonian, time_evolution
from hornbill import operators as ops

N = (ops.I - ops.Z) / 2


def test_matches_definition():
    ham = RydbergHamiltonian(distance=2.0, c6=3.0)
    expected = (
        0.7 / 2 * (ops.kron(ops.X, ops.I) + ops.kron(ops.I, ops.X))
        - 0.4 * (ops.kron(N, ops.I) + ops.kron(ops.I, N))
        + 3.0 / 2.0**6 * ops.kron(N, N)
    )
    torch.testing.assert_close(ham(0.7, 0.4), expected)


def test_diagonal_energies():
    # basis |00>, |01>, |10>, |11>; |1> is the Rydberg state
    ham = RydbergHamiltonian(distance=1.0, c6=5.0)
    torch.testing.assert_close(ham(0.0, 2.0).diagonal().real, torch.tensor([0.0, -2.0, -2.0, -4.0 + 5.0], dtype=torch.float64))


def test_hermitian():
    h = RydbergHamiltonian()(1.3, -0.6)
    torch.testing.assert_close(h, h.mH)


def test_batched():
    ham = RydbergHamiltonian()
    rabi = torch.linspace(0, 1, 4, dtype=torch.float64)
    h = ham(rabi, 0.5)
    assert h.shape == (4, 4, 4)
    torch.testing.assert_close(h[2], ham(rabi[2].item(), 0.5))


def test_blockade():
    # strong interaction: resonant drive from |00> cannot reach |11>
    ham = RydbergHamiltonian(distance=0.3)
    u = time_evolution(ham(1.0, 0.0), 3.0)
    assert u[3, 0].abs().item() ** 2 < 1e-3


def test_pulse_integration():
    ham = RydbergHamiltonian(distance=1.5)
    params = torch.tensor([[0.3, -0.2], [1.1, 0.5], [0.0, 0.9]], dtype=torch.float64)
    pulse = Pulse(3, 1.2, params=params)
    expected = torch.eye(4, dtype=ops.DTYPE)
    for rabi, detuning in params:
        expected = time_evolution(ham(rabi, detuning), pulse.dt) @ expected
    torch.testing.assert_close(pulse.unitary(ham.controls, ham.drift), expected)


def test_gradient_wrt_distance():
    r = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda r: RydbergHamiltonian(distance=r)(0.5, 0.1), (r,))


def test_gradient_wrt_controls():
    args = (torch.tensor(0.5, dtype=torch.float64, requires_grad=True), torch.tensor(0.1, dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(RydbergHamiltonian(), args)
