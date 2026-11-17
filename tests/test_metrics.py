import cmath
import math

import pytest
import torch

from hornbill import gates, gradient_descent
from hornbill.metrics import unitary_distance, unitary_infidelity


def random_unitary(dim, seed, batch=()):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(*batch, dim, dim, dtype=torch.complex128, generator=gen)
    q, _ = torch.linalg.qr(a)
    return q


def test_identical_unitaries_have_zero_distance():
    u = random_unitary(4, seed=0)
    assert unitary_distance(u, u).item() == pytest.approx(0.0, abs=1e-7)
    assert unitary_infidelity(u, u).item() == pytest.approx(0.0, abs=1e-12)


def test_global_phase_is_ignored():
    u = random_unitary(4, seed=1)
    v = cmath.exp(0.77j) * u
    assert unitary_distance(u, v).item() == pytest.approx(0.0, abs=1e-7)
    assert unitary_infidelity(u, v).item() == pytest.approx(0.0, abs=1e-12)


def test_orthogonal_unitaries_have_maximal_distance():
    # Tr(X^dagger Z) = 0
    assert unitary_distance(gates.X, gates.Z).item() == pytest.approx(1.0)
    assert unitary_infidelity(gates.X, gates.Z).item() == pytest.approx(1.0)


def test_rotation_distance_from_identity():
    theta = 0.9
    expected = math.sqrt(1 - abs(math.cos(theta / 2)))
    assert unitary_distance(gates.rx(theta), gates.I).item() == pytest.approx(expected, abs=1e-12)


def test_matches_phase_minimised_frobenius_norm():
    """D(U, V) = min_phi ||U - e^{i phi} V||_F / sqrt(2d), checked by brute force over phi."""
    u, v = random_unitary(3, seed=2), random_unitary(3, seed=3)
    phis = torch.linspace(0, 2 * math.pi, 20001, dtype=torch.float64)
    phases = torch.polar(torch.ones_like(phis), phis)[:, None, None]
    brute = torch.linalg.matrix_norm(u - phases * v).min() / math.sqrt(2 * 3)
    assert unitary_distance(u, v).item() == pytest.approx(brute.item(), abs=1e-6)


def test_symmetric_and_bounded():
    u, v = random_unitary(4, seed=4, batch=(50,)), random_unitary(4, seed=5, batch=(50,))
    d_uv, d_vu = unitary_distance(u, v), unitary_distance(v, u)
    torch.testing.assert_close(d_uv, d_vu)
    assert ((d_uv >= 0) & (d_uv <= 1)).all()


def test_triangle_inequality():
    a, b, c = (random_unitary(4, seed=s, batch=(200,)) for s in (6, 7, 8))
    lhs = unitary_distance(a, c)
    rhs = unitary_distance(a, b) + unitary_distance(b, c)
    assert (lhs <= rhs + 1e-12).all()


def test_infidelity_relation_to_distance():
    u, v = random_unitary(4, seed=9, batch=(10,)), random_unitary(4, seed=10, batch=(10,))
    d = unitary_distance(u, v)
    torch.testing.assert_close(unitary_infidelity(u, v), d**2 * (2 - d**2))


def test_batching_and_broadcasting():
    thetas = torch.linspace(0, math.pi, 5, dtype=torch.float64)
    out = unitary_distance(gates.rx(thetas), gates.I)
    assert out.shape == (5,)
    for t, d in zip(thetas, out):
        assert d.item() == pytest.approx(unitary_distance(gates.rx(t.item()), gates.I).item())


def test_mixed_precision_is_promoted():
    u = gates.rx(torch.tensor(0.3, dtype=torch.float32))  # complex64
    assert unitary_distance(u, gates.I).dtype == torch.float64


@pytest.mark.parametrize(
    "u, v",
    [
        (gates.X, gates.CNOT),              # different sizes
        (torch.zeros(2, 3), torch.zeros(2, 3)),  # not square
        (torch.zeros(2), torch.zeros(2)),   # not a matrix
    ],
)
def test_rejects_bad_shapes(u, v):
    with pytest.raises(ValueError):
        unitary_distance(u, v)


def test_infidelity_gradcheck():
    def loss(a, b, c):
        return unitary_infidelity(gates.u3(a, b, c), gates.H)

    args = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (0.4, 0.2, -0.5))
    assert torch.autograd.gradcheck(loss, args)


def test_infidelity_gradient_is_zero_at_minimum():
    theta = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    loss = unitary_infidelity(gates.rx(theta), gates.rx(0.6))
    (grad,) = torch.autograd.grad(loss, theta)
    assert torch.isfinite(grad) and grad.abs().item() < 1e-10


def test_fit_u3_to_hadamard():
    """Gradient descent on the infidelity recovers H (up to global phase) with a u3 gate."""
    params = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (0.4, 0.2, -0.5)]
    gradient_descent(lambda a, b, c: unitary_infidelity(gates.u3(a, b, c), gates.H), params, lr=0.5, steps=2000)
    assert unitary_distance(gates.u3(*params), gates.H).item() < 1e-5
