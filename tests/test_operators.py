import math

import pytest
import torch

from hornbill import operators as ops


def assert_close(a, b, atol=1e-12):
    torch.testing.assert_close(a, torch.as_tensor(b, dtype=a.dtype), atol=atol, rtol=0)


def is_unitary(u):
    eye = torch.eye(u.shape[-1], dtype=u.dtype).expand_as(u)
    return torch.allclose(u @ u.mH, eye, atol=1e-12)


FIXED = [ops.I, ops.X, ops.Y, ops.Z, ops.H, ops.S, ops.T, ops.CNOT, ops.CZ, ops.SWAP]
PARAMETRIC_1 = [ops.rx, ops.ry, ops.rz, ops.phase, ops.rxx, ops.ryy, ops.rzz, ops.crx, ops.cry, ops.crz]


@pytest.mark.parametrize("u", FIXED)
def test_fixed_gates_are_unitary(u):
    assert is_unitary(u)


@pytest.mark.parametrize("gate", PARAMETRIC_1)
def test_parametric_gates_are_unitary(gate):
    assert is_unitary(gate(0.734))


def test_u3_is_unitary():
    assert is_unitary(ops.u3(0.3, -1.2, 2.5))


def test_pauli_algebra():
    assert_close(ops.X @ ops.Y, 1j * ops.Z)
    assert_close(ops.H @ ops.Z @ ops.H, ops.X)
    assert_close(ops.S @ ops.S, ops.Z)
    assert_close(ops.T @ ops.T, ops.S)


def test_rotations_at_pi():
    assert_close(ops.rx(math.pi), -1j * ops.X)
    assert_close(ops.ry(math.pi), -1j * ops.Y)
    assert_close(ops.rz(math.pi), -1j * ops.Z)


def test_rz_and_phase_differ_by_global_phase():
    phi = 0.81
    assert_close(ops.rz(phi), complex(math.cos(-phi / 2), math.sin(-phi / 2)) * ops.phase(phi))


def test_rzz_is_diagonal():
    theta = 0.4
    m, p = complex(math.cos(theta / 2), -math.sin(theta / 2)), complex(math.cos(theta / 2), math.sin(theta / 2))
    assert_close(ops.rzz(theta), torch.diag(torch.tensor([m, p, p, m], dtype=torch.complex128)))


def test_u3_special_cases():
    theta, phi, lam = 0.3, -1.2, 2.5
    assert_close(ops.u3(theta, 0.0, 0.0), ops.ry(theta))
    assert_close(ops.u3(0.0, 0.0, lam), ops.phase(lam))
    assert_close(ops.u3(math.pi / 2, 0.0, math.pi), ops.H)
    # U3 = e^{i(phi+lam)/2} RZ(phi) RY(theta) RZ(lam)
    gp = complex(math.cos((phi + lam) / 2), math.sin((phi + lam) / 2))
    assert_close(ops.u3(theta, phi, lam), gp * ops.rz(phi) @ ops.ry(theta) @ ops.rz(lam))


def test_controlled():
    assert_close(ops.controlled(ops.X), ops.CNOT)
    assert_close(ops.controlled(ops.Z), ops.CZ)


def test_batched_angles():
    thetas = torch.linspace(0, 2 * math.pi, 7, dtype=torch.float64)
    batch = ops.rx(thetas)
    assert batch.shape == (7, 2, 2)
    for t, u in zip(thetas, batch):
        assert_close(u, ops.rx(t.item()))
    assert ops.crz(thetas.reshape(7, 1)).shape == (7, 1, 4, 4)
    assert ops.u3(thetas, 0.1, thetas).shape == (7, 2, 2)


def test_dtype_follows_angle_precision():
    assert ops.rx(torch.tensor(0.1, dtype=torch.float32)).dtype == torch.complex64
    assert ops.rx(torch.tensor(0.1, dtype=torch.float64)).dtype == torch.complex128
    assert ops.rx(0.1).dtype == torch.complex128


def test_complex_angle_rejected():
    with pytest.raises(TypeError):
        ops.rx(torch.tensor(0.1 + 0.2j))


def test_gradient_matches_parameter_shift():
    """d<0|RX(t)^† Z RX(t)|0>/dt = -sin(t), equal to the parameter-shift rule."""
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    psi0 = torch.tensor([1, 0], dtype=torch.complex128)

    def expval(t):
        psi = ops.rx(t) @ psi0
        return torch.vdot(psi, ops.Z @ psi).real

    (grad,) = torch.autograd.grad(expval(theta), theta)
    shift = (expval(theta + math.pi / 2) - expval(theta - math.pi / 2)) / 2
    assert grad.item() == pytest.approx(-math.sin(0.7), abs=1e-12)
    assert grad.item() == pytest.approx(shift.item(), abs=1e-12)


@pytest.mark.parametrize("gate", [ops.rx, ops.ry, ops.rz, ops.phase, ops.rzz, ops.crx])
def test_gradcheck(gate):
    theta = torch.tensor([0.3, -1.1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(gate, (theta,))


def test_gradcheck_u3():
    args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.3, -1.2, 2.5))
    assert torch.autograd.gradcheck(ops.u3, args)


# --- kron ---------------------------------------------------------------------
def test_kron_matches_torch_kron():
    torch.testing.assert_close(ops.kron(ops.X, ops.Z), torch.kron(ops.X, ops.Z))
    torch.testing.assert_close(ops.kron(ops.H, ops.CNOT), torch.kron(ops.H, ops.CNOT))


def test_kron_single_and_many():
    assert ops.kron(ops.Y) is ops.Y
    expected = torch.kron(torch.kron(ops.X, ops.Y), ops.Z)
    torch.testing.assert_close(ops.kron(ops.X, ops.Y, ops.Z), expected)


def test_kron_qubit_ordering():
    # X on qubit 0 (most significant) maps |00> -> |10>, i.e. index 0 -> 2
    assert_close(ops.kron(ops.X, ops.I)[:, 0], [0, 0, 1, 0])
    assert_close(ops.kron(ops.I, ops.X)[:, 0], [0, 1, 0, 0])


def test_kron_non_square():
    ket0 = torch.tensor([[1], [0]], dtype=ops.DTYPE)
    ket1 = torch.tensor([[0], [1]], dtype=ops.DTYPE)
    assert_close(ops.kron(ket0, ket1), [[0], [1], [0], [0]])


def test_kron_batched():
    thetas = torch.linspace(0, 2, 5, dtype=torch.float64)
    out = ops.kron(ops.rx(thetas), ops.Z)
    assert out.shape == (5, 4, 4)
    for o, t in zip(out, thetas):
        torch.testing.assert_close(o, torch.kron(ops.rx(t), ops.Z))
    # batch dimensions broadcast between operands
    both = ops.kron(ops.rx(thetas[:, None]), ops.rz(thetas))
    assert both.shape == (5, 5, 4, 4)
    torch.testing.assert_close(both[1, 3], torch.kron(ops.rx(thetas[1]), ops.rz(thetas[3])))


def test_kron_dtype_promotion():
    u = ops.rx(torch.tensor(0.3, dtype=torch.float32))  # complex64
    assert ops.kron(u, u).dtype == torch.complex64
    assert ops.kron(u, ops.X).dtype == torch.complex128


def test_kron_gradcheck():
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: ops.kron(ops.ry(t), ops.rz(2 * t), ops.H), (theta,))


@pytest.mark.parametrize("args", [(), (ops.X, torch.ones(2, dtype=ops.DTYPE))])
def test_kron_invalid(args):
    with pytest.raises(ValueError):
        ops.kron(*args)
