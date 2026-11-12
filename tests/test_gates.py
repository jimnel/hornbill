import math

import pytest
import torch

from hornbill import gates as g


def assert_close(a, b, atol=1e-12):
    torch.testing.assert_close(a, torch.as_tensor(b, dtype=a.dtype), atol=atol, rtol=0)


def is_unitary(u):
    eye = torch.eye(u.shape[-1], dtype=u.dtype).expand_as(u)
    return torch.allclose(u @ u.mH, eye, atol=1e-12)


FIXED = [g.I, g.X, g.Y, g.Z, g.H, g.S, g.T, g.CNOT, g.CZ, g.SWAP]
PARAMETRIC_1 = [g.rx, g.ry, g.rz, g.phase, g.rxx, g.ryy, g.rzz, g.crx, g.cry, g.crz]


@pytest.mark.parametrize("u", FIXED)
def test_fixed_gates_are_unitary(u):
    assert is_unitary(u)


@pytest.mark.parametrize("gate", PARAMETRIC_1)
def test_parametric_gates_are_unitary(gate):
    assert is_unitary(gate(0.734))


def test_u3_is_unitary():
    assert is_unitary(g.u3(0.3, -1.2, 2.5))


def test_pauli_algebra():
    assert_close(g.X @ g.Y, 1j * g.Z)
    assert_close(g.H @ g.Z @ g.H, g.X)
    assert_close(g.S @ g.S, g.Z)
    assert_close(g.T @ g.T, g.S)


def test_rotations_at_pi():
    assert_close(g.rx(math.pi), -1j * g.X)
    assert_close(g.ry(math.pi), -1j * g.Y)
    assert_close(g.rz(math.pi), -1j * g.Z)


def test_rz_and_phase_differ_by_global_phase():
    phi = 0.81
    assert_close(g.rz(phi), complex(math.cos(-phi / 2), math.sin(-phi / 2)) * g.phase(phi))


def test_rzz_is_diagonal():
    theta = 0.4
    m, p = complex(math.cos(theta / 2), -math.sin(theta / 2)), complex(math.cos(theta / 2), math.sin(theta / 2))
    assert_close(g.rzz(theta), torch.diag(torch.tensor([m, p, p, m], dtype=torch.complex128)))


def test_u3_special_cases():
    theta, phi, lam = 0.3, -1.2, 2.5
    assert_close(g.u3(theta, 0.0, 0.0), g.ry(theta))
    assert_close(g.u3(0.0, 0.0, lam), g.phase(lam))
    assert_close(g.u3(math.pi / 2, 0.0, math.pi), g.H)
    # U3 = e^{i(phi+lam)/2} RZ(phi) RY(theta) RZ(lam)
    gp = complex(math.cos((phi + lam) / 2), math.sin((phi + lam) / 2))
    assert_close(g.u3(theta, phi, lam), gp * g.rz(phi) @ g.ry(theta) @ g.rz(lam))


def test_controlled():
    assert_close(g.controlled(g.X), g.CNOT)
    assert_close(g.controlled(g.Z), g.CZ)


def test_batched_angles():
    thetas = torch.linspace(0, 2 * math.pi, 7, dtype=torch.float64)
    batch = g.rx(thetas)
    assert batch.shape == (7, 2, 2)
    for t, u in zip(thetas, batch):
        assert_close(u, g.rx(t.item()))
    assert g.crz(thetas.reshape(7, 1)).shape == (7, 1, 4, 4)
    assert g.u3(thetas, 0.1, thetas).shape == (7, 2, 2)


def test_dtype_follows_angle_precision():
    assert g.rx(torch.tensor(0.1, dtype=torch.float32)).dtype == torch.complex64
    assert g.rx(torch.tensor(0.1, dtype=torch.float64)).dtype == torch.complex128
    assert g.rx(0.1).dtype == torch.complex128


def test_complex_angle_rejected():
    with pytest.raises(TypeError):
        g.rx(torch.tensor(0.1 + 0.2j))


def test_gradient_matches_parameter_shift():
    """d<0|RX(t)^† Z RX(t)|0>/dt = -sin(t), equal to the parameter-shift rule."""
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    psi0 = torch.tensor([1, 0], dtype=torch.complex128)

    def expval(t):
        psi = g.rx(t) @ psi0
        return torch.vdot(psi, g.Z @ psi).real

    (grad,) = torch.autograd.grad(expval(theta), theta)
    shift = (expval(theta + math.pi / 2) - expval(theta - math.pi / 2)) / 2
    assert grad.item() == pytest.approx(-math.sin(0.7), abs=1e-12)
    assert grad.item() == pytest.approx(shift.item(), abs=1e-12)


@pytest.mark.parametrize("gate", [g.rx, g.ry, g.rz, g.phase, g.rzz, g.crx])
def test_gradcheck(gate):
    theta = torch.tensor([0.3, -1.1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(gate, (theta,))


def test_gradcheck_u3():
    args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.3, -1.2, 2.5))
    assert torch.autograd.gradcheck(g.u3, args)
