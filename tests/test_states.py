import pytest
import torch

from hornbill.states import zero_state


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_zero_state(n):
    psi = zero_state(n)
    assert psi.shape == (2**n,)
    assert psi.dtype == torch.complex128
    assert psi[0] == 1
    assert torch.count_nonzero(psi) == 1


def test_zero_state_dtype():
    assert zero_state(2, dtype=torch.complex64).dtype == torch.complex64


@pytest.mark.parametrize("n", [-1, 2.0, True])
def test_zero_state_rejects_bad_input(n):
    with pytest.raises((TypeError, ValueError)):
        zero_state(n)


# --- StateVector ---------------------------------------------------------------
import math

from hornbill import gates
from hornbill.states import StateVector


def kron_all(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = torch.kron(out, op)
    return out


def random_unitary(dim, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, dim, dtype=torch.complex128, generator=gen)
    q, _ = torch.linalg.qr(a)
    return q.contiguous()  # QR returns column-major storage, which torch.kron rejects


def test_initial_state_is_all_zeros():
    sv = StateVector(3)
    assert torch.equal(sv.state, zero_state(3))
    torch.testing.assert_close(sv.probabilities(), torch.eye(8, dtype=torch.float64)[0])


def test_single_qubit_gate_on_each_qubit():
    for q, expected in [(0, [0.5, 0, 0.5, 0]), (1, [0.5, 0.5, 0, 0])]:
        sv = StateVector(2).apply(gates.H, [q])
        torch.testing.assert_close(sv.probabilities(), torch.tensor(expected, dtype=torch.float64))


def test_bell_state():
    sv = StateVector(2).apply(gates.H, [0]).apply(gates.CNOT, [0, 1])
    torch.testing.assert_close(sv.probabilities(), torch.tensor([0.5, 0, 0, 0.5], dtype=torch.float64))


def test_qubit_order_of_two_qubit_gate():
    # X on qubit 1 then CNOT with control 1, target 0: |00> -> |01> -> |11>
    sv = StateVector(2).apply(gates.X, [1]).apply(gates.CNOT, [1, 0])
    torch.testing.assert_close(sv.probabilities(), torch.tensor([0, 0, 0, 1], dtype=torch.float64))


def test_full_register_operator():
    u = random_unitary(8, seed=0)
    sv = StateVector(3).apply(u)
    torch.testing.assert_close(sv.state, u[:, 0])


@pytest.mark.parametrize("qubits", [[0], [2], [1, 3], [3, 0], [2, 0, 3]])
def test_local_apply_matches_full_matrix(qubits):
    """Applying U to some qubits equals applying the permuted full-register matrix."""
    n, k = 4, len(qubits)
    u = random_unitary(2**k, seed=k)
    psi0 = random_unitary(2**n, seed=99)[:, 0]

    sv = StateVector(n)
    sv.state = psi0
    sv.apply(u, qubits)

    # Build the full operator: U on qubits [0..k) then permute qubits into place.
    full = kron_all(u, torch.eye(2 ** (n - k), dtype=u.dtype)).reshape([2] * (2 * n))
    rest = [q for q in range(n) if q not in qubits]
    order = qubits + rest
    inv = [order.index(i) for i in range(n)]
    full = full.permute(inv + [n + i for i in inv]).reshape(2**n, 2**n)

    torch.testing.assert_close(sv.state, full @ psi0)


def test_probabilities_sum_to_one():
    sv = StateVector(3)
    for q, t in enumerate([0.3, 1.1, -0.7]):
        sv.apply(gates.ry(t), [q])
    sv.apply(gates.CNOT, [0, 2])
    assert sv.probabilities().sum().item() == pytest.approx(1.0, abs=1e-12)


def test_gradients_flow_through_apply():
    """P(|1>) after RY(t) is sin^2(t/2), so dP/dt = sin(t)/2."""
    theta = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    sv = StateVector(2).apply(gates.ry(theta), [1])
    p1 = sv.probabilities()[1]
    (grad,) = torch.autograd.grad(p1, theta)
    assert grad.item() == pytest.approx(math.sin(0.9) / 2, abs=1e-12)


def test_gradcheck_through_circuit():
    def circuit(a, b):
        sv = StateVector(2).apply(gates.rx(a), [0]).apply(gates.CNOT, [0, 1]).apply(gates.ry(b), [1])
        return sv.probabilities()

    args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.4, -1.3))
    assert torch.autograd.gradcheck(circuit, args)


@pytest.mark.parametrize(
    "op, qubits",
    [
        (gates.X, [2]),          # out of range
        (gates.CNOT, [0, 0]),    # repeated qubit
        (gates.CNOT, [0]),       # wrong size for number of qubits
        (gates.X, None),         # single-qubit operator on whole register
    ],
)
def test_apply_rejects_bad_input(op, qubits):
    with pytest.raises(ValueError):
        StateVector(2).apply(op, qubits)
