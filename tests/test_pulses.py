import pytest
import torch

from hornbill import Pulse, gradient_descent, time_evolution, unitary_distance, unitary_infidelity
from hornbill import operators as ops


def test_attributes():
    pulse = Pulse(n=10, time=2.0)
    assert pulse.dt == pytest.approx(0.2)
    assert pulse.params.shape == (10, 2)
    assert pulse.params.is_leaf and pulse.params.requires_grad
    assert pulse.params.dtype == torch.float64
    torch.testing.assert_close(pulse.times, torch.arange(10, dtype=torch.float64) * 0.2)


def test_random_init_reproducible():
    a = Pulse(5, 1.0, generator=torch.Generator().manual_seed(0))
    b = Pulse(5, 1.0, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(a.params, b.params)
    assert ((a.params >= 0) & (a.params < 1)).all()


def test_given_params_are_copied():
    init = torch.ones(4, 3)
    pulse = Pulse(4, 1.0, n_controls=3, params=init)
    assert pulse.params is not init and not init.requires_grad
    assert pulse.params.dtype == torch.float32


def test_hamiltonians():
    params = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    pulse = Pulse(2, 1.0, params=params)
    h = pulse.hamiltonians([ops.X, ops.Y], drift=ops.Z)
    torch.testing.assert_close(h[0], ops.X + ops.Z)
    torch.testing.assert_close(h[1], 2 * ops.Y + ops.Z)


def test_constant_pulse_matches_single_evolution():
    pulse = Pulse(7, 1.3, params=torch.full((7, 2), 0.4, dtype=torch.float64))
    h = 0.4 * ops.X + 0.4 * ops.Y + 0.2 * ops.Z
    torch.testing.assert_close(pulse.unitary([ops.X, ops.Y], drift=0.2 * ops.Z), time_evolution(h, 1.3))


def test_step_ordering():
    # X for the first half, Z for the second: U = exp(-i Z dt) exp(-i X dt)
    params = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    pulse = Pulse(2, 1.0, params=params)
    expected = time_evolution(ops.Z, 0.5) @ time_evolution(ops.X, 0.5)
    torch.testing.assert_close(pulse.unitary([ops.X, ops.Z]), expected)


def test_multi_qubit_controls():
    pulse = Pulse(3, 1.0, n_controls=1)
    u = pulse.unitary([ops.kron(ops.X, ops.X)], drift=ops.kron(ops.Z, ops.I))
    assert u.shape == (4, 4)
    torch.testing.assert_close(u @ u.mH, torch.eye(4, dtype=u.dtype))


def test_gradcheck():
    pulse = Pulse(4, 1.0, generator=torch.Generator().manual_seed(1))

    def unitary(params):
        pulse.params = params  # gradcheck perturbs its own copy, so route it through the pulse
        return pulse.unitary([ops.X, ops.Y], drift=ops.Z)

    assert torch.autograd.gradcheck(unitary, (pulse.params.detach().clone().requires_grad_(),))


def test_gradients_reach_params():
    pulse = Pulse(5, 1.0, generator=torch.Generator().manual_seed(2))
    loss = unitary_infidelity(pulse.unitary([ops.X, ops.Y]), ops.X)
    (grad,) = torch.autograd.grad(loss, pulse.params)
    assert grad.shape == (5, 2) and grad.abs().sum() > 0


def test_optimise_to_hadamard():
    pulse = Pulse(10, 1.0, generator=torch.Generator().manual_seed(0))
    controls, drift = [ops.X, ops.Y], 0.5 * ops.Z
    gradient_descent(
        lambda p: unitary_infidelity(pulse.unitary(controls, drift), ops.H), [pulse.params], lr=1.0, steps=500
    )
    assert unitary_distance(pulse.unitary(controls, drift), ops.H).item() < 1e-3


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=0, time=1.0),
        dict(n=3, time=0.0),
        dict(n=3, time=1.0, params=torch.zeros(2, 2)),
        dict(n=3, time=1.0, params=torch.zeros(3, 2, dtype=torch.complex128)),
    ],
)
def test_invalid_construction(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Pulse(**kwargs)


def test_invalid_controls():
    pulse = Pulse(3, 1.0)
    with pytest.raises(ValueError):
        pulse.unitary([ops.X])  # wrong number of controls
    with pytest.raises(ValueError):
        pulse.unitary([ops.X, ops.Y], drift=ops.CNOT)  # wrong drift size


def test_plot():
    import matplotlib

    matplotlib.use("Agg")
    pulse = Pulse(n=4, time=2.0, n_controls=3)
    ax = pulse.plot(title="pulse")

    assert ax.get_title() == "pulse"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["control 0", "control 1", "control 2"]
    assert len(ax.patches) == 3
    assert ax.patches[0].get_data().edges[-1] == pytest.approx(2.0)


def test_plot_label_count_checked():
    with pytest.raises(ValueError, match="labels"):
        Pulse(n=4, time=1.0).plot(labels=["only one"])
