import math

import pytest
import torch

from hornbill import StateVector, gates
from hornbill.estimation import GridPosterior, choose_setting, sample_outcomes


def ramsey(gamma, t):
    """P(0) = cos^2(gamma t), P(1) = sin^2(gamma t), batched over gamma."""
    p0 = torch.cos(torch.as_tensor(gamma, dtype=torch.float64) * t) ** 2
    return torch.stack([p0, 1 - p0], dim=-1)


GRID = torch.linspace(0, math.pi / 2, 2001, dtype=torch.float64)


# --- sample_outcomes ------------------------------------------------------------
def test_sample_outcomes_frequencies():
    gen = torch.Generator().manual_seed(0)
    probs = torch.tensor([0.1, 0.0, 0.6, 0.3], dtype=torch.float64)
    shots = sample_outcomes(probs, 100_000, generator=gen)
    assert shots.shape == (100_000,) and shots.dtype == torch.int64
    freqs = torch.bincount(shots, minlength=4).double() / shots.numel()
    torch.testing.assert_close(freqs, probs, atol=0.01, rtol=0)


def test_sample_outcomes_from_state_vector():
    sv = StateVector(2).apply(gates.H, [0]).apply(gates.CNOT, [0, 1])
    shots = sample_outcomes(sv.probabilities(), 1000, generator=torch.Generator().manual_seed(1))
    assert set(shots.tolist()) <= {0, 3}


@pytest.mark.parametrize("probs, shots", [(torch.ones(2, 2), 1), (torch.ones(2), 0)])
def test_sample_outcomes_rejects_bad_input(probs, shots):
    with pytest.raises(ValueError):
        sample_outcomes(probs, shots)


# --- GridPosterior --------------------------------------------------------------
def test_uniform_prior():
    post = GridPosterior(torch.linspace(0, 1, 101, dtype=torch.float64))
    torch.testing.assert_close(post.weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(post.mean(), torch.tensor(0.5, dtype=torch.float64))
    # variance of n evenly spaced points with spacing h is h^2 (n^2 - 1) / 12
    torch.testing.assert_close(post.variance(), torch.tensor((101**2 - 1) / 12 / 100**2, dtype=torch.float64))


def test_custom_prior_is_normalised():
    post = GridPosterior(torch.tensor([0.0, 1.0, 2.0]), prior=torch.tensor([1.0, 3.0, 0.0]))
    torch.testing.assert_close(post.weights, torch.tensor([0.25, 0.75, 0.0], dtype=torch.float64))
    assert post.map() == 1.0


def test_update_matches_bayes_rule():
    grid = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    likelihood = torch.stack([grid, 1 - grid], dim=-1)  # P(0 | theta) = theta
    post = GridPosterior(grid).update(likelihood, torch.tensor([0, 0, 1]))
    expected = grid**2 * (1 - grid)
    torch.testing.assert_close(post.weights, expected / expected.sum())


def test_update_order_and_batching_do_not_matter():
    outcomes = torch.tensor([0, 1, 1, 0, 1])
    lik = ramsey(GRID, 1.0)
    batched = GridPosterior(GRID).update(lik, outcomes)
    one_by_one = GridPosterior(GRID)
    for k in outcomes.flip(0):
        one_by_one.update(lik, int(k))
    torch.testing.assert_close(batched.weights, one_by_one.weights)


def test_impossible_outcome_keeps_posterior_finite():
    grid = torch.tensor([0.0, 1.0], dtype=torch.float64)
    likelihood = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    post = GridPosterior(grid).update(likelihood, 1)
    assert torch.isfinite(post.weights).all()
    torch.testing.assert_close(post.weights.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_many_shots_do_not_underflow():
    gen = torch.Generator().manual_seed(2)
    shots = sample_outcomes(ramsey(torch.tensor(0.7), 1.0), 100_000, generator=gen)
    post = GridPosterior(GRID).update(ramsey(GRID, 1.0), shots)
    assert torch.isfinite(post.weights).all()


def test_posterior_concentrates_on_true_value():
    gamma = 0.7
    gen = torch.Generator().manual_seed(3)
    post = GridPosterior(GRID)
    for n in [100, 10_000]:
        shots = sample_outcomes(ramsey(torch.tensor(gamma), 1.0), n, generator=gen)
        post.update(ramsey(GRID, 1.0), shots)
    # Fisher information per shot for P0 = cos^2(gamma t) is 4 t^2, so the std is ~1 / (2 sqrt(N))
    n_total = 10_100
    assert post.std() == pytest.approx(1 / (2 * math.sqrt(n_total)), rel=0.2)
    assert abs(post.mean() - gamma) < 4 * post.std()
    assert abs(post.map() - gamma) < 4 * post.std()


def test_state_vector_likelihood_matches_formula():
    """The batched gate formula used as a model agrees with the StateVector circuit."""
    for gamma in [0.1, 0.7, 1.3]:
        sv = StateVector(1).apply(gates.H, [0]).apply(gates.rz(2 * gamma), [0]).apply(gates.H, [0])
        torch.testing.assert_close(sv.probabilities(), ramsey(torch.tensor(gamma), 1.0))


@pytest.mark.parametrize(
    "grid, prior",
    [
        (torch.zeros(2, 2), None),
        (torch.tensor([]), None),
        (torch.zeros(3), torch.ones(2)),
        (torch.zeros(3), torch.tensor([1.0, -1.0, 1.0])),
        (torch.zeros(3), torch.zeros(3)),
    ],
)
def test_grid_posterior_rejects_bad_input(grid, prior):
    with pytest.raises(ValueError):
        GridPosterior(grid, prior)


def test_update_rejects_bad_likelihood_and_outcomes():
    post = GridPosterior(GRID)
    with pytest.raises(ValueError):
        post.update(torch.ones(3, 2), 0)
    with pytest.raises(ValueError):
        post.update(ramsey(GRID, 1.0), 2)


# --- expected_variance / choose_setting -----------------------------------------
def test_expected_variance_brute_force():
    gen = torch.Generator().manual_seed(4)
    post = GridPosterior(GRID).update(ramsey(GRID, 1.0), sample_outcomes(ramsey(torch.tensor(0.4), 1.0), 20, generator=gen))
    lik = ramsey(GRID, 2.5)
    predictive = post.weights @ lik
    brute = 0.0
    for k in range(2):
        updated = GridPosterior(GRID, prior=post.weights).update(lik, k)
        brute += predictive[k] * updated.variance()
    torch.testing.assert_close(post.expected_variance(lik), brute)


def test_uninformative_measurement_does_not_reduce_variance():
    post = GridPosterior(GRID)
    flat = torch.full((GRID.numel(), 2), 0.5, dtype=torch.float64)
    torch.testing.assert_close(post.expected_variance(flat), post.variance())


def test_expected_variance_never_exceeds_current_variance():
    post = GridPosterior(GRID)
    for t in [0.1, 1.0, 3.0, 10.0]:
        assert post.expected_variance(ramsey(GRID, t)) <= post.variance() + 1e-12


def test_choose_setting_avoids_useless_time():
    post = GridPosterior(GRID)
    # t = 0 gives P(0) = 1 for every gamma, so it carries no information
    best, expected = choose_setting(post, ramsey, [0.0, 0.5, 1.0])
    assert best != 0.0
    assert expected.shape == (3,)
    assert int(torch.argmin(expected)) == [0.0, 0.5, 1.0].index(best)


def test_adaptive_beats_fixed_time():
    gamma, rounds = torch.tensor(0.9), 60
    times = [0.5 * 2**j for j in range(6)]
    fixed_std, adaptive_std = [], []
    for seed in range(5):
        gen = torch.Generator().manual_seed(seed)
        fixed, adaptive = GridPosterior(GRID), GridPosterior(GRID)
        for _ in range(rounds):
            fixed.update(ramsey(GRID, 1.0), sample_outcomes(ramsey(gamma, 1.0), 1, generator=gen))
            t, _ = choose_setting(adaptive, ramsey, times)
            adaptive.update(ramsey(GRID, t), sample_outcomes(ramsey(gamma, t), 1, generator=gen))
        fixed_std.append(fixed.std())
        adaptive_std.append(adaptive.std())
        assert abs(adaptive.mean() - gamma) < 5 * adaptive.std()
    assert torch.stack(adaptive_std).median() < 0.5 * torch.stack(fixed_std).median()


def test_choose_setting_rejects_empty():
    with pytest.raises(ValueError):
        choose_setting(GridPosterior(GRID), ramsey, [])
