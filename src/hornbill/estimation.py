"""Bayesian estimation of a single parameter from measurement outcomes.

The unknown parameter ``theta`` (e.g. a field strength gamma) is represented by
a posterior distribution on a fixed grid of candidate values. A measurement
model gives the probability of each outcome at every grid point,

    likelihood[g, k] = P(outcome k | theta = grid[g]),

and Bayes' rule multiplies the posterior by the likelihood of the outcomes
actually observed. Weights are stored as logarithms so that many updates do
not underflow.

Adaptive estimation picks each measurement setting (e.g. the evolution time
``t``) to minimise the *expected* posterior variance after the next shot; see
:func:`choose_setting`.

Example (Ramsey estimation of gamma, with P(0) = cos^2(gamma t))::

    def model(gamma, t):
        p0 = torch.cos(gamma * t) ** 2
        return torch.stack([p0, 1 - p0], dim=-1)

    posterior = GridPosterior(torch.linspace(0, math.pi / 2, 1000))
    shots = sample_outcomes(model(torch.tensor([0.7]), 1.0)[0], 100)
    posterior.update(model(posterior.grid, 1.0), shots)
    posterior.mean(), posterior.std()
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def sample_outcomes(
    probabilities: torch.Tensor,
    shots: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``shots`` independent outcomes from a probability vector.

    Args:
        probabilities: Non-negative 1-D tensor of outcome probabilities, e.g.
            :meth:`hornbill.StateVector.probabilities`. It is normalised before
            sampling, so small rounding errors are harmless.
        shots: Number of outcomes to draw (positive).
        generator: Optional random generator, for reproducible samples.

    Returns:
        A 1-D ``int64`` tensor of ``shots`` outcome indices.
    """
    if probabilities.ndim != 1:
        raise ValueError(f"probabilities must be 1-D, got shape {tuple(probabilities.shape)}")
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    probs = probabilities.detach().clamp_min(0)
    return torch.multinomial(probs / probs.sum(), shots, replacement=True, generator=generator)


class GridPosterior:
    """Posterior distribution of a scalar parameter on a fixed grid.

    Args:
        grid: 1-D real tensor of candidate parameter values. It should cover
            every value the parameter could plausibly take; the posterior is
            zero outside it.
        prior: Optional 1-D tensor of non-negative prior weights, one per grid
            point (normalised automatically). Defaults to uniform.
    """

    def __init__(self, grid: torch.Tensor, prior: torch.Tensor | None = None) -> None:
        if grid.ndim != 1 or grid.numel() == 0:
            raise ValueError(f"grid must be a non-empty 1-D tensor, got shape {tuple(grid.shape)}")
        if grid.is_complex():
            raise TypeError("grid must be real")
        self.grid = grid.detach().to(torch.float64)

        if prior is None:
            log_weights = torch.zeros_like(self.grid)
        else:
            if prior.shape != self.grid.shape:
                raise ValueError(f"prior must have shape {tuple(self.grid.shape)}, got {tuple(prior.shape)}")
            if (prior < 0).any() or not (prior > 0).any():
                raise ValueError("prior weights must be non-negative and not all zero")
            log_weights = torch.log(prior.detach().to(torch.float64))
        self.log_weights = log_weights - torch.logsumexp(log_weights, dim=0)

    @property
    def weights(self) -> torch.Tensor:
        """Normalised posterior probability of each grid point."""
        return torch.exp(self.log_weights)

    def update(self, likelihood: torch.Tensor, outcomes: int | torch.Tensor) -> GridPosterior:
        """Condition on observed outcomes from one measurement setting.

        Args:
            likelihood: Tensor of shape ``(len(grid), n_outcomes)`` with
                ``likelihood[g, k] = P(k | grid[g])`` for the setting used.
            outcomes: A single outcome index, or a 1-D tensor of outcome
                indices (one per shot, all taken with this setting), e.g. the
                output of :func:`sample_outcomes`.

        Returns:
            ``self``, so calls can be chained.
        """
        likelihood = self._check_likelihood(likelihood)
        counts = torch.bincount(
            torch.as_tensor(outcomes, dtype=torch.int64).reshape(-1),
            minlength=likelihood.shape[1],
        )
        if counts.numel() != likelihood.shape[1]:
            raise ValueError(f"outcome indices must be in [0, {likelihood.shape[1]})")

        self.log_weights = self.log_weights + _safe_log(likelihood) @ counts.to(torch.float64)
        self.log_weights = self.log_weights - torch.logsumexp(self.log_weights, dim=0)
        return self

    def mean(self) -> torch.Tensor:
        """Posterior mean, the estimate minimising expected squared error."""
        return self.weights @ self.grid

    def variance(self) -> torch.Tensor:
        """Posterior variance."""
        return self.weights @ (self.grid - self.mean()) ** 2

    def std(self) -> torch.Tensor:
        """Posterior standard deviation."""
        return self.variance().sqrt()

    def map(self) -> torch.Tensor:
        """Maximum a posteriori estimate (the most probable grid point)."""
        return self.grid[torch.argmax(self.log_weights)]

    def expected_variance(self, likelihood: torch.Tensor) -> torch.Tensor:
        """Posterior variance expected after one more shot with this setting.

        Averages the variance of each possible updated posterior over the
        predicted probability of the outcome that produces it. Lower is better.

        Args:
            likelihood: Tensor of shape ``(len(grid), n_outcomes)``, as for
                :meth:`update`.
        """
        likelihood = self._check_likelihood(likelihood)
        joint = self.weights[:, None] * likelihood  # P(theta_g, k)
        predictive = joint.sum(dim=0)  # P(k)
        # Var after outcome k, weighted by P(k):
        #   P(k) Var_k = sum_g P(theta_g, k) theta_g^2 - (sum_g P(theta_g, k) theta_g)^2 / P(k)
        first = self.grid @ joint
        second = (self.grid**2) @ joint
        observed = predictive > 0
        return (second - torch.where(observed, first**2 / predictive.where(observed, 1.0), 0)).sum()

    def _check_likelihood(self, likelihood: torch.Tensor) -> torch.Tensor:
        if likelihood.ndim != 2 or likelihood.shape[0] != self.grid.numel():
            raise ValueError(
                f"likelihood must have shape ({self.grid.numel()}, n_outcomes), got {tuple(likelihood.shape)}"
            )
        return likelihood.detach().to(torch.float64)

    def __repr__(self) -> str:
        return f"GridPosterior(points={self.grid.numel()}, mean={self.mean().item():.6g}, std={self.std().item():.3g})"


def choose_setting(
    posterior: GridPosterior,
    model: Callable[[torch.Tensor, Any], torch.Tensor],
    settings: Sequence[Any],
) -> tuple[Any, torch.Tensor]:
    """Pick the measurement setting with the lowest expected posterior variance.

    Args:
        posterior: The current posterior.
        model: ``model(grid, setting)`` returning the likelihood matrix of
            shape ``(len(grid), n_outcomes)`` for that setting.
        settings: Candidate settings to choose from, e.g. evolution times.

    Returns:
        ``(best_setting, expected_variances)``, where ``expected_variances[i]``
        is the expected posterior variance for ``settings[i]``.
    """
    if len(settings) == 0:
        raise ValueError("settings must not be empty")
    expected = torch.stack([posterior.expected_variance(model(posterior.grid, s)) for s in settings])
    return settings[int(torch.argmin(expected))], expected


def _safe_log(p: torch.Tensor) -> torch.Tensor:
    """log(p) with zeros mapped to a large negative number instead of -inf.

    Keeps the posterior normalisable if an outcome the model deems impossible
    is observed (e.g. due to rounding).
    """
    return torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))
