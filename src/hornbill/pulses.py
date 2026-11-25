"""Piecewise-constant control pulses (units with hbar = 1)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from hornbill.evolution import time_evolution


class Pulse:
    """A piecewise-constant pulse on ``n_controls`` control channels.

    The total ``time`` is split into ``n`` equal steps of length ``dt``. During
    step ``j`` the Hamiltonian is

        H_j = drift + sum_k params[j, k] * controls[k]

    so ``params`` (shape ``(n, n_controls)``) holds the control amplitudes. It
    is a leaf tensor with ``requires_grad=True``, ready for
    :func:`hornbill.gradient_descent` or any ``torch.optim`` optimiser.

    Example:
        >>> from hornbill import Pulse, operators as ops
        >>> pulse = Pulse(n=50, time=1.0)
        >>> u = pulse.unitary([ops.X, ops.Y], drift=0.5 * ops.Z)

    Args:
        n: Number of time steps.
        time: Total pulse duration.
        n_controls: Number of control channels (columns of ``params``).
        params: Initial amplitudes of shape ``(n, n_controls)``. Copied into a
            new leaf tensor. If omitted, drawn uniformly from ``[0, 1)``.
        dtype: Real dtype of ``params`` when they are drawn at random.
        generator: Optional ``torch.Generator`` for the random initialisation.
    """

    def __init__(
        self,
        n: int,
        time: float,
        n_controls: int = 2,
        params: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        generator: torch.Generator | None = None,
    ):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if time <= 0:
            raise ValueError(f"time must be positive, got {time}")

        if params is None:
            params = torch.rand((n, n_controls), dtype=dtype, generator=generator)
        else:
            params = torch.as_tensor(params).detach().clone()
            if params.is_complex() or not params.is_floating_point():
                raise TypeError("params must be a real floating-point tensor")
            if params.shape != (n, n_controls):
                raise ValueError(f"params must have shape ({n}, {n_controls}), got {tuple(params.shape)}")

        self.n = n
        self.time = time
        self.n_controls = n_controls
        self.dt = time / n
        self.params = params.requires_grad_(True)

    def __repr__(self) -> str:
        return f"Pulse(n={self.n}, time={self.time}, n_controls={self.n_controls})"

    @property
    def times(self) -> torch.Tensor:
        """Start time of each step, shape ``(n,)``."""
        return torch.arange(self.n, dtype=self.params.dtype) * self.dt

    def hamiltonians(
        self, controls: Sequence[torch.Tensor], drift: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Hamiltonian during each step, shape ``(n, d, d)``.

        Args:
            controls: ``n_controls`` operators of shape ``(d, d)``, one per channel.
            drift: Optional always-on Hamiltonian of shape ``(d, d)``.
        """
        if len(controls) != self.n_controls:
            raise ValueError(f"expected {self.n_controls} control operators, got {len(controls)}")
        ops = torch.stack(list(controls))
        if ops.ndim != 3 or ops.shape[-1] != ops.shape[-2]:
            raise ValueError(f"control operators must all have shape (d, d), got {tuple(ops.shape[1:])}")

        dtype = torch.promote_types(torch.promote_types(ops.dtype, self.params.dtype), torch.complex64)
        h = torch.einsum("nk,kij->nij", self.params.to(dtype), ops.to(dtype))
        if drift is not None:
            if drift.shape != ops.shape[1:]:
                raise ValueError(f"drift must have shape {tuple(ops.shape[1:])}, got {tuple(drift.shape)}")
            h = h + drift.to(dtype)
        return h

    def step_unitaries(
        self, controls: Sequence[torch.Tensor], drift: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Unitary of each step, exp(-i H_j dt), shape ``(n, d, d)``."""
        return time_evolution(self.hamiltonians(controls, drift), self.dt)

    def unitary(self, controls: Sequence[torch.Tensor], drift: torch.Tensor | None = None) -> torch.Tensor:
        """Total unitary U = U_{n-1} ... U_1 U_0 of the pulse, shape ``(d, d)``."""
        steps = self.step_unitaries(controls, drift)
        u = steps[0]
        for step in steps[1:]:
            u = step @ u
        return u
