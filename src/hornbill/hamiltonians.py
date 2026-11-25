"""Model Hamiltonians (units with hbar = 1)."""

from __future__ import annotations

import torch

from hornbill import operators as ops

# Projector onto the Rydberg state |1>: n = |1><1| = (I - Z) / 2
_N = (ops.I - ops.Z) / 2


class RydbergHamiltonian:
    """Two neutral atoms driven to a Rydberg state, with van der Waals blockade.

        H(Ω, Δ) = Ω/2 (X⊗I + I⊗X) − Δ (n⊗I + I⊗n) + V n⊗n,   V = C6 / r^6

    where ``n = |1><1|`` and ``|1>`` is the Rydberg state. Ω (Rabi frequency)
    and Δ (detuning) are the controls; the interaction ``V n⊗n`` is always on.

    The control and drift operators are exposed as :attr:`controls` (in the
    order ``[rabi, detuning]``) and :attr:`drift`, so the Hamiltonian plugs
    straight into a :class:`~hornbill.Pulse` with two channels:

        >>> from hornbill import Pulse, RydbergHamiltonian
        >>> ham = RydbergHamiltonian(distance=1.0)
        >>> u = Pulse(n=50, time=1.0).unitary(ham.controls, ham.drift)

    Args:
        distance: Separation ``r`` between the atoms. May be a real tensor
            (e.g. with ``requires_grad=True``) to differentiate through it.
        c6: Van der Waals coefficient ``C6``.
    """

    rabi_op = 0.5 * (ops.kron(ops.X, ops.I) + ops.kron(ops.I, ops.X))
    detuning_op = -(ops.kron(_N, ops.I) + ops.kron(ops.I, _N))

    def __init__(self, distance: float | torch.Tensor = 1.0, c6: float | torch.Tensor = 1.0):
        self.distance = distance
        self.c6 = c6

    def __repr__(self) -> str:
        return f"RydbergHamiltonian(distance={self.distance}, c6={self.c6})"

    @property
    def interaction(self) -> torch.Tensor:
        """Interaction strength V = C6 / r^6."""
        return torch.as_tensor(self.c6, dtype=torch.float64) / torch.as_tensor(self.distance, dtype=torch.float64) ** 6

    @property
    def drift(self) -> torch.Tensor:
        """Always-on interaction term ``V n⊗n``, shape ``(4, 4)``."""
        return self.interaction.to(ops.DTYPE) * ops.kron(_N, _N)

    @property
    def controls(self) -> list[torch.Tensor]:
        """Control operators ``[rabi_op, detuning_op]``, matching Pulse parameter columns."""
        return [self.rabi_op, self.detuning_op]

    def __call__(self, rabi: float | torch.Tensor, detuning: float | torch.Tensor) -> torch.Tensor:
        """H(Ω, Δ). Tensor arguments broadcast, giving shape ``(*batch, 4, 4)``."""
        rabi = torch.as_tensor(rabi, dtype=torch.float64)
        detuning = torch.as_tensor(detuning, dtype=torch.float64)
        rabi, detuning = torch.broadcast_tensors(rabi, detuning)
        return (
            rabi.to(ops.DTYPE)[..., None, None] * self.rabi_op
            + detuning.to(ops.DTYPE)[..., None, None] * self.detuning_op
            + self.drift
        )
