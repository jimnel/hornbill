"""Quantum state vectors as PyTorch tensors."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def zero_state(
    n_qubits: int,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the n-qubit computational basis state |00...0>.

    Args:
        n_qubits: Number of qubits (non-negative).
        dtype: Complex dtype of the returned vector.
        device: Device to allocate the vector on.

    Returns:
        A vector of length ``2**n_qubits`` with amplitude 1 at index 0 and 0
        elsewhere.
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
        raise TypeError(f"n_qubits must be an int, got {type(n_qubits).__name__}")
    if n_qubits < 0:
        raise ValueError(f"n_qubits must be non-negative, got {n_qubits}")

    psi = torch.zeros(2**n_qubits, dtype=dtype, device=device)
    psi[0] = 1
    return psi


class StateVector:
    """The state of an n-qubit register, stored as a vector of 2**n amplitudes.

    The register starts in |00...0>. Operators are applied with :meth:`apply`,
    which replaces the stored state with a new tensor rather than modifying it
    in place, so gradients flow from any later quantity (e.g.
    :meth:`probabilities`) back to the parameters of the gates applied.

    Qubit ordering matches :mod:`hornbill.gates`: qubit 0 is the most
    significant bit, so basis state index ``i`` is the binary string
    ``q0 q1 ... q_{n-1}``.

    Example:
        >>> from hornbill import StateVector, gates
        >>> sv = StateVector(2)
        >>> sv.apply(gates.H, [0]).apply(gates.CNOT, [0, 1])  # Bell state
        >>> sv.probabilities()  # [0.5, 0, 0, 0.5]
    """

    def __init__(
        self,
        n_qubits: int,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.state = zero_state(n_qubits, dtype=dtype, device=device)

    def apply(self, operator: torch.Tensor, qubits: Sequence[int] | None = None) -> StateVector:
        """Apply an operator to the register, updating the stored state.

        Args:
            operator: A ``(2**k, 2**k)`` matrix acting on ``k`` qubits.
            qubits: The ``k`` qubits the operator acts on, in the order of the
                operator's own qubits (e.g. ``[control, target]`` for CNOT).
                If ``None``, the operator must act on the whole register
                (``k == n_qubits``).

        Returns:
            ``self``, so calls can be chained.
        """
        qubits = list(range(self.n_qubits)) if qubits is None else list(qubits)
        k = len(qubits)

        if len(set(qubits)) != k:
            raise ValueError(f"qubits must be distinct, got {qubits}")
        if any(not 0 <= q < self.n_qubits for q in qubits):
            raise ValueError(f"qubits must be in [0, {self.n_qubits}), got {qubits}")
        if operator.shape != (2**k, 2**k):
            raise ValueError(
                f"operator on {k} qubit(s) must have shape {(2**k, 2**k)}, got {tuple(operator.shape)}"
            )

        operator = operator.to(dtype=self.state.dtype, device=self.state.device)

        if k == self.n_qubits and qubits == list(range(k)):
            self.state = operator @ self.state
            return self

        # View the state as an n-index tensor, contract the operator's input
        # indices with the target qubits, then move the output indices back.
        psi = self.state.reshape([2] * self.n_qubits)
        op = operator.reshape([2] * (2 * k))
        psi = torch.tensordot(op, psi, dims=(list(range(k, 2 * k)), qubits))
        psi = psi.movedim(list(range(k)), qubits)
        self.state = psi.reshape(-1)
        return self

    def probabilities(self) -> torch.Tensor:
        """Probability of measuring each computational basis state, |amplitude|**2."""
        # real^2 + imag^2 rather than abs()**2: same value, but smooth at zero
        # amplitude, where abs() has an undefined derivative.
        return self.state.real**2 + self.state.imag**2

    def __repr__(self) -> str:
        return f"StateVector(n_qubits={self.n_qubits}, dtype={self.state.dtype})"
