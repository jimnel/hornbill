"""Distances between unitary operators.

Two unitaries that differ only by a global phase, ``U`` and ``e^{i phi} U``,
describe the same physical operation, so both functions here ignore global
phase. Both are built from the normalised Hilbert-Schmidt overlap

    |Tr(U^dagger V)| / d,

which equals 1 exactly when ``U`` and ``V`` agree up to a global phase.

All functions accept batches: inputs of shape ``(..., d, d)`` broadcast
against each other and return a real tensor of the broadcast batch shape.
"""

from __future__ import annotations

import torch


def _overlap(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Tr(U^dagger V) / d for batches of square matrices, without forming U^dagger V."""
    if u.ndim < 2 or v.ndim < 2:
        raise ValueError("inputs must be matrices of shape (..., d, d)")
    d = u.shape[-1]
    if u.shape[-2] != d or v.shape[-2:] != (d, d):
        raise ValueError(
            f"inputs must be square matrices of the same size, got {tuple(u.shape)} and {tuple(v.shape)}"
        )
    dtype = torch.promote_types(torch.promote_types(u.dtype, v.dtype), torch.complex64)
    u, v = u.to(dtype), v.to(dtype)
    # Tr(U^dagger V) = sum_ij conj(U_ij) V_ij
    return torch.einsum("...ij,...ij->...", u.conj(), v) / d


def unitary_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Phase-invariant distance between two unitaries.

    D(U, V) = sqrt(1 - |Tr(U^dagger V)| / d)
            = min_phi ||U - e^{i phi} V||_F / sqrt(2 d)

    This is the Frobenius distance minimised over a global phase, normalised so
    that D lies in [0, 1]. It is a true metric on unitaries modulo global phase:
    D(U, V) = 0 if and only if U = e^{i phi} V. D = 1 when U^dagger V is
    traceless (e.g. X and Z).

    Note:
        The square root makes the gradient undefined at U = V (the minimum),
        so for optimisation use :func:`unitary_infidelity`, which is smooth.

    Args:
        u, v: Unitary matrices of shape ``(..., d, d)``.

    Returns:
        Real tensor of the broadcast batch shape.
    """
    overlap = _overlap(u, v).abs()
    # clamp: rounding can push the overlap fractionally above 1
    return torch.sqrt(torch.clamp(1 - overlap, min=0))


def unitary_infidelity(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Smooth, phase-invariant loss between two unitaries.

    1 - |Tr(U^dagger V) / d|^2

    Zero if and only if U = e^{i phi} V, in [0, 1], and differentiable
    everywhere (with zero gradient at the minimum), so it is suited as a loss
    for gradient-based gate fitting. Related to :func:`unitary_distance` by
    ``infidelity = D^2 (2 - D^2)``.

    Args:
        u, v: Unitary matrices of shape ``(..., d, d)``.

    Returns:
        Real tensor of the broadcast batch shape.
    """
    overlap = _overlap(u, v)
    return 1 - (overlap.real**2 + overlap.imag**2)
