"""Quantum gates as PyTorch tensors.

Fixed gates are module-level ``complex128`` constants. Parametric gates are
functions of their angles, built entirely from differentiable torch ops, so
gradients flow from any downstream quantity back to the angles.

Angles may be Python floats or real tensors of any shape. A tensor angle of
shape ``batch`` gives a gate of shape ``(*batch, d, d)``, so many gates can be
built and differentiated at once. The complex dtype follows the angle's
precision (float32 -> complex64, float64 -> complex128); Python floats are
treated as float64.

Qubit ordering: in two-qubit gates the first qubit is the most significant,
i.e. the basis is |00>, |01>, |10>, |11> with |q0 q1>.
"""

from __future__ import annotations

import math

import torch

Angle = float | torch.Tensor

DTYPE = torch.complex128

# --- Fixed single-qubit gates -------------------------------------------------
I = torch.eye(2, dtype=DTYPE)
X = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
Z = torch.tensor([[1, 0], [0, -1]], dtype=DTYPE)
H = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
S = torch.tensor([[1, 0], [0, 1j]], dtype=DTYPE)
T = torch.tensor([[1, 0], [0, complex(math.cos(math.pi / 4), math.sin(math.pi / 4))]], dtype=DTYPE)

# --- Fixed two-qubit gates ----------------------------------------------------
CNOT = torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=DTYPE
)
CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=DTYPE))
SWAP = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=DTYPE
)


# --- Helpers ------------------------------------------------------------------
def _as_angle(theta: Angle) -> torch.Tensor:
    """Convert an angle to a real floating tensor, preserving autograd."""
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float64)
    if theta.is_complex():
        raise TypeError("angles must be real")
    if not theta.is_floating_point():
        theta = theta.to(torch.float64)
    return theta


def _complex_dtype(real: torch.Tensor) -> torch.dtype:
    return torch.complex64 if real.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128


def _expi(x: torch.Tensor) -> torch.Tensor:
    """exp(i x) for real x, as a complex tensor."""
    return torch.polar(torch.ones_like(x), x)


def _matrix2(a, b, c, d) -> torch.Tensor:
    """Stack batched entries into ``(..., 2, 2)`` matrices [[a, b], [c, d]]."""
    a, b, c, d = torch.broadcast_tensors(a, b, c, d)
    return torch.stack([torch.stack([a, b], dim=-1), torch.stack([c, d], dim=-1)], dim=-2)


# --- Parametric gates ---------------------------------------------------------
def pauli_rotation(pauli: torch.Tensor, theta: Angle) -> torch.Tensor:
    """exp(-i theta/2 P) = cos(theta/2) I - i sin(theta/2) P for any Pauli string P.

    ``pauli`` must square to the identity (e.g. X, Z, or kron(Z, Z)).
    """
    theta = _as_angle(theta)
    dtype = _complex_dtype(theta)
    pauli = pauli.to(dtype=dtype, device=theta.device)
    eye = torch.eye(pauli.shape[-1], dtype=dtype, device=theta.device)
    c = torch.cos(theta / 2).to(dtype)[..., None, None]
    s = torch.sin(theta / 2).to(dtype)[..., None, None]
    return c * eye - 1j * s * pauli


def rx(theta: Angle) -> torch.Tensor:
    """Rotation about X: exp(-i theta X / 2)."""
    return pauli_rotation(X, theta)


def ry(theta: Angle) -> torch.Tensor:
    """Rotation about Y: exp(-i theta Y / 2)."""
    return pauli_rotation(Y, theta)


def rz(theta: Angle) -> torch.Tensor:
    """Rotation about Z: exp(-i theta Z / 2)."""
    return pauli_rotation(Z, theta)


def rxx(theta: Angle) -> torch.Tensor:
    """Two-qubit XX rotation: exp(-i theta X⊗X / 2)."""
    return pauli_rotation(torch.kron(X, X), theta)


def ryy(theta: Angle) -> torch.Tensor:
    """Two-qubit YY rotation: exp(-i theta Y⊗Y / 2)."""
    return pauli_rotation(torch.kron(Y, Y), theta)


def rzz(theta: Angle) -> torch.Tensor:
    """Two-qubit ZZ rotation: exp(-i theta Z⊗Z / 2)."""
    return pauli_rotation(torch.kron(Z, Z), theta)


def phase(phi: Angle) -> torch.Tensor:
    """Phase gate diag(1, e^{i phi})."""
    phi = _as_angle(phi)
    one = torch.ones_like(phi).to(_complex_dtype(phi))
    zero = torch.zeros_like(one)
    return _matrix2(one, zero, zero, _expi(phi))


def u3(theta: Angle, phi: Angle, lam: Angle) -> torch.Tensor:
    """General single-qubit unitary (OpenQASM U gate).

    U(theta, phi, lam) = [[cos(theta/2),          -e^{i lam} sin(theta/2)],
                          [e^{i phi} sin(theta/2), e^{i(phi+lam)} cos(theta/2)]]
    """
    theta, phi, lam = torch.broadcast_tensors(_as_angle(theta), _as_angle(phi), _as_angle(lam))
    dtype = _complex_dtype(theta)
    c = torch.cos(theta / 2).to(dtype)
    s = torch.sin(theta / 2).to(dtype)
    return _matrix2(c, -_expi(lam) * s, _expi(phi) * s, _expi(phi + lam) * c)


def controlled(u: torch.Tensor) -> torch.Tensor:
    """Controlled-U with the control as the first (most significant) qubit.

    Returns the block-diagonal matrix [[I, 0], [0, U]]; batch dimensions of
    ``u`` are preserved.
    """
    d = u.shape[-1]
    out = torch.zeros(*u.shape[:-2], 2 * d, 2 * d, dtype=u.dtype, device=u.device)
    out[..., :d, :d] = torch.eye(d, dtype=u.dtype, device=u.device)
    out[..., d:, d:] = u
    return out


def crx(theta: Angle) -> torch.Tensor:
    """Controlled RX."""
    return controlled(rx(theta))


def cry(theta: Angle) -> torch.Tensor:
    """Controlled RY."""
    return controlled(ry(theta))


def crz(theta: Angle) -> torch.Tensor:
    """Controlled RZ."""
    return controlled(rz(theta))
