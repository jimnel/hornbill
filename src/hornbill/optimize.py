"""Gradient-based optimisation of differentiable parameters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from hornbill.pulses import Pulse


def gradient_descent(
    loss_fn: Callable[..., torch.Tensor],
    params: Sequence[torch.Tensor],
    lr: float = 0.1,
    steps: int = 100,
    tol: float | None = None,
) -> list[float]:
    """Minimise ``loss_fn(*params)`` with plain gradient descent.

    Each step computes the loss, back-propagates with autograd and applies
    ``p <- p - lr * dL/dp`` to every parameter in place.

    Args:
        loss_fn: Callable taking the parameters and returning a real scalar tensor.
        params: Leaf tensors with ``requires_grad=True``. Updated in place.
        lr: Learning rate (step size).
        steps: Maximum number of iterations.
        tol: If given, stop early once the change in loss between steps is
            below ``tol``.

    Returns:
        The loss value at each step.
    """
    params = list(params)
    for p in params:
        if not p.requires_grad:
            raise ValueError("all params must have requires_grad=True")

    history: list[float] = []
    for _ in range(steps):
        loss = loss_fn(*params)
        if loss.ndim != 0 or loss.is_complex():
            raise ValueError("loss_fn must return a real scalar tensor")

        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p, g in zip(params, grads):
                p -= lr * g

        history.append(loss.item())
        if tol is not None and len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break

    return history


def pulse_gradient_descent(
    loss_fn: Callable[[Pulse], torch.Tensor],
    pulse: Pulse,
    lr: float = 0.1,
    steps: int = 100,
    tol: float | None = None,
) -> list[float]:
    """Minimise ``loss_fn(pulse)`` over the pulse amplitudes with gradient descent.

    Same algorithm as :func:`gradient_descent`, applied to ``pulse.params``:
    each step computes ``loss_fn(pulse)``, back-propagates and applies
    ``params <- params - lr * dL/dparams`` in place.

    Example:
        >>> from hornbill import Pulse, operators as ops, unitary_infidelity
        >>> pulse = Pulse(n=20, time=1.0)
        >>> history = pulse_gradient_descent(
        ...     lambda p: unitary_infidelity(p.unitary([ops.X, ops.Y]), ops.X), pulse
        ... )

    Args:
        loss_fn: Callable taking the :class:`~hornbill.Pulse` and returning a
            real scalar tensor.
        pulse: The pulse to optimise. Its ``params`` are updated in place.
        lr: Learning rate (step size).
        steps: Maximum number of iterations.
        tol: If given, stop early once the change in loss between steps is
            below ``tol``.

    Returns:
        The loss value at each step.
    """
    return gradient_descent(lambda _: loss_fn(pulse), [pulse.params], lr=lr, steps=steps, tol=tol)
