"""Gradient-based optimisation of differentiable parameters."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


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
