"""hornbill: a differentiable quantum simulator built on PyTorch."""

from hornbill import operators
from hornbill.estimation import GridPosterior, choose_setting, sample_outcomes
from hornbill.evolution import time_evolution
from hornbill.hamiltonians import RydbergHamiltonian
from hornbill.metrics import unitary_distance, unitary_infidelity
from hornbill.optimize import gradient_descent, pulse_gradient_descent
from hornbill.pulses import Pulse
from hornbill.states import StateVector, zero_state

__all__ = [
    "GridPosterior",
    "Pulse",
    "RydbergHamiltonian",
    "StateVector",
    "choose_setting",
    "gradient_descent",
    "operators",
    "pulse_gradient_descent",
    "sample_outcomes",
    "time_evolution",
    "unitary_distance",
    "unitary_infidelity",
    "zero_state",
]
__version__ = "0.1.0"
