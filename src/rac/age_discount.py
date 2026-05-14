"""Age discount w_age(Δ) for RAC corrections."""
from __future__ import annotations

import math
import numpy as np


def exp_age_discount(delta: int | np.ndarray, tau_age: float = 50.0) -> float | np.ndarray:
    """w_age(Δ) = exp(−Δ/τ_age). Default choice per Proposal 2 §4."""
    if isinstance(delta, np.ndarray):
        return np.exp(-delta / tau_age)
    return math.exp(-delta / tau_age)


def adaptive_tau_age(
    recent_delays: np.ndarray,
    kappa: float = 1.0,
    q: float = 0.95,
) -> float:
    """τ_age(t) = κ · q̂_q(Δ; t). Rank-2 second-contribution per main-paper case.

    Tracks the q-th percentile of the observed delay distribution. Heavy-tail
    robust per wave-36 bursty-delay finding.
    """
    if len(recent_delays) < 5:
        return 50.0  # default
    return float(kappa * np.quantile(recent_delays, q))
